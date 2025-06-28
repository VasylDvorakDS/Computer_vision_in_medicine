# =======================================================================
#  segmentation_exp_seria.py      (запуск:  nohup python segmentation_exp_seria.py &)
#  ──────────────────────────────────────────────────────────────────────
#  • собирает датасет (фон + «Патология»)
#  • считает pixel‑weights   (pos / neg)  и class‑weights
#  • объявляет 6 простых функций‑ошибок
#  • последовательно запускает 6 экспериментов по 50 эпох,
#    каждый со своим loss‑функционалом и именем экспа
# =======================================================================

import os, sys, json, cv2, torch, numpy as np
from pathlib import Path
from torch.utils.data import random_split, DataLoader, ConcatDataset
import torch.nn as nn
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────────────
# 0.  пути и собственные модули
# ───────────────────────────────────────────────────────────────────────
seminar_path = Path("/home/alexskv/seminar_2")
sys.path.insert(0, str(seminar_path / "core"))

from dataset  import SimpleCocoDataset
from trainers import SimpleSegmentationTrainer
from losses import BCELoss, FocalLoss, IoULoss, ComboLoss

device = torch.device("cuda:1")

# ───────────────────────────────────────────────────────────────────────
# 1.  классы и датасет
# ───────────────────────────────────────────────────────────────────────
resize      = (512, 512)
batch_size  = 96
epochs      = 1

pathology_ids = [i for i in range(6, 27) if i != 15]
out_classes   = [{"id": 1, "name": "Патология",
                  "summable_masks": pathology_ids, "subtractive_masks": []}]

base_names = [   # имена базовых классов (фон не нужен)
    "Правое лёгкое", "Левое лёгкое", "Контуры сердца", "Купола диафрагмы и нижележащая область",
    "Сложный случай", "нельзя составить заключение", "Иная патология", "Гидроторакс",
    "Легочно-венозная гипертензия 2 стадии и выше", "Пневмоторакс", "Доброкачественное новообразование",
    "Перелом ребра свежий", "Буллезное вздутие, тонкостенная киста", "Рак лёгкого (включая дорожку к корню при наличии)",
    "Кардиомегалия (отмечается всё сердце, как патология)", "Интерстициальная пневмония.",
    "Метастатическое поражение лёгких", "Полость с уровнем жидкости", "Грыжа пищевого отверстия диафрагмы",
    "Спавшийся сегмент лёгкого при ателектазе", "Инфильтративный туберкулёз",
    "Пневмония. В том числе сегментарная и полисегментарная", "Область распада, деструкции тканей лёгкого",
    "Участок пневмофиброза", "Кальцинаты. Каждый кальцинат выделяется отдельным контуром",
    "Консолидированный перелом ребра"
]
base_classes = [{"id": i + 1, "name": n} for i, n in enumerate(base_names)]

# найти все task‑папки
data_roots = {p.parent.parent for p in (seminar_path / "data")
              .rglob("annotations/instances_default.json")}


datasets = [SimpleCocoDataset(str(d), base_classes, out_classes, resize=resize)
            for d in sorted(data_roots)]
full_ds   = ConcatDataset(datasets)

val_size  = int(0.2 * len(full_ds))
train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

# ───────────────────────────────────────────────────────────────────────
# 2.  ВЕСА
#     • pixel_weights  – для каждого пикселя (pos / neg) класса "патология"
#     • class_weights  – для каждого СНИМКА   (pos / neg) класса "патология"
# ───────────────────────────────────────────────────────────────────────
print("→ считаем pixel‑weights и class‑weights…")

pos_pix = neg_pix = 0          # пиксели
pos_img = neg_img = 0          # снимки

for batch in tqdm(train_loader, total=len(train_loader)):
    m = batch["masks"][:, 1]                 # (B,H,W) патология
    b = m.view(m.size(0), -1).sum(dim=1)     # сумма по каждому изображению

    # пиксели
    pos_pix += m.sum().item()
    neg_pix += (1 - m).sum().item()

    # снимки
    pos_img += (b > 0).sum().item()          # есть патология
    neg_img += (b == 0).sum().item()         # нет патологии

# ---------- pixel weights ----------
tot_pix = pos_pix + neg_pix
w_pos_pix = tot_pix / (2 * pos_pix + 1e-8)
w_neg_pix = tot_pix / (2 * neg_pix + 1e-8)
pixel_weights = torch.tensor([w_pos_pix, w_neg_pix], device=device)

# ---------- class weights ----------
tot_img = pos_img + neg_img
w_pos_cls = tot_img / (2 * pos_img + 1e-8)
w_neg_cls = tot_img / (2 * neg_img + 1e-8)
class_weights = torch.tensor([w_pos_cls, w_neg_cls], device=device)

print(f"Pixel‑weights :  pos={w_pos_pix:.3f}  neg={w_neg_pix:.3f}")
print(f"Class‑weights :  pos={w_pos_cls:.3f}  neg={w_neg_cls:.3f}")


# ───────────────────────────────────────────────────────────────────────
# 2.  Инициализируем разные Loss функции.  
# ───────────────────────────────────────────────────────────────────────

# pixel‑weights и class‑weights посчитаны заранее
bce_loss        = BCELoss()                               
bce_weighted    = BCELoss(pixel_weights)
focal           = FocalLoss()
focal_weighted  = FocalLoss(pixel_weights=pixel_weights)  
iou_loss        = IoULoss()                               
iou_weighted    = IoULoss(class_weight=class_weights)
combo_loss  = ComboLoss(pixel_weights=pixel_weights,
                            class_weight=class_weights) 
combo_loss_weighted  = ComboLoss(pixel_weights=pixel_weights,
                            class_weight=class_weights)   

experiments = [
    ("01_BCE_clean_N", bce_loss),
    ("02_BCE_pixelweight_N", bce_weighted),
    ("03_Focal_clean_N", focal),
    ("04_Focal_pixelweight_N",focal_weighted),
    ("05_IoU_clean_N", iou_loss),
    ("06_IoU_class_weight_N", iou_weighted),
    ("06_Combo_loss_clean_N", combo_loss),
    ("06_Combo_loss_weight_N", combo_loss_weighted)]

# ───────────────────────────────────────────────────────────────────────
# 3.  Цикл экспериментов
# ───────────────────────────────────────────────────────────────────────



for exp_name, loss_fn in experiments:
    print(f"\n================  EXPERIMENT  {exp_name}  ================\n")

    # модель: FPN - SOTA  (in=1, out=1)
    model = smp.FPN(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )

    # обёртка‑trainer
    trainer = SimpleSegmentationTrainer(
        model=model,
        classes=out_classes,          # один класс «Патология»
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        loss_fn=loss_fn,
        epochs=epochs,
        exp_name=exp_name,
    )

    trainer.train()
