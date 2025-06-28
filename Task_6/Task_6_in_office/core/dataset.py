from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize as tv_resize
from pycocotools.coco import COCO
import numpy as np

class SimpleCocoDataset(Dataset):
    def __init__(self, data_dir, base_classes, out_classes, resize=None):
        self.dir = Path(data_dir)
        self.coco = COCO(str(self.dir / "annotations/instances_default.json"))
        self.img_ids = self.coco.getImgIds()
        self.base_classes = base_classes
        self.out_classes = out_classes
        self.resize = resize

        self.cat2base = {}
        names = {c["name"]: c["id"] for c in base_classes}
        for c in self.coco.loadCats(self.coco.getCatIds()):
            if c["name"] in names:
                self.cat2base[c["id"]] = names[c["name"]]

    def __len__(self):
        return len(self.img_ids)

    def _to_out(self, base_mask):
        h, w = base_mask.shape[-2:]
        out = torch.zeros(len(self.out_classes) + 1, h, w)
        out[0] = 1
        for oc in self.out_classes:
            idx = oc["id"]
            for s in oc["summable_masks"]:
                out[idx][base_mask[s] == 1] = 1
            for s in oc["subtractive_masks"]:
                out[idx][base_mask[s] == 1] = 0
            out[0][out[idx] == 1] = 0
        return out

    def __getitem__(self, i):
        # --- информация о файле -------------------------------------------------
        img_id = self.img_ids[i]
        info = self.coco.loadImgs(img_id)[0]
        img_path = self.dir / "images" / info["file_name"]
        img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)  # shape (H, W)


        h, w = img.shape

        # --- базовые маски ------------------------------------------------------
        base_mask = torch.zeros(len(self.base_classes) + 1, h, w)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
            cid = ann["category_id"]
            if cid in self.cat2base:
                m = torch.tensor(self.coco.annToMask(ann))
                base_mask[self.cat2base[cid]][m == 1] = 1

        mask = self._to_out(base_mask)

        # --- нормализованное изображение ---------------------------------------
        img = torch.tensor(img).unsqueeze(0).float()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # --- ресайз (если задан) ------------------------------------------------
        if self.resize:
            img = tv_resize(img, self.resize, interpolation=2)  # bilinear
            mask = tv_resize(mask, self.resize, interpolation=0)  # nearest

        # --- метки и площади ----------------------------------------------------
        labels = mask.amax((-1, -2))
        values = mask.sum((-1, -2))

        return {"images": img,
                "masks": mask.long(),
                "labels": labels,
                "values": values}

