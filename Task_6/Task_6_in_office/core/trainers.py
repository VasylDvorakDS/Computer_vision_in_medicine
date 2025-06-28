import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import List, Dict


def safe_div(numerator, denominator):
    return numerator / (denominator + 1e-8)


# ───────────── КЛАССИФИКАЦИЯ ─────────────
class SimpleClassificationTrainer:
    def __init__(self, model: nn.Module,
                 classes: List[Dict],
                 train_loader,
                 val_loader,
                 device,
                 loss_fn = nn.BCEWithLogitsLoss(),
                 epochs=10,
                 exp_name="classification",
                 log_dir="runs",
                 lr=1e-4):

        self.model = model.to(device)
        self.classes = classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        tag = torch.randint(0, 99999, ()).item()
        self.exp_name = f"{tag}_{exp_name}"
        self.writer = SummaryWriter(os.path.join(log_dir, self.exp_name))
        self.best_loss = float("inf")

    def _step(self, x, y):
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        return loss, torch.sigmoid(logits)

    def _run_epoch(self, loader, phase):
        self.model.train() if phase == "train" else self.model.eval()
        stats = {"tp": 0, "fp": 0, "fn": 0}
        total_loss = 0

        with torch.set_grad_enabled(phase == "train"):
            for batch in tqdm(loader, desc=phase.upper()):
                x = batch["images"].to(self.device)
                y = batch["labels"][:, 1:].float().to(self.device)

                loss, probs = self._step(x, y)
                total_loss += loss.item()

                if phase == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                bin_preds = (probs > 0.5).float()
                stats["tp"] += (bin_preds * y).sum(0).cpu()
                stats["fp"] += (bin_preds * (1 - y)).sum(0).cpu()
                stats["fn"] += ((1 - bin_preds) * y).sum(0).cpu()

        recall = safe_div(stats["tp"], stats["tp"] + stats["fn"])
        precision = safe_div(stats["tp"], stats["tp"] + stats["fp"])
        f1 = safe_div(2 * precision * recall, precision + recall)
        return total_loss / len(loader), f1

    def train(self):
        for epoch in range(self.epochs):
            tr_loss, tr_f1 = self._run_epoch(self.train_loader, "train")
            val_loss, val_f1 = self._run_epoch(self.val_loader, "val")

            self.writer.add_scalars("Loss", {"train": tr_loss, "val": val_loss}, epoch)
            for i, cl in enumerate(self.classes):
                self.writer.add_scalars(f"F1/{cl['name']}", {
                    "train": tr_f1[i],
                    "val": val_f1[i]
                }, epoch)

            print(f"Epoch {epoch + 1}/{self.epochs} | "
                  f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")
            for i, cl in enumerate(self.classes):
                print(f"  {cl['name']}: F1 train/val = {tr_f1[i]:.3f}/{val_f1[i]:.3f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(),
                           f"save_models/{self.exp_name}_best.pth")

        torch.save(self.model.state_dict(),
                   f"save_models/{self.exp_name}_final.pth")


# ───────────── СЕГМЕНТАЦИЯ ─────────────
class SimpleSegmentationTrainer:
    def __init__(self, model: nn.Module,
                 classes: List[Dict],
                 train_loader,
                 val_loader,
                 device,
                 loss_fn = nn.BCEWithLogitsLoss(),
                 epochs=10,
                 exp_name="segmentation",
                 log_dir="runs",
                 lr=1e-4):

        self.model = model.to(device)
        self.classes = classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        tag = torch.randint(0, 99999, ()).item()
        self.exp_name = f"{tag}_{exp_name}"
        self.writer = SummaryWriter(os.path.join(log_dir, self.exp_name))
        self.best_iou_loss = float("inf")

    def _step(self, x, y):
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        return loss, torch.sigmoid(logits)

    def _run_epoch(self, loader, phase):
        self.model.train() if phase == "train" else self.model.eval()
        total_loss = 0
        intersection = torch.zeros(len(self.classes))
        union = torch.zeros(len(self.classes))

        with torch.set_grad_enabled(phase == "train"):
            for batch in tqdm(loader, desc=phase.upper()):
                x = batch["images"].to(self.device)
                y = batch["masks"][:, 1:].float().to(self.device)

                loss, probs = self._step(x, y)
                total_loss += loss.item()

                if phase == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                preds = (probs > 0.5).float()
                intersection += (preds * y).sum((0, 2, 3)).cpu()
                union += (preds + y - preds * y).sum((0, 2, 3)).cpu()

        iou = safe_div(intersection, union)
        return total_loss / len(loader), iou

    def train(self):
        for epoch in range(self.epochs):
            tr_loss, tr_iou = self._run_epoch(self.train_loader, "train")
            val_loss, val_iou = self._run_epoch(self.val_loader, "val")

            self.writer.add_scalars("Loss", {"train": tr_loss, "val": val_loss}, epoch)
            for i, cl in enumerate(self.classes):
                self.writer.add_scalars(f"IoU/{cl['name']}", {
                    "train": tr_iou[i],
                    "val": val_iou[i]
                }, epoch)

            print(f"Epoch {epoch + 1}/{self.epochs} | "
                  f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")
            for i, cl in enumerate(self.classes):
                print(f"  {cl['name']}: IoU train/val = {tr_iou[i]:.3f}/{val_iou[i]:.3f}")

            iou_loss = 1.0 - val_iou.mean().item()
            if iou_loss < self.best_iou_loss:
                self.best_iou_loss = iou_loss
                torch.save(self.model.state_dict(),
                           f"save_models/{self.exp_name}_best.pth")

        torch.save(self.model.state_dict(),
                   f"save_models/{self.exp_name}_final.pth")


# ───────────── РЕГРЕССИЯ ─────────────
class SimpleRegressionTrainer:
    def __init__(self, model: nn.Module,
                 classes: List[Dict],
                 train_loader,
                 val_loader,
                 device,
                 loss_fn = nn.MSELoss(),
                 epochs=10,
                 exp_name="regression",
                 log_dir="runs",
                 lr=1e-4):

        self.model = model.to(device)
        self.classes = classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        tag = torch.randint(0, 99999, ()).item()
        self.exp_name = f"{tag}_{exp_name}"
        self.writer = SummaryWriter(os.path.join(log_dir, self.exp_name))
        self.best_mse = float("inf")

    def _step(self, x, y):
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        return loss, preds

    def _run_epoch(self, loader, phase):
        self.model.train() if phase == "train" else self.model.eval()
        total_loss = 0
        squared_error = torch.zeros(len(self.classes))
        count = 0

        with torch.set_grad_enabled(phase == "train"):
            for batch in tqdm(loader, desc=phase.upper()):
                x = batch["images"].to(self.device)
                y = batch["values"][:, 1:].float().to(self.device)

                loss, preds = self._step(x, y)
                total_loss += loss.item()
                count += preds.size(0)

                if phase == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                squared_error += ((preds - y) ** 2).sum(0).cpu()

        mse = squared_error / count
        return total_loss / len(loader), mse

    def train(self):
        for epoch in range(self.epochs):
            tr_loss, tr_mse = self._run_epoch(self.train_loader, "train")
            val_loss, val_mse = self._run_epoch(self.val_loader, "val")

            self.writer.add_scalars("Loss", {"train": tr_loss, "val": val_loss}, epoch)
            for i, cl in enumerate(self.classes):
                self.writer.add_scalars(f"MSE/{cl['name']}", {
                    "train": tr_mse[i],
                    "val": val_mse[i]
                }, epoch)

            print(f"Epoch {epoch + 1}/{self.epochs} | "
                  f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")
            for i, cl in enumerate(self.classes):
                print(f"  {cl['name']}: MSE train/val = {tr_mse[i]:.3f}/{val_mse[i]:.3f}")

            if val_mse.mean().item() < self.best_mse:
                self.best_mse = val_mse.mean().item()
                torch.save(self.model.state_dict(),
                           f"save_models/{self.exp_name}_best.pth")

        torch.save(self.model.state_dict(),
                   f"save_models/{self.exp_name}_final.pth")
