import torch
import torch.nn as nn
import pytorch_lightning as pl

from dataloader import MRIDataset
from torch.utils.data import DataLoader
from torchvision import models
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, AUROC, F1Score


class MRINet(pl.LightningModule):
    def __init__(self,
                 n_classes: int,
                 plane: str,
                 transformer: dict,
                 lr: float):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features=num_filters, out_features=n_classes)

        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.plane = plane
        self.transformer = transformer
        self.lr = lr

        self.train_acc = Accuracy()
        self.train_f1 = F1Score(number_classes=3, average="micro")
        self.train_auroc = AUROC(number_classes=3, average="micro")
        self.val_acc = Accuracy()
        self.val_f1 = F1Score(number_classes=3, average="micro")
        self.val_auroc = AUROC(number_classes=3, average="micro")

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        return self.classifier(representations)

    def train_dataloader(self):
        train_ds = MRIDataset('/content/drive/MyDrive/',
                              self.plane,
                              transform=self.transformer["train"],
                              train=True)

        train_dl = DataLoader(train_ds,
                              batch_size=1,
                              shuffle=True)
        return train_dl

    def val_dataloader(self):
        val_ds = MRIDataset('/content/drive/MyDrive/',
                            self.plane,
                            transform=self.transformer["valid"],
                            train=False)
        val_dl = DataLoader(val_ds,
                            batch_size=1,
                            shuffle=False)
        return val_dl

    def training_step(self, batch, batch_idx):
        image = torch.squeeze(batch["image"], dim=0)
        target = batch["labels"]
        weights = torch.squeeze(batch["weights"], dim=0)

        prediction = self(image.float())

        target = target.repeat(prediction.shape[0], 1)

        batch_loss = self.loss(prediction, target)
        batch_loss = (batch_loss * weights).mean()

        prediction = nn.Sigmoid()(prediction)
        target = target.type(torch.int64)

        acc = self.train_acc(prediction, target)
        f1 = self.train_f1(prediction, target)

        self.train_auroc.update(prediction, target)
        self.log("train_loss", batch_loss)
        self.log("train_accuracy", acc)
        self.log("train_f1", f1)

        return batch_loss

    def validation_step(self, batch, batch_idx):
        image = torch.squeeze(batch["image"], dim=0)
        target = batch["labels"]
        weights = torch.squeeze(batch["weights"], dim=0)

        prediction = self(image.float())

        target = target.repeat(prediction.shape[0], 1)

        batch_loss = self.loss(prediction, target)
        batch_loss = (batch_loss * weights).mean()

        prediction = nn.Sigmoid()(prediction)
        target = target.type(torch.int64)

        self.val_acc.update(prediction, target)
        self.val_f1.update(prediction, target)
        self.val_auroc.update(prediction, target)

        return batch_loss

    def training_epoch_end(self, training_step_outputs):
        train_accuracy = self.train_acc.compute()
        train_f1 = self.train_f1.compute()
        train_auroc = self.train_auroc.compute()

        self.log("epoch_train_accuracy", train_accuracy)
        self.log("epoch_train_f1", train_f1)

        self.train_acc.reset()
        self.train_f1.reset()
        print(f"\ntraining accuracy: {train_accuracy:.4}, " \
              f"f1: {train_f1:.4}, auroc: {train_auroc:.4}")

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = torch.tensor(validation_step_outputs).mean()
        val_accuracy = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        val_auroc = self.val_auroc.compute()

        self.log("val_accuracy", val_accuracy)
        self.log("val_loss", val_loss)
        self.log("val_f1", val_f1)
        self.log("val_auroc", val_auroc)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        print(f"\nvalidation accuracy: {val_accuracy:.4} " \
              f"f1: {val_f1:.4}, auroc: {val_auroc:.4}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}
