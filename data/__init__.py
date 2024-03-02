"""
DataModule
Author: Daehyeon Gi <spliter2157@gmail.com>
"""

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from .datasets import (
    TripletDataset,
    PatentDataset,
    STSDataset
)


class DataModule(LightningDataModule):
    """
    SimCSE DataModule
    - Train: KorNLI
    - Val & Test: KorSTS
    """
    def __init__(
        self,
        hparams: dict,
        train_dataset: Dataset = None,
        val_dataset: Dataset = None,
        test_dataset: Dataset = None) -> None:
        super().__init__()
        self.hparams.update(hparams)
        if train_dataset is None:
            self.train_dataset = PatentDataset(
                phase="train"
            )
        if val_dataset is None:
            self.val_dataset = PatentDataset(
                phase="validation"
            )
        self.test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.hparams["num_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams["batch_size"] * 4,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams["num_workers"],
        )
