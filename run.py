"""
Run Training Script
Author: DaeHyeon Gi <spliter2157@gmail.com>
"""

import os
from pathlib import Path

import yaml
import torch
import wandb
import pytorch_lightning as pl

from model import Model
from data import DataModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"

wandb.login()
torch.set_float32_matmul_precision("medium")

with Path("./config.yaml").open("r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

data_module = DataModule(hparams=config["HP"]["DATAMODULE"])
train_module = Model(hparams=config["HP"]["MODEL"])

# Set Loggers
loggers = []
for key, kwargs in config["loggers"].items():
    loggers.append(getattr(pl.loggers, key)(**kwargs))


# Set Callbacks
callbacks = []
for key, kwargs in config["callbacks"].items():
    callbacks.append(getattr(pl.callbacks, key)(**kwargs))

# Set Trainer
trainer = pl.Trainer(**config["Trainer"], logger=loggers, callbacks=callbacks,)


# LET'S TRAIN!
if __name__ == "__main__":
    trainer.fit(model=train_module, datamodule=data_module)
