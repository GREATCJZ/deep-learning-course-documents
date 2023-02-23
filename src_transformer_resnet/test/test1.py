from argparse import Namespace
from typing import List, Optional
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import hydra
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
from image_to_latex.data import Im2Latex
from image_to_latex.lit_models import LitResNetTransformer

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    datamodule = Im2Latex(**cfg.data)
    datamodule.setup("fit")

    lit_model = LitResNetTransformer(**cfg.lit_model)
    a=torch.tensor([[1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3]])
    print(list(a[0].numpy()))
    decoded = lit_model.tokenizer.decode(list(a[0].numpy()))  # type: ignore
    print(decoded)


if __name__ == '__main__':
    main()