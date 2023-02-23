import sys, os
sys.path.append("/root/autodl-tmp/image-to-latex-main/")

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
from multiprocessing import Process
import threading

# PATH="/hy-tmp/data/formula_images_processed/"
# core=6
# lit_model=0
# def fun1(lit_model,pid):
#     pid=eval(pid)
#     result = []
#     transform = ToTensorV2()
#     with open("/hy-tmp/scripts/test_ids.txt", 'r') as f:
#         txt = f.readlines()
#         length=len(txt)
#         inter_l=int(length/(core-1))*pid
#         inter_r=min(length, int(length/(core-1))*(pid+1))
#         for j in range(inter_l,inter_r):
#             line = txt[j].replace('\n', '')
#             path = PATH + line + ".png"
#             if line == "65147" or line == "78755" or line == "84261" or line == "78755":
#                 result.append("-1\n")
#             else:
#                 image = Image.open(path).convert("L")
#                 image_tensor = transform(image=np.array(image))["image"]
#                 pred = lit_model.model.predict(image_tensor.unsqueeze(0).float())[0].numpy()
#                 decoded = lit_model.tokenizer.decode(list(pred))
#                 decoded_str = " ".join(decoded)
#                 result.append(decoded_str + "\n")
#                 print(decoded_str)
#     with open("/hy-tmp/scripts/result"+str(pid)+".txt", 'w') as f:
#         for i in result:
#             f.write(i)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    global lit_model
    datamodule = Im2Latex(**cfg.data)
    datamodule.setup("fit")

    # lit_model = LitResNetTransformer(**cfg.lit_model)
    lit_model = LitResNetTransformer.load_from_checkpoint("/root/autodl-tmp/image-to-latex-main/artifacts/model.pt")

    callbacks: List[Callback] = []
    if cfg.callbacks.model_checkpoint:
        callbacks.append(ModelCheckpoint(**cfg.callbacks.model_checkpoint))
    if cfg.callbacks.early_stopping:
        callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))
    
    logger: Optional[WandbLogger] = None
    if cfg.logger:
        logger = WandbLogger(**cfg.logger)

    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    if trainer.logger:
        trainer.logger.log_hyperparams(Namespace(**cfg))

    trainer.tune(lit_model, datamodule=datamodule)
    trainer.fit(lit_model, datamodule=datamodule)
    # trainer.test(lit_model, datamodule=datamodule)

    #
    # p0 = Process(target=fun1, args=(lit_model, str(0),))  # 实例化进程对象
    # p1 = Process(target=fun1, args=(lit_model, str(1),))  # 实例化进程对象
    # p2 = Process(target=fun1, args=(lit_model, str(2),))  # 实例化进程对象
    # p3 = Process(target=fun1, args=(lit_model, str(3),))  # 实例化进程对象
    # p4 = Process(target=fun1, args=(lit_model, str(4),))  # 实例化进程对象
    # p5 = Process(target=fun1, args=(lit_model, str(5),))  # 实例化进程对象
    # p0.start()
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p0.join()
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    # p5.join()
    #
    #
    # print("temp")

    #
    # print(image)
    # image_tensor = transform(image=np.array(image))["image"]  # type: ignore
    # print(image_tensor)
    # pred = lit_model.model.predict(image_tensor.unsqueeze(0).float())[0].numpy()
    # print(pred)
    # decoded = lit_model.tokenizer.decode(list(pred))  # type: ignore
    # print(decoded)
    # decoded_str = " ".join(decoded)
    # print(decoded_str)


if __name__ == "__main__":
    main()
