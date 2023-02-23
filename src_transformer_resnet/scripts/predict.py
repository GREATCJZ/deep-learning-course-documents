# import numpy as np
# from albumentations.pytorch.transforms import ToTensorV2
# from PIL import Image
#
# from image_to_latex.lit_models import LitResNetTransformer
#
#
# lit_model = LitResNetTransformer.load_from_checkpoint("../artifacts/model.pt")
# lit_model.freeze()
# transform = ToTensorV2()
#
# print(lit_model)
#
# image = Image.open("../data/formula_images_processed/0.png").convert("L")
# image_tensor = transform(image=np.array(image))["image"]  # type: ignore
# pred = lit_model.model.predict(image_tensor.unsqueeze(0).float())[0]  # type: ignore
# decoded = lit_model.tokenizer.decode(pred.tolist())  # type: ignore

PATH="/hy-tmp/hy-tmp/data/formula_images_processed/"
result=[]
with open("test_ids.txt",'r') as f:
    txt=f.readlines()
    for line in txt:
        line=line.replace('\n','')
        path=PATH+line+".png"
        result.append("0\n")
with open("result.txt",'w') as f:
    for i in result:
        f.write(i)

