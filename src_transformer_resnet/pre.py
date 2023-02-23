from http import HTTPStatus
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from image_to_latex.lit_models import LitResNetTransformer
global lit_model
global transform
import threading

lit_model = LitResNetTransformer.load_from_checkpoint("./artifacts/model.pt")
lit_model.freeze()
transform = ToTensorV2()
PATH="/root/autodl-tmp/image-to-latex-main/data/formula_images_processed/"
core=8
def fun1(pid):
    global lit_model
    pid=eval(pid)
    result = []
    transform = ToTensorV2()
    with open("/root/autodl-tmp/image-to-latex-main/scripts/test_ids.txt", 'r') as f:
        txt = f.readlines()
        length=len(txt)
        inter_l=int(length/(core-1))*pid
        inter_r=min(length, int(length/(core-1))*(pid+1))
        for j in range(inter_l,inter_r):
            line = txt[j].replace('\n', '')
            path = PATH + line + ".png"
            if line == "65147" or line == "78755" or line == "84261" or line == "78755":
                result.append("-1\n")
            else:
                image = Image.open(path).convert("L")
                image_tensor = transform(image=np.array(image))["image"]
                pred = lit_model.model.predict(image_tensor.unsqueeze(0).float())[0].numpy()
                decoded = lit_model.tokenizer.decode(list(pred))
                decoded_str = " ".join(decoded)
                result.append(decoded_str + "\n")
                print(decoded_str)
    with open("/root/autodl-tmp/image-to-latex-main/scripts/result"+str(pid)+".txt", 'w') as f:
        for i in result:
            f.write(i)
            

t0 = threading.Thread(target=fun1, args=("0",))
t1 = threading.Thread(target=fun1, args=("1",))
t2 = threading.Thread(target=fun1, args=("2",))
t3 = threading.Thread(target=fun1, args=("3",))
t4 = threading.Thread(target=fun1, args=("4",))
t5 = threading.Thread(target=fun1, args=("5",))
t6 = threading.Thread(target=fun1, args=("6",))
t7 = threading.Thread(target=fun1, args=("7",))
t0.setDaemon(True)   #把子进程设置为守护线程，必须在start()之前设置
t1.setDaemon(True)
t2.setDaemon(True)
t3.setDaemon(True)
t4.setDaemon(True)
t5.setDaemon(True)
t6.setDaemon(True)
t7.setDaemon(True)
t0.start()
t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
t0.join()
t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
t7.join()

formula_tlist=[]
for i in range(0,core):
    with open("./scripts/result"+str(i)+".txt",'r') as f:
        txt=f.readlines()
        for line in txt:
              line=line.replace("\n",'')
              formula_tlist.append(line)
            
with open("./scripts/result.txt",'w') as f:
    for i in formula_tlist:
        f.write(i+"\n")
        
        
        


# result=[]
# with open("./scripts/test_ids.txt", "r") as f:
#     txt=f.readlines()
#     for temp in txt:
#         temp=temp.strip('\n')
#         image_name = temp + '.png'
#         if image_name=="65147.png" or image_name=="78755.png" or image_name=="84261.png":
#             result.append(" -1")
#         else:
#             file = "./data/formula_images_processed/" + image_name
#             image = Image.open(file).convert("L")
#             image_tensor = transform(image=np.array(image))["image"]  # type: ignore
#             pred = lit_model.model.predict(image_tensor.unsqueeze(0).float())[0]  # type: ignore
#             decoded = lit_model.tokenizer.decode(pred.tolist())  # type: ignore
#             decoded_str = " ".join(decoded)
#             result.append(decoded_str)
#             print(image_name," ",decoded_str)
# with open("./scripts/labels.txt", "w") as fp:
#     for i in result:
#         fp.write(i)
#         fp.write('\n')
