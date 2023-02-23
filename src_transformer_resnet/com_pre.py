difference=[]
with open("./scripts/test_ids.txt",'r')as f:
    imagename=f.readlines()

with open("./result_hqy.txt",'r') as f:
    with open("./result(resnet50).txt",'r') as f2:
        txt1=f.readlines()
        txt2=f2.readlines()
        for i in range(0,len(txt1)):
            if txt1[i] != txt2[i]:
                difference.append(imagename[i]+txt1[i]+txt2[i])
         
with open("./compare.txt",'w') as f:
    for i in difference:
        f.write(i)