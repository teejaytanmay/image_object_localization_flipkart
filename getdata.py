# coding: utf-8

from PIL import Image
import numpy as np
import pickle
import pandas as pd

def Normalize(image,mean,std):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image

id_to_data={}
id_to_size={}

imgs = pd.read_csv("/media/teejay/TJ HDD2/data/training.csv")
images = imgs.image_name
for i in range(64):
    path=images[i]
    image=Image.open("/media/teejay/TJ HDD2/data/images/"+path).convert('RGB')
    id_to_size[i]=np.array(image,dtype=np.float32).shape[0:2]
    # image=image.resize((224,224))
    image=np.array(image,dtype=np.float32)
    # image=image/255
    # image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
    id_to_data[i]=image
l=list(id_to_data.values())
m=list(id_to_size.values())
id_to_data=np.array(l)
id_to_size=np.array(m)
f=open("/media/teejay/TJ HDD2/data/id_to_data","wb+")
pickle.dump(id_to_data,f)
f=open("/media/teejay/TJ HDD2/data/id_to_size","wb+")
pickle.dump(id_to_size,f)
# id_to_box={}
# with open("./data/images.txt") as f:
#     lines=f.read().splitlines()
#     for line in lines:
#         id,path=line.split(" ",1)
#         image=Image.open("./data/images/"+path).convert('RGB')
#         id_to_size[int(id)]=np.array(image,dtype=np.float32).shape[0:2]
#         image=image.resize((224,224))
#         image=np.array(image,dtype=np.float32)
#         image=image/255
#         image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
#         id_to_data[int(id)]=image

# id_to_data=np.array(list(id_to_data.values()))
# id_to_size=np.array(list(id_to_size.values()))
# f=open("./id_to_data","wb+")
# pickle.dump(id_to_data,f,protocol=4)
# f=open("./id_to_size","wb+")
# pickle.dump(id_to_size,f,protocol=4)

id_to_box={}
# print (id_to_size.shape[0])
# for i in range(id_to_size.shape[0]):
imgs.x1 = imgs.x1/id_to_size[1][1]
imgs.x2 = imgs.x2/id_to_size[0][0]
imgs.y1 = imgs.y1/id_to_size[1][1]
imgs.y2 = imgs.y2/id_to_size[0][0]

for i in range(id_to_size.shape[0]):
    id_to_box[i] = np.array([imgs.x1[i],imgs.x2[i],imgs.y1[i],imgs.y2[i]])

# imgs.head(5)
# with open("./data/bounding_boxes.txt") as f:
#     lines=f.read().splitlines()
#     for line in lines:
#         id,box=line.split(" ",1)
#         box=np.array([float(i) for i in box.split(" ")],dtype=np.float32)
#         box[0]=box[0]/id_to_size[int(id)-1][1]*224
#         box[1]=box[1]/id_to_size[int(id)-1][0]*224
#         box[2]=box[2]/id_to_size[int(id)-1][1]*224
#         box[3]=box[3]/id_to_size[int(id)-1][0]*224
#         id_to_box[int(id)]=box
n=list(id_to_box.values())
id_to_box=np.array(n)
f=open("/media/teejay/TJ HDD2/data/id_to_box","wb+")
pickle.dump(id_to_box,f)
# id_to_box=np.array(list(id_to_box.values()))
# f=open("./id_to_box","wb+")
# pickle.dump(id_to_box,f,protocol=4)
