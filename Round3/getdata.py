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

id_to_data_test={}
id_to_size_test={}

imgs = pd.read_csv("./training.csv")
images = imgs.image_name
for i in range(24000):
    path=images[i]
    image=Image.open("./images1/images"+path).convert('RGB')
    id_to_size[i]=np.array(image,dtype=np.float32).shape[0:2]
    image=image.resize((120,90))
    image=np.array(image,dtype=np.float32)
    image=image/255
    image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
    id_to_data[i]=image

test = pd.read_csv("./test.csv")
test_imgs = test.image_name
for i in range(24045):
    print (i)
    patht=test_imgs[i]
    image=Image.open("./test1/test"+patht).convert('RGB')
    id_to_size_test[i]=np.array(image,dtype=np.float32).shape[0:2]
    image=image.resize((120,90))
    image=np.array(image,dtype=np.float32)
    image=image/255
    image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
    id_to_data_test[i]=image

    
l=list(id_to_data.values())
m=list(id_to_size.values())
id_to_data=np.array(l)
id_to_size=np.array(m)
f=open("./id_to_data","wb+")
pickle.dump(id_to_data,f)
f=open("./id_to_size","wb+")
pickle.dump(id_to_size,f)

lt=list(id_to_data_test.values())
mt=list(id_to_size_test.values())
id_to_data_test=np.array(lt)
id_to_size_test=np.array(mt)
f=open("./id_to_data_test","wb+")
pickle.dump(id_to_data_test,f)
f=open("./id_to_size_test","wb+")
pickle.dump(id_to_size_test,f)
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
id_to_box_test={}
# print (id_to_size.shape[0])
# for i in range(id_to_size.shape[0]):
imgs.x1 = imgs.x1/id_to_size[1][1]
imgs.x2 = imgs.x2/id_to_size[0][0]
imgs.y1 = imgs.y1/id_to_size[1][1]
imgs.y2 = imgs.y2/id_to_size[0][0]

test.x1 = test.x1/id_to_size_test[1][1]
test.x2 = test.x2/id_to_size_test[0][0]
test.y1 = test.y1/id_to_size_test[1][1]
test.y2 = test.y2/id_to_size_test[0][0]



for i in range(id_to_size.shape[0]):
    id_to_box[i] = np.array([imgs.x1[i],imgs.x2[i],imgs.y1[i],imgs.y2[i]])

for i in range(id_to_size_test.shape[0]):
    id_to_box_test[i] = np.array([test.x1[i],test.x2[i],test.y1[i],test.y2[i]])

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
f=open("./id_to_box","wb+")
pickle.dump(id_to_box,f)

nt=list(id_to_box_test.values())
id_to_box_test=np.array(nt)
f=open("./id_to_box_test","wb+")
pickle.dump(id_to_box_test,f)
# id_to_box=np.array(list(id_to_box.values()))
# f=open("./id_to_box","wb+")
# pickle.dump(id_to_box,f,protocol=4)
