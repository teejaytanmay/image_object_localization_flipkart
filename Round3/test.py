import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import random
import tensorflow as tf
import pandas as pd

plt.switch_backend('agg')

f=open("./id_to_data_test","rb+")
data=pickle.load(f)

f=open("./id_to_box_test","rb+")
box=pickle.load(f)

f=open("./id_to_size_test","rb+")
size=pickle.load(f)

# index=[i for i in range(24045)]
# index=random.sample(index,10)

def smooth_l1_loss(true_box,pred_box):
    loss=0.0
    for i in range(4):
        residual=tf.abs(true_box[:,i]-pred_box[:,i])
        condition=tf.less(residual,1.0)
        small_res=0.5*tf.square(residual)
        large_res=residual-0.5
        loss=loss+tf.where(condition,small_res,large_res)
    return tf.reduce_mean(loss)

def my_metric(labels,predictions):
    threshhold=0.8
    x=predictions[:,0]*8
    x=tf.maximum(tf.minimum(x,80.0),0.0)
    y=predictions[:,1]*8
    y=tf.maximum(tf.minimum(y,60.0),0.0)
    width=predictions[:,2]*8
    width=tf.maximum(tf.minimum(width,80.0),0.0)
    height=predictions[:,3]*8
    height=tf.maximum(tf.minimum(height,60.0),0.0)
    label_x=labels[:,0]
    label_y=labels[:,1]
    label_width=labels[:,2]
    label_height=labels[:,3]
    a1=tf.multiply(width,height)
    a2=tf.multiply(label_width,label_height)
    x1=tf.maximum(x,label_x)
    y1=tf.maximum(y,label_y)
    x2=tf.minimum(x+width,label_x+label_width)
    y2=tf.minimum(y+height,label_y+label_height)
    IoU=tf.abs(tf.multiply((x1-x2),(y1-y2)))/(a1+a2-tf.abs(tf.multiply((x1-x2),(y1-y2))))
    condition=tf.less(threshhold,IoU)
    sum=tf.where(condition,tf.ones(tf.shape(condition)),tf.zeros(tf.shape(condition)))
    return tf.reduce_mean(sum)

model=keras.models.load_model("./model.h5",custom_objects={"smooth_l1_loss":smooth_l1_loss,"my_metric":my_metric})
result=model.predict(data)
index=['x1','x2','y1','y2']
pred = pd.DataFrame(result,columns=index)
# pred.x1 *= 640
# pred.x2 *= 640
# pred.y1 *= 480
# pred.y2 *= 480
# pred['x1'][pred['x1'] < 0] = 0
# pred['x1'][pred['x1'] > 640] = 640
#
# pred['x2'][pred['x2'] < 0] = 0
# pred['x2'][pred['x2'] > 640] = 640
#
# pred['y1'][pred['y1'] < 0] = 0
# pred['y1'][pred['y1'] > 480] = 480
#
# pred['y2'][pred['y2'] < 0] = 0
# pred['y2'][pred['y2'] > 480] = 480
test=pd.read_csv("./test.csv")
sub = pd.concat([test.image_name,pred],axis=1)
print(sub)
sub.to_csv("submission.csv")
# mean=[0.485,0.456,0.406]
# std=[0.229,0.224,0.225]
# j=0
# for i in index:
#     print("Predicting "+str(i)+"th image.")
#     true_box=box[i]
#     image=data[i]
#     prediction=result[j]
#     j+=1
#     for channel in range(3):
#         image[:,:,channel]=image[:,:,channel]*std[channel]+mean[channel]
#
#     # image=image*
#     image=image.astype(np.uint8)
#     # plt.imshow(image)
#
#
#     plt.gca().add_patch(plt.Rectangle((true_box[0],true_box[1]),true_box[2],true_box[3],fill=False,edgecolor='red',linewidth=2,alpha=0.5))
#     plt.gca().add_patch(plt.Rectangle((prediction[0],prediction[1]),prediction[2],prediction[3],fill=False,edgecolor='green',linewidth=2,alpha=0.5))
#     # plt.show()
#     plt.savefig("./prediction/"+str(i)+".png")
#     plt.cla()
