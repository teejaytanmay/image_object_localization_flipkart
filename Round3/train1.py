# coding: utf-8

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, MaxPooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
import utils
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import GeneratorEnqueuer


#data_train,box_train,data_test,box_test=utils.getdata()

# metric function
def my_metric(labels,predictions):
    threshhold=0.9
    x=predictions[:,0]*20
    x=tf.maximum(tf.minimum(x,192.0),0.0)
    y=predictions[:,1]*20
    y=tf.maximum(tf.minimum(y,144.0),0.0)
    width=predictions[:,2]*20
    width=tf.maximum(tf.minimum(width,192.0),0.0)
    height=predictions[:,3]*20
    height=tf.maximum(tf.minimum(height,144.0),0.0)
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

# loss function
def smooth_l1_loss(true_box,pred_box):
    loss=0.0
    for i in range(4):
        residual=tf.abs(true_box[:,i]-pred_box[:,i])
        condition=tf.less(residual,1.0)
        small_res=0.5*tf.square(residual)
        large_res=residual-0.5
        loss=loss+tf.where(condition,small_res,large_res)
    return tf.reduce_mean(loss)

def generator(x):
    y=np.ones(24000)
    return(x,y)

def resnet_block(inputs,num_filters,kernel_size,strides,activation='relu'):
    x=Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(inputs)
    x=BatchNormalization()(x)
    if(activation):
        x=Activation('relu')(x)
    return x


def resnet18():
    inputs=Input((192,144,3))

    # conv1
    x=resnet_block(inputs,64,[7,7],2)

    # conv2
    x=MaxPooling2D([3,3],2,'same')(x)
    for i in range(2):
        a=resnet_block(x,64,[3,3],1)
        b=resnet_block(a,64,[3,3],1,activation=None)
        x=keras.layers.add([x,b])
        x=Activation('relu')(x)

    # conv3
    a=resnet_block(x,128,[1,1],2)
    b=resnet_block(a,128,[3,3],1,activation=None)
    x=Conv2D(128,kernel_size=[1,1],strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(x)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    a=resnet_block(x,128,[3,3],1)
    b=resnet_block(a,128,[3,3],1,activation=None)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    # conv4
    a=resnet_block(x,256,[1,1],2)
    b=resnet_block(a,256,[3,3],1,activation=None)
    x=Conv2D(256,kernel_size=[1,1],strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(x)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    a=resnet_block(x,256,[3,3],1)
    b=resnet_block(a,256,[3,3],1,activation=None)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    # conv5
    a=resnet_block(x,512,[1,1],2)
    b=resnet_block(a,512,[3,3],1,activation=None)
    x=Conv2D(512,kernel_size=[1,1],strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(x)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    a=resnet_block(x,512,[3,3],1)
    b=resnet_block(a,512,[3,3],1,activation=None)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    x=AveragePooling2D(pool_size=1,data_format="channels_last")(x)
    # out:1*1*512

    y=Flatten()(x)
    # out:512
    y=Dense(1000,kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(y)
    outputs=Dense(4,kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(y)

    model=Model(inputs=inputs,outputs=outputs)
    return model

model = resnet18()

train_datagen = ImageDataGenerator(
    samplewise_std_normalization=True,
    rescale=1./255)

test_datagen = ImageDataGenerator(
    samplewise_std_normalization=True,
    rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './images1',
        target_size=(160, 120),
        batch_size=256,
        class_mode=None)

#train_generator = GeneratorEnqueuer(train_generator)

test_generator = train_datagen.flow_from_directory(
        './test1',
        target_size=(160, 120),
        batch_size=256,
        class_mode=None)

model.compile(loss=smooth_l1_loss,optimizer=Adam(),metrics=[my_metric])

model.summary()


def lr_sch(epoch):
    #200 total
    if epoch <50:
        return 1e-3
    if 50<=epoch<100:
        return 1e-4
    if epoch>=100:
        return 1e-5

lr_scheduler=LearningRateScheduler(lr_sch)
lr_reducer=ReduceLROnPlateau(monitor='val_my_metric',factor=0.2,patience=4,mode='max',min_lr=1e-3)

checkpoint=ModelCheckpoint('model.h5',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

model_details=model.fit_generator(
        generator(train_generator),
        steps_per_epoch=100,
        epochs=51,
        callbacks=[lr_scheduler,lr_reducer,checkpoint],
        shuffle=True,
        verbose=1)

model.save('model.h5')

# scores=model.evaluate(data_test,box_test,verbose=1)
# print('Test loss : ',scores[0])
# print('Test accuracy : ',scores[1])

utils.plot_model(model_details)
