# coding: utf-8
from __future__ import print_function
from __future__ import absolute_import

from keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D,
    GlobalAveragePooling2D, ZeroPadding2D, Flatten, Activation, add)
from keras.layers.normalization import BatchNormalization
from keras.models import Model,load_model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file

import warnings
import sys
sys.setrecursionlimit(3000)

import keras
# from keras.layers import Dense, Conv2D, BatchNormalization, Activation
# from keras.layers import AveragePooling2D, MaxPooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
# from keras import backend as K
# from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
import utils
import numpy as np
import tensorflow as tf



data_train,box_train,data_test,box_test=utils.getdata()

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


def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main
            path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(
        nb_filter1,
        (1, 1),
        name=conv_name_base + '2a',
        use_bias=False)(input_tensor)
    x = BatchNormalization(
        epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(
        nb_filter2,
        (kernel_size, kernel_size),
        name=conv_name_base + '2b',
        use_bias=False)(x)
    x = BatchNormalization(
        epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(
        nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(
        epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main
            path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with
    subsample=(2,2). And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(
        nb_filter1,
        (1, 1),
        strides=strides,
        name=conv_name_base + '2a',
        use_bias=False)(input_tensor)
    x = BatchNormalization(
        epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(
        nb_filter2,
        (kernel_size, kernel_size),
        name=conv_name_base + '2b',
        use_bias=False)(x)
    x = BatchNormalization(
        epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(
        nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(
        epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(
        nb_filter3,
        (1, 1),
        strides=strides,
        name=conv_name_base + '1',
        use_bias=False)(input_tensor)
    shortcut = BatchNormalization(
        epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def ResNet152(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000):
    """Instantiates the ResNet152 architecture.
    Optionally loads weights pre-trained on ImageNet. Note that when using
    TensorFlow, for best performance you should set
    image_data_format='channels_last'` in your Keras config at
    ~/.keras/keras.json.
    The model and the weights are compatible with both TensorFlow and Theano.
    The data format convention used by the model is the one specified in your
    Keras config file.
    Parameters
    ----------
        include_top: whether to include the fully-connected layer at the top of
            the network.
        weights: one of `None` (random initialization) or 'imagenet'
            (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified if
            `include_top` is False (otherwise the input shape has to be
            `(224, 224, 3)` (with `channels_last` data format) or
            `(3, 224, 224)` (with `channels_first` data format). It should have
            exactly 3 inputs channels, and width and height should be no
            smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction when
            `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional layer.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional layer, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.
    Returns
    -------
        A Keras model instance.
    Raises
    ------
        ValueError: in case of invalid argument for `weights`, or invalid input
        shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=56,
                                      min_size=56,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(
                tensor=input_tensor, shape=input_shape, name='data')
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    eps = 1.1e-5

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet152')

    # load weights
    if weights == 'imagenet':
        filename = 'resnet152_weights_{}.h5'.format(K.image_dim_ordering())
        if K.backend() == 'theano':
            path = WEIGHTS_PATH_TH
            md5_hash = MD5_HASH_TH
        else:
            path = WEIGHTS_PATH_TF
            md5_hash = MD5_HASH_TF
        weights_path = get_file(
            fname=filename,
            origin=path,
            cache_subdir='models',
            md5_hash=md5_hash,
            hash_algorithm='md5')
        model.load_weights(weights_path, by_name=True)

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
return model

model = resnet152(weights=None, include_top=False, )

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
lr_reducer=ReduceLROnPlateau(monitor='val_my_metric',factor=0.2,patience=5,mode='max',min_lr=1e-3)

checkpoint=ModelCheckpoint('model10.h5',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

model_details=model.fit(data_train,box_train,batch_size=128,epochs=55,shuffle=True,validation_split=0.1,callbacks=[lr_scheduler,lr_reducer,checkpoint],verbose=1)

model.save('model10.h5')

scores=model.evaluate(data_test,box_test,verbose=1)
print('Test loss : ',scores[0])
print('Test accuracy : ',scores[1])

utils.plot_model(model_details)
