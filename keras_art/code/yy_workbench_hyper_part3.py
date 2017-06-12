
# coding: utf-8

# In[8]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
# img_width, img_height = 224, 224
img_width, img_height = 500, 500

pretrained_model = "inceptionv3_500_part3"
best_weight_path_part2 = '../model/inceptionv3_500_0.5542/best_weights_inceptionv3_500.h5'
incpetion_weight_path = '../model/inceptionv3_500_0.5857/finetune_weights_0.5857.h5'
# best_weight_path = '../model/inceptionv3_500_0.5857/finetune_weights.h5'

# top_model_weights_path = '../model/bottleneck_fc_model_%s.h5'%pretrained_model
# train_data_dir = '../data/Pandora18K_small_train_val_test_split/train'
# validation_data_dir = '../data/Pandora18K_small_train_val_test_split/val'
# test_data_dir = '../data/Pandora18K_small_train_val_test_split/test'
# nb_train_samples = 1462
# nb_validation_samples = 167
# nb_test_samples = 171
train_data_dir = '../data/Pandora18K_train_val_test_split/train'
validation_data_dir = '../data/Pandora18K_train_val_test_split/val'
test_data_dir = '../data/Pandora18K_train_val_test_split/test'
nb_train_samples = 14313
nb_validation_samples = 1772
nb_test_samples = 1791






# In[2]:


datagen = ImageDataGenerator(rescale=1. / 255)
# build the VGG16 network
# model = applications.VGG16(include_top=False, weights='imagenet')


# In[3]:


from keras.utils.np_utils import to_categorical

train_data = np.load(open('../model/bottleneck_features_train_%s.npy'%pretrained_model))
#full
train_labels = np.array([0]*684+[1]*598+[2]*655+[3]*657+[4]*808+[5]*675+[6]*715+[7]*946+[8]*995+[9]*1021+[10]*803+[11]*816+[12]*566+[13]*959+[14]*842+[15]*831+[16]*849+[17]*893)

#small
#train_labels = np.array([0]*78+[1]*79+[2]*77+[3]*85+[4]*78+[5]*87+[6]*80+[7]*81+[8]*81+[9]*80+[10]*85+[11]*83+[12]*74+[13]*87+[14]*83+[15]*83+[16]*78+[17]*83)

train_labels = to_categorical(train_labels, num_classes=18)

validation_data = np.load(open('../model/bottleneck_features_validation_%s.npy'%pretrained_model))
#full
validation_labels = np.array([0]*72+[1]*73+[2]*72+[3]*93+[4]*78+[5]*74+[6]*85+[7]*124+[8]*131+[9]*118+[10]*109+[11]*105+[12]*80+[13]*130+[14]*108+[15]*89+[16]*111+[17]*120)

#small
#validation_labels = np.array([0]*8+[1]*10+[2]*11+[3]*5+[4]*11+[5]*6+[6]*8+[7]*8+[8]*9+[9]*12+[10]*7+[11]*10+[12]*14+[13]*5+[14]*11+[15]*11+[16]*12+[17]*9)
validation_labels = to_categorical(validation_labels, num_classes=18)

test_data = np.load(open('../model/bottleneck_features_test_%s.npy'%pretrained_model))
#full
test_labels = np.array([0]*91+[1]*60+[2]*75+[3]*82+[4]*104+[5]*83+[6]*95+[7]*121+[8]*131+[9]*123+[10]*103+[11]*117+[12]*65+[13]*123+[14]*121+[15]*112+[16]*89+[17]*96)

#small
#test_labels = np.array([0]*14+[1]*11+[2]*12+[3]*10+[4]*11+[5]*7+[6]*12+[7]*11+[8]*10+[9]*8+[10]*8+[11]*7+[12]*12+[13]*8+[14]*6+[15]*6+[16]*10+[17]*8)
test_labels = to_categorical(test_labels, num_classes=18)


# In[4]:


# base_inception_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')


# In[5]:


from keras.models import Model, Input
from keras.applications.inception_v3 import conv2d_bn
from keras import layers
from keras.layers import AveragePooling2D
inputs = Input(shape=train_data.shape[1:], name="input_layer")
i = 1
channel_axis = 3
branch1x1 = conv2d_bn(inputs, 320, 1, 1)

branch3x3 = conv2d_bn(inputs, 384, 1, 1)
branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
branch3x3 = layers.concatenate(
    [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

branch3x3dbl = conv2d_bn(inputs, 448, 1, 1)
branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
branch3x3dbl = layers.concatenate(
    [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

branch_pool = AveragePooling2D(
    (3, 3), strides=(1, 1), padding='same')(inputs)
branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
x = layers.concatenate(
    [branch1x1, branch3x3, branch3x3dbl, branch_pool],
    axis=channel_axis,
    name='mixed' + str(9 + i))

base_model = Model(inputs, x)


# In[6]:


base_model.layers[1 ].name='conv2d_90'
base_model.layers[2 ].name='batch_normalization_90'
base_model.layers[3 ].name='activation_90'
base_model.layers[4 ].name='conv2d_87'
base_model.layers[5 ].name='conv2d_91'
base_model.layers[6 ].name='batch_normalization_87'
base_model.layers[7 ].name='batch_normalization_91'
base_model.layers[8 ].name='activation_87'
base_model.layers[9 ].name='activation_91'
base_model.layers[10].name='conv2d_88'
base_model.layers[11].name='conv2d_89'
base_model.layers[12].name='conv2d_92'
base_model.layers[13].name='conv2d_93'
base_model.layers[14].name='average_pooling2d_9'
base_model.layers[15].name='conv2d_86'
base_model.layers[16].name='batch_normalization_88'
base_model.layers[17].name='batch_normalization_89'
base_model.layers[18].name='batch_normalization_92'
base_model.layers[19].name='batch_normalization_93'
base_model.layers[20].name='conv2d_94'
base_model.layers[21].name='batch_normalization_86'
base_model.layers[22].name='activation_88'
base_model.layers[23].name='activation_89'
base_model.layers[24].name='activation_92'
base_model.layers[25].name='activation_93'
base_model.layers[26].name='batch_normalization_94'
base_model.layers[27].name='activation_86'
base_model.layers[28].name='mixed9_1'
base_model.layers[29].name='concatenate_2'
base_model.layers[30].name='activation_94'
base_model.layers[31].name='mixed10'


# In[9]:


base_model = Model(inputs, x, name='inception_v3')
base_model.load_weights(incpetion_weight_path, by_name=True)


# In[11]:


params=(1.0e-07, 394, 0.7, 0.7)
top_model=Sequential()
top_model.add(Flatten(input_shape=train_data.shape[1:]))
top_model.add(Dropout(params[2]))
top_model.add(Dense(params[1], activation='relu'))
top_model.add(Dropout(params[2]))
top_model.add(Dense(18, activation='softmax'))

top_model.load_weights(best_weight_path_part2)



# In[12]:


model = Model(inputs= base_model.input, outputs= top_model(base_model.output))


# In[13]:


model.summary()
model.layers[-1].summary()


# In[14]:


# import keras
# model.compile(optimizer=keras.optimizers.Adam(lr=params[0], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
#                   loss='categorical_crossentropy', metrics=['accuracy'])
# loss, acc = model.evaluate(validation_data, validation_labels, batch_size=32, verbose=1, sample_weight=None)
# print(loss, acc)


# In[15]:


import keras
import os
import pickle


epochs = 50
batch_size = 64
history = dict()
for i in range(1):
    best_acc = 0
    
    #0:lr,1:dense middle,2:dropout1, 3:dropout2
    params=(10**np.random.uniform(low=-10, high=-5),             np.random.randint(low = 50, high=800),             np.random.uniform(low=0.3,high=0.8),            np.random.uniform(low=0.5,high=0.8))
    params =(params[0], 394, params[2], params[3])
    print(params)
    
    top_model=Sequential()
    top_model.add(Flatten(input_shape=train_data.shape[1:]))
    top_model.add(Dropout(params[2]))
    top_model.add(Dense(params[1], activation='relu'))
    top_model.add(Dropout(params[2]))
    top_model.add(Dense(18, activation='softmax'))

    top_model.load_weights(best_weight_path_part2)
    model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

    model.compile(optimizer=keras.optimizers.Adam(lr=params[0], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  loss='categorical_crossentropy', metrics=['accuracy'])


    weight_dir = "../model/%s"%pretrained_model
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    weight_path = os.path.join(weight_dir, "temp_weights_%s.h5"%pretrained_model)
    best_weight_path = os.path.join(weight_dir, "best_weights_%s.h5"%pretrained_model)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=weight_path, verbose=1,monitor='val_acc', save_best_only=True)
    stopper = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='auto')
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),callbacks=[checkpointer,stopper])
    model.load_weights(weight_path)
    loss, acc = model.evaluate(validation_data, validation_labels, batch_size=32, verbose=1, sample_weight=None)
    if acc>best_acc:
        model.save_weights(best_weight_path)
    history[acc] = (loss, acc, params)
    
print(history)
pickle.dump(history, open(os.path.join(weight_dir,'history_%s.p'%pretrained_model),'w'))

# model.save_weights(top_model_weights_path,overwrite=True)


# In[ ]:


model.load_weights(best_weight_path)
model.evaluate(validation_data, validation_labels, batch_size=32, verbose=1, sample_weight=None)


# In[6]:


model.evaluate(test_data, test_labels, batch_size=32, verbose=1, sample_weight=None)


# In[8]:



# import pickle
# import os

# weight_dir = "../model/%s"%pretrained_model
# if not os.path.exists(weight_dir):
#     os.makedirs(weight_dir)
# weight_path = os.path.join(weight_dir, "temp_weights_%s.h5"%pretrained_model)
# history = pickle.load(open(os.path.join(weight_dir,'history_%s.p'%pretrained_model),'rb'))


# # In[9]:


# history


# In[ ]:




