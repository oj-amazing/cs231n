
# coding: utf-8

# In[1]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
# img_width, img_height = 224, 224
img_width, img_height = 500, 500

pretrained_model = "inceptionv3_500"


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
model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')


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


import keras
import os
import pickle


epochs = 50
batch_size = 64
history = dict()
best_acc = 0
for i in range(10):
    #0:lr,1:dense middle,2:dropout1, 3:dropout2
    params=(10**np.random.uniform(low=-6, high=-4),np.random.randint(low = 50, high=800),np.random.uniform(low=0.4,high=0.9), np.random.uniform(low=0.4,high=0.9))
    print(params)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dropout(params[2]))#0.6
    model.add(Dense(params[1], activation='relu'))#256
    model.add(Dropout(params[3]))#0.6
    model.add(Dense(18, activation='softmax'))

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
        print("Best accuracy imporved from %s to %s"%(best_acc, acc))
        best_acc = acc
        
    if (i%2==0):
        print("history saved at iteration %s"%i)
        print("best val_acc so far: %s"%max(history.keys()))
        pickle.dump(history, open(os.path.join(weight_dir,'history_%s.p'%pretrained_model),'w'))
    del model
    
print(history)
pickle.dump(history, open(os.path.join(weight_dir,'history_%s.p'%pretrained_model),'w'))

# # model.save_weights(top_model_weights_path,overwrite=True)


# # In[5]:


# model.load_weights(best_weight_path)
# model.evaluate(validation_data, validation_labels, batch_size=32, verbose=1, sample_weight=None)


# # In[6]:


# model.evaluate(test_data, test_labels, batch_size=32, verbose=1, sample_weight=None)


# # In[7]:


# history = pickle.load(open(os.path.join(weight_dir,'history_%s.p'%pretrained_model),'rb'))


# # In[8]:


# history


# # In[ ]:




