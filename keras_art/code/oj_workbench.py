import keras

from keras.preprocessing.image import ImageDataGenerator

######################################################################

# dimensions of our images.
img_width, img_height = 150, 150

if keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
######################################################################

train_dir = '../data/Pandora18K_small_train_val_test_split/train'
valid_dir = '../data/Pandora18K_small_train_val_test_split/val'
test_dir  = '../data/Pandora18K_small_train_val_test_split/test'
batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale= 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
valid_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen  = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

valid_gen = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

######################################################################

import subprocess

out = subprocess.Popen(['find', train_dir, '-name', "*\.[jJ][pP][gG]"], stdout = subprocess.PIPE).communicate()
train_num = out[0].count('\n')

out = subprocess.Popen(['find', valid_dir, '-name', "*\.[jJ][pP][gG]"], stdout = subprocess.PIPE).communicate()
valid_num = out[0].count('\n')

out = subprocess.Popen(['find', test_dir, '-name', "*\.[jJ][pP][gG]"], stdout = subprocess.PIPE).communicate()
test_num = out[0].count('\n')

print("Sanity check: " + str(train_num) + " train, " + str(valid_num) + " valid, " + str(test_num) + " test")

######################################################################

def create_model(input_shape):
    
    main_input = keras.layers.Input(shape=input_shape)
    
    # VGG19, 32-64-128-256-512-512
    x = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", \
                                  input_shape=input_shape)(main_input)
    x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    
    # [(None, 69, 69, 64), (None, 67, 67, 128), (None, 65, 65, 32), (None, 34, 34, 64)]
    
    # Inception 64 / 96-128-64 / 16-32-64
    x11 = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), activation="relu", padding="same")(x)
    x13 = keras.layers.Conv2D(filters=16, kernel_size=(1, 1), activation="relu")(x)
    x33 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same")(x13)
    x31 = keras.layers.Conv2D(filters=32, kernel_size=(1, 1))(x33)
    x15 = keras.layers.Conv2D(filters=8, kernel_size=(1, 1), activation="relu")(x)
    x55 = keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="same")(x15)
    x51 = keras.layers.Conv2D(filters=32, kernel_size=(1, 1))(x55)
    x = keras.layers.concatenate([x11, x31, x51], axis=3)
    
    # Dense 4096-4096
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    out = keras.layers.Dense(18, activation='softmax')(x)

    model = keras.models.Model(main_input, out)
    
    return model

######################################################################

import os

model_name = "custom"
weight_dir = "../model/%s"%model_name
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

weight_path = os.path.join(weight_dir, "temp_weights_%s.h5"%model_name)
best_weight_path = os.path.join(weight_dir, "best_weights_%s.h5"%model_name)

######################################################################

import numpy as np
import pickle

# Simulation parameters
epochs = 50
num_iters = 1

# Training parameters
learning_rates = [-3.5, -2.5]
betas_1  = [0.9, 0.999]
betas_2  = [0.9, 0.999]
epsilons = [-9, -7]
decays = 0.0

# History parameters
best_acc = -1
best_params = ()
history = dict()

for _ in range(num_iters):
    
    ##### RANDOM HYPERPARAMETERS #####
    
    learning_rate = 10**np.random.uniform(learning_rates[0], \
                                          learning_rates[1])
    beta_1  = np.random.uniform(betas_1[0], betas_1[1])
    beta_2  = np.random.uniform(betas_2[0], betas_2[1])
    epsilon = 10**np.random.uniform(epsilons[0], \
                                    epsilons[1])
    decay   = np.random.uniform(decays)
    
    print ("LR=" + str(learning_rate) + \
           ", B1=" + str(beta_1) + \
           ", B2=" + str(beta_2) + \
           ", E=" + str(epsilon) + \
           " Decay=" + str(decay))
    
    params = (learning_rate, beta_1, beta_2, epsilon, decay)
    
    ##### CREATING MODEL #####
    
    model = create_model(input_shape)
    opt = keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    ##### TRAINING MODEL #####

    checkpointer = keras.callbacks.ModelCheckpoint(filepath=weight_path, verbose=1,monitor='val_acc', save_best_only=True)
    stopper = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='auto')
    reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    
    model.fit_generator(
        train_gen,
        steps_per_epoch=train_num // batch_size,
        epochs=epochs,
        validation_data=valid_gen,
        validation_steps=valid_num // batch_size,
        callbacks=[checkpointer,stopper])
    
    model.load_weights(weight_path)
    loss, acc = model.evaluate_generator(
        valid_gen,
        steps = valid_num // batch_size)
    print("Test accuracy: " + str(acc))
    
    if acc>best_acc:
        print("Saving as best accuracy")
        model.save_weights(best_weight_path)
        best_params = params
        best_acc = acc
        
    history[acc] = (loss, acc, params)
    del model
    
pickle.dump(history, open(os.path.join(weight_dir,'history_%s.p'%model_name),'w'))

######################################################################

model = create_model(input_shape)

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.load_weights(best_weight_path)

loss, acc = model.evaluate_generator(
    test_gen,
    steps = test_num // batch_size)

print("Best accuracy: " + str(acc))
