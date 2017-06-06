
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense,Input
from keras import applications
from keras.models import Model

# dimensions of our images.
# img_width, img_height = 224, 224
img_width, img_height = 500, 500

pretrained_model = "inceptionv3_500"
weight_folder = "part_3_weights"


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

batch_size = 4


datagen = ImageDataGenerator(rescale=1. / 255)
model = Sequential()
model.add(Dropout(0,input_shape = (img_width,img_height,3)))

# generator = datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode=None,
#     shuffle=False)
# img_test = model.predict_generator(
#     generator, nb_test_samples // batch_size+1, verbose=True, pickle_safe=True, workers=1)
# del model
# np.save(open('../data/image_test.npy', 'w'),
#         img_test)


generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
img_train = model.predict_generator(
    generator, nb_train_samples // batch_size+1, verbose=True, pickle_safe=True, workers=1)
del model
print(img_train.shape)
np.save(open('../data/image_train.npy', 'w'),
        img_train)





