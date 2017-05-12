import tensorflow as tf
import numpy as np
import os
from PIL import Image

cur_dir = os.getcwd()
print("resizing images")
print("current directory:",cur_dir)

def modify_image(image):

    h, w, c = image.shape

    if (h > w):
      resized = tf.image.resize_images(image, [500, 500]) # need to make this w/h
      #resized.set_shape([500,500,3])
      padded = tf.image.resize_image_with_crop_or_pad(resized, 500, 500)
      return padded
    else:
      resized = tf.image.resize_images(image, [500, 500]) # need to make this h/w
      #resized.set_shape([500,500,3])
      padded = tf.image.resize_image_with_crop_or_pad(resized, 500, 500)
      return padded

def read_image(filename_queue):
    reader = tf.WholeFileReader()
    key,value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)
    return image

def inputs():
    filenames = ['06_Rococo/Unidentified_artists/978.jpg', '06_Rococo/Unidentified_artists/977.jpg']
    #filenames = ['img1.jpg', 'img2.jpg' ]
    filename_queue = tf.train.string_input_producer(filenames,num_epochs=2)
    read_input = read_image(filename_queue)
    reshaped_image = modify_image(read_input)
    return reshaped_image

with tf.Graph().as_default():
    image = inputs()
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    for i in range(2):
        img = sess.run(image)
        img = Image.fromarray(img, "RGB")
        img.save(os.path.join(cur_dir,"foo"+str(i)+".jpeg"))

#if __name__ == '__main__':
#  parser = argparse.ArgumentParser()
#  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
#                      help='Directory for storing input data')
#  FLAGS, unparsed = parser.parse_known_args()
#  tf.app.run(main=main)#, argv=[sys.argv[0]] + unparsed)
