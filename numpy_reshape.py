import os
from scipy.misc import imread
from scipy.misc import imresize
import numpy as np
from PIL import Image

#art_dir = "/home/eraserwars/assignment2/project"
art_dir = os.getcwd()

style_dir = [(art_dir + "/" + name) for name in os.listdir(art_dir)
              if os.path.isdir(os.path.join(art_dir, name))]

#print(style_dir)

for style in style_dir:

  #print("STYLE IS: " + style + "\n\n")
  artist_dir = [(style + "/" + name) for name in os.listdir(style)
                 if os.path.isdir(os.path.join(style, name))]

  #print(artist_dir)

  for artist in artist_dir:
    for image in os.listdir(artist):
      image_file = artist + "/" + image
      print(image_file)

      imagedata = imread(image_file)
      print(imagedata.shape)
      h,w,c = imagedata.shape

      if (h > w):
        resized = imresize(imagedata, (500, int(500 * w/h), 3), interp='bilinear', mode=None)
        num_zeros = int((500 - resized.shape[1]) / 2)
        npad = ((0, 0), (num_zeros, num_zeros ), (0, 0))
        padded = np.pad(resized, pad_width=npad, mode='constant')
        padded = imresize(padded, (500, 500, 3), interp='bilinear', mode=None)
        print("padded size is = ")
        print(padded.shape)
        im = Image.fromarray(padded)
        im.save(artist + "/" + image[:-4] + "_reshaped.jpg")
      else:
        resized = imresize(imagedata, (int(500 * h/w), 500, 3), interp='bilinear', mode=None)
        num_zeros = int((500 - resized.shape[0]) / 2)
        npad = ((num_zeros, num_zeros), (0, 0), (0, 0))
        padded = np.pad(resized, pad_width=npad, mode='constant')
        padded = imresize(padded, (500, 500, 3), interp='bilinear', mode=None)
        print("padded size is = ")
        print(padded.shape)
        im = Image.fromarray(padded)
        im.save(artist + "/" + image[:-4] + "_reshaped.jpg")

