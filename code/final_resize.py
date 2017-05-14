from PIL import ImageFile
import os
import numpy as np
import pickle
from scipy.misc import imread
from scipy.misc import imresize
from PIL import Image

file_name_list = list()
img_sizes = np.zeros([18038,2])
if __name__ == "__main__":
    count = 0

    for root, dirs, files in os.walk("../dataset/Pandora_18k_resized"):
        path = root.split(os.sep)
        for file in files:
            if '.jpg' in str(file).lower():
                file_path = os.path.join(root,file)

                imagedata = imread(file_path)
                h,w,c = imagedata.shape

                if (h > w):
                  resized = imresize(imagedata, (500, int(500 * w/h), 3), interp='bilinear', mode=None)
                  num_zeros = int((500 - resized.shape[1]) / 2)
                  npad = ((0, 0), (num_zeros, num_zeros ), (0, 0))
                  padded = np.pad(resized, pad_width=npad, mode='constant')
                  padded = imresize(padded, (500, 500, 3), interp='bilinear', mode=None)
                  im = Image.fromarray(padded)
                  im.save(file_path)
                else:
                  resized = imresize(imagedata, (int(500 * h/w), 500, 3), interp='bilinear', mode=None)
                  num_zeros = int((500 - resized.shape[0]) / 2)
                  npad = ((num_zeros, num_zeros), (0, 0), (0, 0))
                  padded = np.pad(resized, pad_width=npad, mode='constant')
                  padded = imresize(padded, (500, 500, 3), interp='bilinear', mode=None)
                  im = Image.fromarray(padded)
                  im.save(file_path)

