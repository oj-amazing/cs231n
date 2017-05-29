import subprocess as sub
import os

home = os.path.expanduser('~')
path = home + "/tensorflow/"
image_dir = home + "/cs231n/dataset_flat"

#Default values
learning_rate = 0.01
how_many_training_steps = 4000
train_batch_size = 100
random_crop = 0
random_scale = 0
random_brightness = 0
flip_left_right = False
dataset_size = 18000

# Sweep values
learning_rates = [x / 100.0 for x in range(1,21,2)]

for learning_rate in learning_rates:
  filename = str(learning_rate) + "_" + \
    str(how_many_training_steps) + "_" + \
    str(train_batch_size) + "_" + \
    str(random_crop) + "_" + \
    str(random_scale) + "_" + \
    str(random_brightness) + "_" + \
    str(flip_left_right) + ".out"
  
  outFile = open(filename, 'w')
  
  sub.Popen([path + 'bazel-bin/tensorflow/examples/image_retraining/retrain', '--image_dir', image_dir, '--learning_rate', str(learning_rate)], stdout = outFile)

  outFile.close()
