from PIL import ImageFile
import os
import numpy as np
import pickle




if __name__ == "__main__":
    count = 0
    for root, dirs, files in os.walk("../data/Pandora_small"):
        path = root.split(os.sep)
        for file in files:
            if '.jpg' in str(file).lower():
                cur_path = os.path.join(root,file)
                new_path = os.path.join(os.path.dirname(root),file)
                os.rename(cur_path, new_path)
                # print(cur_path)
                # print(new_path)
                count += 1
    print(count)