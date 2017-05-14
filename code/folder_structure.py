from PIL import ImageFile
import os
import numpy as np
import pickle




if __name__ == "__main__":
    count = 0
    for root, dirs, files in os.walk("/home/User/cs231n/dataset"):
        path = root.split(os.sep)
        folder_to_remove = set()
        for file in files:
            if '.jpg' in str(file).lower():
                path_len = len(path)
                new_dir = root
                #print(path_len)
                while path_len >= 7:
                    folder_to_remove.add(new_dir)
                    new_dir = os.path.dirname(new_dir)
                    path_len -= 1

                cur_path = os.path.join(root,file)
                new_path = os.path.join(new_dir,file)
                os.rename(cur_path, new_path)
                #print(cur_path)
                print(new_path)
                count += 1
        #print(folder_to_remove)
        folder_to_remove = list(folder_to_remove)
        folder_to_remove.sort(key = len, reverse=True)
        #print(folder_to_remove)
        for folder in folder_to_remove:
            try:
                os.rmdir(folder)
            except OSError as ex:
                print ("directory not empty")
    print(count)
