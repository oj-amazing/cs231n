from PIL import ImageFile
import os
import numpy as np
import pickle
def get_img_size(file_path):
    ImPar=ImageFile.Parser()
    with open(file_path, "rb") as f:
        ImPar=ImageFile.Parser()
        chunk = f.read(2048)
        count=2048
        while chunk != "":
            ImPar.feed(chunk)
            if ImPar.image:
                break
            chunk = f.read(2048)
            count+=2048
        #print(ImPar.image.size)
        #print(count)
        return ImPar.image.size


file_name_list = list()
img_sizes = np.zeros([18038,2])
if __name__ == "__main__":
    count = 0
    for root, dirs, files in os.walk("../dataset_copy"):
        path = root.split(os.sep)
        for file in files:
            if '.jpg' in str(file).lower():
                print(len(path) * '---', file)
                file_path = os.path.join(root,file)
                img_size = get_img_size(file_path)
                img_sizes[count] = np.array(img_size)
                file_name_list.append(file_path)
                count += 1
    print(count)
    #pickle.dump(img_sizes, open('../data/img_sizes.p','wb'))
    #pickle.dump(file_name_list, open('../data/file_name_list.p','wb'))


    #img_sizes = pickle.load(open("../data/img_sizes.p", 'rb'))
    #file_name_list = pickle.load(open('../data/file_name_list.p','rb'))
