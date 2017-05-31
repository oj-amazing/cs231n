from PIL import ImageFile
import os
import numpy as np
import pickle
import shutil
#for small data
CAT_SIZE = 100

#for full data
#CAT_SIZE = 10**5 

#for small data
BASE_DIR = "data/Pandora18K_small_train_val_test_split/"
#for full data
# BASE_DIR = "data/Pandora18K_train_val_test_split/"




cmd = "np.array([0]*65+[1]*14+[2]*125)"
cat_count_train_list = []
cat_count_val_list = []
cat_count_total_list = []
cat_count_test_list = []
np.random.seed(231)
if __name__ == "__main__":
    total_count = 0
    for root, dirs, files in os.walk("data/Pandora_18k_flat/"):
        for dir in dirs:
            same_cat_count =0
            same_cat_count_train = 0
            same_cat_count_val = 0
            same_cat_count_test = 0
            print(dir)
            for root_f, _, files in os.walk(os.path.join(root,dir)):
                for file in files:
                    if '.jpg' in str(file).lower():
                        if same_cat_count >= CAT_SIZE:
                            break
                        else:
                            cur_path = os.path.join(root_f,file)

                            rand_num = np.random.uniform()
                            if rand_num > 0.2:
                                new_path = os.path.join(BASE_DIR,'train')
                                same_cat_count_train+=1
                            elif rand_num > 0.1 and rand_num <= 0.2:
                                new_path = os.path.join(BASE_DIR,'val')
                                same_cat_count_val+=1
                            else:
                                new_path = os.path.join(BASE_DIR,'test')
                                same_cat_count_test+=1
                            same_cat_count+=1
                            new_path = os.path.join(new_path,dir)
                            if not os.path.exists(new_path):
                                os.makedirs(new_path)
                            shutil.copy2(cur_path, new_path)
                        total_count+=1
            cat_count_train_list.append(same_cat_count_train)
            cat_count_val_list.append(same_cat_count_val)
            cat_count_test_list.append(same_cat_count_test)
            cat_count_total_list.append(same_cat_count)
    print(cat_count_train_list)
    print("total_train_size: %s"%sum(cat_count_train_list))
    print(cat_count_val_list)
    print("total_val_size: %s"%sum(cat_count_val_list))
    print(cat_count_test_list)
    print("total_test_size: %s"%sum(cat_count_test_list))
    print(cat_count_total_list)
    print("total_data_size: %s"%sum(cat_count_total_list))
    print(total_count)
    cmd_train_l = ["[%s]*%s"%(i,count) for i,count in enumerate(cat_count_train_list)]
    cmd_train = '+'.join(cmd_train_l)
    cmd_train = 'train_labels = np.array('+cmd_train+')'

    cmd_val_l = ["[%s]*%s"%(i,count) for i,count in enumerate(cat_count_val_list)]
    cmd_val = '+'.join(cmd_val_l)
    cmd_val = 'val_labels = np.array('+cmd_val+')'


    print(cmd_train)
    print(cmd_val)
    

    






    # for root, dirs, files in os.walk("data/Pandora_small_restructured"):
    #     path = root.split(os.sep)
    #     folder_to_remove = set()
    #     for file in files:
    #         if '.jpg' in str(file).lower():
    #             if last_sep == os.sep:
    #                 same_cat_count += 1
    #                 last_sep = os.sep
    #             else:
    #                 print("cat_count:")
    #                 print(same_cat_count)
    #                 same_cat_count = 1
    #                 last_sep = os.sep
    #             total_count+=1
    print(total_count)