import os
import random
import shutil
from shutil import copy2

import numpy as np
from PIL import Image
from tqdm import tqdm


def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹
    :param target_data_folder: 目标文件夹
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
        test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
        train_stop_flag = current_data_length * train_scale
        val_stop_flag = current_data_length * (train_scale + val_scale)
        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0
        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx <= train_stop_flag:
                copy2(src_img_path, train_folder)
                # print("{}复制到了{}".format(src_img_path, train_folder))
                train_num = train_num + 1
            elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
                copy2(src_img_path, val_folder)
                # print("{}复制到了{}".format(src_img_path, val_folder))
                val_num = val_num + 1
            else:
                copy2(src_img_path, test_folder)
                # print("{}复制到了{}".format(src_img_path, test_folder))
                test_num = test_num + 1

            current_idx = current_idx + 1

        print("*********************************{}*************************************".format(class_name))
        print(
            "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_data_length))
        print("训练集{}：{}张".format(train_folder, train_num))
        print("验证集{}：{}张".format(val_folder, val_num))
        print("测试集{}：{}张".format(test_folder, test_num))


def data_txt_split(data_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):
    """
    按比例划分数据集，并将划分出的数据名称保存在相对应的txt文档中
    :parar data_folder:数据集的根目录
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    """
    random.seed(0)
    print("Generate txt in ImageSets.")

    SegFilePath = os.path.join(data_folder, 'labels')
    SavePath = os.path.join(data_folder, "ImageSets")
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)

    temp_seg = os.listdir(SegFilePath)
    random.shuffle(temp_seg)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num = len(temp_seg)
    list = range(num)
    num_trainval = int(num * (train_scale + val_scale))##train+val
    num_train = int(num * train_scale)##train

    train_val = random.sample(list, num_trainval)
    train = random.sample(train_val, num_train)
    test = [i for i in list if i not in train_val]


    ftrain = open(os.path.join(SavePath, 'train.txt'), 'w')
    fval = open(os.path.join(SavePath, 'val.txt'), 'w')
    ftest = open(os.path.join(SavePath, 'test.txt'), 'w')

    for i in tqdm(list):
        name = temp_seg[i][:-4] + '\n'
        if i in train_val:
            pass
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrain.close()
    fval.close()
    ftest.close()
    print("Gennrate txt in Set Imagesets done")

    # print("计算样本类别的占比，这可能需要一段时间。")
    # train_labels = []
    # with open(os.path.join(SavePath, 'train.txt'), 'r') as f:
    #     train_file_names = f.readlines()
    #
    # for seg in temp_seg:
    #     if seg.endswith(".png"):
    #         name = seg[:-4]
    #         if name in train_file_names:
    #
    #
    # classes_nums = np.zeros([256], int)
    # for i in tqdm(train_labels):
    #     # name = total_seg[i]
    #     # label_file_name = os.path.join(SegFilePath, name)
    #     png = np.array(Image.open(i), np.uint8)
    #     classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
    #
    # total_pixels = np.sum(classes_nums)
    # class_percentages = classes_nums / total_pixels * 100
    #
    # print("打印像素点的值与数量。")
    # print('-' * 54)
    # print("| %15s | %15s | %15s |" % ("Key", "Value", "Percentage"))
    # print('-' * 54)
    # for i in range(256):
    #     if classes_nums[i] > 0:
    #         print("| %15s | %15s | %15.2f%% |" % (str(i), str(classes_nums[i]), class_percentages[i]))
    #         print('-' * 54)

if __name__ == '__main__':
    data_folder = 'D:/User/Desktop/train'
    data_txt_split(data_folder,
                   train_scale=0.8,
                   val_scale=0.1,
                   test_scale=0.1
                   )
