import os

'''要重命名的图片路径'''
file_path = "D:/User/Desktop/paper/TF/train256/label111"

files = os.listdir(file_path)
for file in files:
    if file.endswith('png'):
        # 要指明重命名之后的路径
        src = os.path.join(file_path, file)
        r_name = file.split('.')[0] + '.jpg'
        dct = os.path.join(file_path, r_name)
        os.rename(src, dct)
print('Finish')