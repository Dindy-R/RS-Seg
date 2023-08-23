import os
import cv2
def convert_to_rgb(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
     # 获取输入文件夹中的文件列表
    file_list = os.listdir(input_folder)
    for file_name in file_list:
        # 构建输入和输出文件的完整路径
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
         # 读取灰度图像
        grayscale_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
         # 将灰度图像转换为RGB图像
        rgb_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
         # 保存RGB图像
        cv2.imwrite(output_path, rgb_img)
        print(f"已将 {file_name} 转换为RGB图像")
 # 示例用法
input_folder = "D:/User/Desktop/paper/TF/train256/label"
output_folder = "D:/User/Desktop/paper/TF/train256/label111"
convert_to_rgb(input_folder, output_folder)