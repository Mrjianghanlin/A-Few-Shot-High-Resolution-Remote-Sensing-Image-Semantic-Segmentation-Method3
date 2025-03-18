import os
import random

labeled_txt_file = r'D:\study\banjiandu\UniMatch-main\UniMatch-main\splits\test\labeled.txt'  # 包含带标签文件列表的txt文件

output_dir = 'un_1_3'  # Output directory (without the space character)
  # 输出目录
split_ratio = 0.333333  # 划分比例

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取labeled.txt文件内容
with open(labeled_txt_file, 'r') as file:
    labeled_file_list = file.readlines()

# 计算划分的行数
split_index = int(len(labeled_file_list) * split_ratio)

# 获取前split_index行的内容作为labeled.txt
labeled_subset = labeled_file_list[:split_index]
labeled_subset_path = os.path.join(output_dir, 'labeled.txt')
with open(labeled_subset_path, 'w') as file:
    file.writelines(labeled_subset)

# 获取剩余行的内容作为unlabeled.txt
unlabeled_subset = labeled_file_list[split_index:]
unlabeled_subset_path = os.path.join(output_dir, 'unlabeled.txt')
with open(unlabeled_subset_path, 'w') as file:
    file.writelines(unlabeled_subset)

# 打印读取的行数
print("读取了 {} 行数据。".format(len(labeled_file_list)))
print("生成的 labeled.txt 包含 {} 行数据。".format(len(labeled_subset)))
print("生成的 unlabeled.txt 包含 {} 行数据。".format(len(unlabeled_subset)))
