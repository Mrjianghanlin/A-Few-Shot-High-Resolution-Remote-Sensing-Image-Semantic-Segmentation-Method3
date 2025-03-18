import os
import random

# 定义输入文件和输出目录
labeled_txt_file = 'labeled.txt'
output_dir = 'data_splits'

# 划分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取labeled.txt文件内容
with open(labeled_txt_file, 'r') as file:
    labeled_file_list = file.readlines()

# 随机打乱文件列表
random.shuffle(labeled_file_list)

# 计算划分的行数
total_count = len(labeled_file_list)
train_count = int(total_count * train_ratio)
val_count = int(total_count * val_ratio)

# 划分数据集
train_data = labeled_file_list[:train_count]
val_data = labeled_file_list[train_count:train_count + val_count]
test_data = labeled_file_list[train_count + val_count:]

# 写入划分后的文件列表
with open(os.path.join(output_dir, 'train.txt'), 'w') as file:
    file.writelines(train_data)

with open(os.path.join(output_dir, 'val.txt'), 'w') as file:
    file.writelines(val_data)

with open(os.path.join(output_dir, 'test.txt'), 'w') as file:
    file.writelines(test_data)

# 打印划分结果
print(f"总共 {total_count} 条数据。")
print(f"训练集包含 {len(train_data)} 条数据。")
print(f"验证集包含 {len(val_data)} 条数据。")
print(f"测试集包含 {len(test_data)} 条数据。")
