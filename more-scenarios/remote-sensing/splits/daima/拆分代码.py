def split_file(filename, ratio):
    with open(filename, 'r') as file:
        lines = file.readlines()

    total_lines = len(lines)
    split_index = int(total_lines * ratio)

    file1_lines = lines[:split_index]
    file2_lines = lines[split_index:]

    with open('../pascal/loveda/Rural/labeled.txt', 'w') as file1:
        file1.writelines(file1_lines)

    with open('unlabeled.txt', 'w') as file2:
        file2.writelines(file2_lines)


# 拆分比例  labeled.txt
split_ratio = 0.9999999999999999999

# 读取文件并进行拆分
split_file(r'C:\Users\wang\Desktop\VOCdevkit\VOC2007\ImageSets\Segmentation\train.txt', split_ratio)
