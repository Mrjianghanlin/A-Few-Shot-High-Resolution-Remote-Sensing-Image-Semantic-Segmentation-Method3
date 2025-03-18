input_file = 'data_info.txt'
output_file = 'data_info_without_extension.txt'

with open(input_file, 'r') as file:
    lines = file.readlines()

data_info_without_extension = []
for line in lines:
    line = line.strip()  # 去掉行首和行尾的空格和换行符
    line = line.replace('.png', '')  # 去掉.png后缀
    data_info_without_extension.append(line)

with open(output_file, 'w') as file:
    file.write('\n'.join(data_info_without_extension))
