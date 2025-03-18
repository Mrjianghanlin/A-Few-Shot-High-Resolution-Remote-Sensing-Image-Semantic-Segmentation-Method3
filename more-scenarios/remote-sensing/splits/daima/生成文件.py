
input_file = r'D:\study\banjiandu\UniMatch-main\UniMatch-main\splits\pascal\val.txt'
output_file = './val.txt'

# with open(input_file, 'r') as f_input, open(output_file, 'w') as f_output:   labeled.txt
#     lines = f_input.readlines()
#     for line in lines:
#         line = line.strip()
#         parts = line.split('_')
#         image_name = f"JPEGImages/{parts[1]}_{parts[2]}_{parts[3]}.tif"
#         seg_name = f"SegmentationClass/{parts[1]}_{parts[2]}_{parts[3]}.tif"
#         output_line = f"{image_name} {seg_name}\n"
#         f_output.write(output_line)



with open(input_file, 'r') as f_input, open(output_file, 'w') as f_output:
    lines = f_input.readlines()
    for line in lines:
        line = line.strip()
        image_name = f"JPEGImages/{line}.png"
        seg_name = f"SegmentationClass/{line}.png"
        output_line = f"{image_name} {seg_name}\n"
        f_output.write(output_line)




