import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cityscapes的19个类别到RGB颜色的映射
label_to_color = {
    0: [128, 64, 128],  # 道路
    1: [244, 35, 232],  # 人行道
    2: [70, 70, 70],    # 建筑
    3: [102, 102, 156], # 墙壁
    4: [190, 153, 153], # 栅栏
    5: [153, 153, 153], # 桥
    6: [250, 170, 30],  # 停车标记
    7: [220, 220, 0],   # 柱子
    8: [107, 142, 35],  # 草地
    9: [152, 251, 152], # 植被
    10: [70, 130, 180], # 天空
    11: [220, 20, 60],  # 人
    12: [255, 0, 0],    # 骑车人
    13: [0, 0, 142],    # 汽车
    14: [0, 0, 70],     # 卡车
    15: [0, 60, 100],   # 巴士
    16: [0, 80, 100],   # 火车
    17: [0, 0, 230],    # 摩托车
    18: [119, 11, 32],  # 自行车
}

# 替换此路径为你的图像文件路径
image_path = r'D:\study\banjiandu\UniMatch-main\Erhai semantic segmentation\predicted_images_city_2\frankfurt_000001_004736_gtFine_labelTrainIds.png'
# image_path = r'D:\study\banjiandu\UniMatch-main\Erhai semantic segmentation\predicted_images_city_2\munster_000063_000019_gtFine_labelTrainIds.png'

# 使用OpenCV加载图像，这里假设图像是单通道的类别标签图
label_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功加载
if label_image is None:
    raise ValueError("Image not found or path is incorrect")

# 创建一个空的RGB图像
colored_image = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)

# 根据预测的类别标签填充颜色
for label, color in label_to_color.items():
    mask = label_image == label
    colored_image[mask] = color

# 使用matplotlib显示上色后的图像（可选）
plt.imshow(colored_image)
plt.axis('off')
plt.show()

# 指定保存路径和文件名
save_path = r'D:\study\banjiandu\UniMatch-main\Erhai semantic segmentation\splits\cityscapes1\uncity/colored_image3.png'

# 保存上色后的图像
cv2.imwrite(save_path, cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR))