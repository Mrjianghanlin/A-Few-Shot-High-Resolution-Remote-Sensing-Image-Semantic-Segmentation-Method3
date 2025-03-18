import os
import numpy as np
import osgeo.gdal as gdal


def colorize_segmentation(input_path, output_path):
    # 读取图像
    dataset = gdal.Open(input_path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()

    # 定义颜色映射
    colormap = {
        0: [0, 0, 0],     # background
        1: [128, 64, 128], # road
        2: [0, 128, 0],   # farmland
        3: [0, 0, 255],   # building
        4: [0, 255, 255], # water
        5: [0, 255, 0]    # grassland
    }

    # 创建RGB图像
    rgb_image = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

    # 根据标签图像的值填充颜色
    for label, color in colormap.items():
        rgb_image[data == label] = color

    # 保存为新的TIF文件
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_path, rgb_image.shape[1], rgb_image.shape[0], 3, gdal.GDT_Byte)
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    for i in range(3):
        out_band = out_dataset.GetRasterBand(i + 1)
        out_band.WriteArray(rgb_image[:, :, i])
        out_band.FlushCache()
    out_dataset = None

if __name__ == "__main__":
    input_folder = "./docs/imsge15"
    output_folder = "./docs/imsge15_color"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            colorize_segmentation(input_path, output_path)
