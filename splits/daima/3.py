import os
import numpy as np
from osgeo import gdal
from PIL import Image

# Define color mapping for six classes
color_mapping = {
    0: (0, 0, 0),      # Class 0: Black
    1: (255, 0, 0),    # Class 1: Red
    2: (0, 255, 0),    # Class 2: Green
    3: (0, 0, 255),    # Class 3: Blue
    4: (255, 255, 0),  # Class 4: Yellow
    5: (255, 0, 255)   # Class 5: Magenta
}

# Define image paths
input_label_image_path = r'C:\Users\wang\Desktop\data\label\8_label.tif'
output_folder = r'D:\study\banjiandu\UniMatch-main\UniMatch-main\predicted_images_result'
merged_image_file = os.path.join(output_folder, 'merged_image_8.png')

# Open the input label image
image_dataset = gdal.Open(input_label_image_path)
if image_dataset is None:
    print(f"Failed to open input label image: {input_label_image_path}")
    exit(1)

# Get image size and data type
image_size_x = image_dataset.RasterXSize
image_size_y = image_dataset.RasterYSize
data_type = image_dataset.GetRasterBand(1).DataType

# Create an output image array with colored labels
merged_image = np.zeros((image_size_y, image_size_x, 3), dtype=np.uint8)

# Iterate through the label image and assign colors
for y in range(image_size_y):
    for x in range(image_size_x):
        class_value = image_dataset.GetRasterBand(1).ReadAsArray(x, y, 1, 1)[0][0]
        rgb_color = color_mapping.get(class_value, (0, 0, 0))  # Default to black for unknown classes
        merged_image[y, x, :] = rgb_color

# Save the merged image as PNG
output_image = Image.fromarray(merged_image)
output_image.save(merged_image_file, format='PNG')

# The merged image with colored labels is now saved as 'merged_image_8.png' in the output folder.
