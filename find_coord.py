from pathlib import Path
import matplotlib.pyplot as plt

import cv2

IMAGE_DIR = Path('image_test/')

img = cv2.imread(str(IMAGE_DIR / 'image_reference.jpeg'), cv2.IMREAD_COLOR_RGB)
print(f'Image height: {img.shape[0]} pixels')
print(f'Image width: {img.shape[1]} pixels')
# print(f'Number of color channels: {img.shape[2]}') # 3 pour RGB

if img is None:
	raise FileNotFoundError(f"Image not found: {IMAGE_DIR / 'image_reference.jpeg'}")


plt.imshow(img)
plt.show()