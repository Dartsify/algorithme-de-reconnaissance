from pathlib import Path
import matplotlib.pyplot as plt

import cv2

IMAGE_DIR = Path('dataset/')

img = cv2.imread(str(IMAGE_DIR / 'cam1_080426_215714.jpg'), cv2.IMREAD_COLOR_RGB)
print(f'Image height: {img.shape[0]} pixels')
print(f'Image width: {img.shape[1]} pixels')
# print(f'Number of color channels: {img.shape[2]}') # 3 pour RGB

if img is None:
	raise FileNotFoundError(f"Image not found: {IMAGE_DIR / 'cam1_080426_215714.jpg'}")

# Voici un label d'exemple pour l'image "cam1_080426_215714.jpg" dans le fichier "label.csv":
# {
# 	"objects":
# 	[
# 		{
# 			"point":
# 			{
# 				"x":100,
# 				"y":200
# 			}
# 		},
# 		{
# 			"point":
# 			{
# 				"x":150,
# 				"y":250
# 			}
# 		},
# 		{
# 			"point":
# 			{
# 				"x":200,
# 				"y":300
# 			}
# 		}
# 	]
# }

plt.imshow(img)
plt.show()