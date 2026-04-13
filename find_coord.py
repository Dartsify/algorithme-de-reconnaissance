from pathlib import Path
import matplotlib.pyplot as plt

import cv2

IMAGE_DIR = Path('image_test')

img = cv2.imread(str(IMAGE_DIR / 'image_cam2_corrigee1152x1536.jpeg'), cv2.IMREAD_COLOR_RGB)
print(f'Image height: {img.shape[0]} pixels')
print(f'Image width: {img.shape[1]} pixels')
# print(f'Number of color channels: {img.shape[2]}') # 3 pour RGB
cv2.putText(img, f'Resolution de l\'image : {img.shape[1]}x{img.shape[0]}', (10, 1350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)  # titre des étiquettes


if img is None:
	raise FileNotFoundError("Image not found")

# Voici un label d'exemple pour l'image "cam1_080426_221128.jpg" dans le fichier "label.csv":
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

# (rayons, couleurs) pour les cercles concentriques à dessiner
rayons = [(12,(255, 0, 0)), (36,(0, 128, 255)), (236,(0, 255, 255)), (258,(0, 255, 0)), (390,(255, 0, 255)), (417,(0, 0, 255))]

cv2.circle(img, (582,727), radius=3, color=(0, 255, 255), thickness=-1)  # centre des cercles
cv2.putText(img, 'Centre (582, 727)', (582 + 5, 727 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)  # étiquette du centre
for r in rayons:
	xText = 10
	yText = 1200 + 20 * rayons.index(r)  # espacement vertical de 20 pixels entre les étiquettes
	cv2.circle(img, (582,727), radius=r[0], color=r[1], thickness=2)  # 6 cercles colorés
	cv2.putText(img, f'rayon {r[0]}px', (xText, yText), cv2.FONT_HERSHEY_SIMPLEX, 0.7, r[1], 2, cv2.LINE_AA)  # étiquette du rayon

# write image with RGB color space (OpenCV uses BGR by default, so we need to convert it back to BGR before saving)
cv2.imwrite(str(IMAGE_DIR / 'image_cam2_corrigee1152x1536_annotated.jpeg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


plt.imshow(img)
plt.show()