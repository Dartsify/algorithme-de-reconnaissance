from pathlib import Path
import matplotlib.pyplot as plt

import cv2

IMAGE_DIR = Path('final/saved_images')

img = cv2.imread(str(IMAGE_DIR / 'cam2_20260422_043952_587246.jpg'), cv2.IMREAD_COLOR_RGB)
print(f'Image height: {img.shape[0]} pixels')
print(f'Image width: {img.shape[1]} pixels')
# print(f'Number of color channels: {img.shape[2]}') # 3 pour RGB
# cv2.putText(img, f'Resolution de l\'image : {img.shape[1]}x{img.shape[0]}', (840, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)  # titre des étiquettes
# cv2.putText(img, "Camera 1", (840, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)  # titre des étiquettes


if img is None:
	raise FileNotFoundError("Image not found")

# (rayons, couleurs) pour les cercles concentriques à dessiner
# rayons = [(8,(0, 255, 255)), (22,(0, 128, 255)), (138,(0, 255, 255)), (151,(0, 255, 0)), (227,(255, 0, 255)), (247,(0, 0, 255))]
# # rayons = [(8,(255, 0, 0)), (20,(0, 128, 255)), (134,(0, 255, 255)), (263,(0, 255, 0)), (283,(255, 0, 255)), (317,(0, 0, 255))]

# cv2.circle(img, (624,325), radius=3, color=(255, 0, 0), thickness=-1)  # centre des cercles
# cv2.putText(img, 'Centre (624, 325)', (624 + 5, 325 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)  # étiquette du centre
# for r in rayons:
# 	xText = 1000 + 20 * rayons.index(r)  # position horizontale de l'étiquette à droite du cercle
# 	yText = 500 + 20 * rayons.index(r)  # espacement vertical de 20 pixels entre les étiquettes
# 	cv2.circle(img, (624,325), radius=r[0], color=r[1], thickness=2)  # 6 cercles colorés
# 	cv2.putText(img, f'rayon {r[0]}px', (xText, yText), cv2.FONT_HERSHEY_SIMPLEX, 0.7, r[1], 2, cv2.LINE_AA)  # étiquette du rayon

# # write image with RGB color space (OpenCV uses BGR by default, so we need to convert it back to BGR before saving)
# cv2.imwrite(str(IMAGE_DIR / 'cam1_homographie_with_circles.jpeg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


plt.imshow(img)
plt.show()