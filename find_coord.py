from pathlib import Path
import matplotlib.pyplot as plt

import cv2

IMAGE_DIR = Path('image_test/homographie_mur')

img = cv2.imread(str(IMAGE_DIR / 'new' / 'cam3_homographie_avec_points.jpg'), cv2.IMREAD_COLOR)

if img is None:
	raise FileNotFoundError("Image not found")

# convert BGR (OpenCV default) to RGB for correct plotting
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f'Image height: {img.shape[0]} pixels')
print(f'Image width: {img.shape[1]} pixels')
# print(f'Number of color channels: {img.shape[2]}') # 3 pour RGB
cv2.putText(img, f'Resolution de l\'image : {img.shape[1]}x{img.shape[0]}', (840, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)  # titre des étiquettes
cv2.putText(img, "Camera 3", (840, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)  # titre des étiquettes


centre = (642, 356)  # coordonnées du centre des cercles concentriques
# (rayons, couleurs) pour les cercles concentriques à dessiner
rayons = [(8,(0, 255, 255)), (24,(0, 128, 255)), (139,(0, 255, 255)), (155,(0, 255, 0)), (235,(255, 0, 255)), (251,(0, 0, 255))]

cv2.circle(img, centre, radius=2, color=(0, 250, 0), thickness=-1)  # centre des cercles
cv2.putText(img, 'Centre (642, 356)', (centre[0] + 5, centre[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)  # étiquette du centre
for r in rayons:
	xText = 1000 + 20 * rayons.index(r)  # position horizontale de l'étiquette à droite du cercle
	yText = 500 + 20 * rayons.index(r)  # espacement vertical de 20 pixels entre les étiquettes
	cv2.circle(img, centre, radius=r[0], color=r[1], thickness=2, lineType=cv2.LINE_AA, shift=0)  # 6 cercles colorés
	cv2.putText(img, f'rayon {r[0]}px', (xText, yText), cv2.FONT_HERSHEY_SIMPLEX, 0.7, r[1], 2, cv2.LINE_AA)  # étiquette du rayon

# write image with RGB color space (OpenCV uses BGR by default, so we need to convert it back to BGR before saving)
cv2.imwrite(str(IMAGE_DIR / 'new' / 'cam3_with_circles.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


plt.imshow(img)
plt.show()