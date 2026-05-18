"""
Correction de perspective pour une caméra inclinée.

L'objectif final est de calculer la matrice d'homographie H pour chaque caméra inclinée.
On peut les récupérer directement et l'appliquer au modèle IA pour corriger les coordonnées avant de calculer le score.
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_points(image: np.ndarray, points: np.ndarray, labels: list[str]) -> np.ndarray:
	"""
	Dessine des points sur une image avec des étiquettes.
	Utile pour visualiser les correspondances entre les images.
	"""

	annotated_image = image.copy()

	for point, label in zip(points, labels):
		x, y = int(point[0]), int(point[1])
		cv2.circle(annotated_image, (x, y), 4, (0, 255, 0), thickness=-1)
		cv2.putText(
			annotated_image,
			label,
			(x + 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(0, 255, 0),
			2,
			cv2.LINE_AA,
		)

	return annotated_image


def detect_concentric_circles(image: np.ndarray, expected_count: int = 6) -> tuple[tuple[int, int], list[int]]:
	"""
	Estime le centre des cercles concentriques et leurs rayons à partir du profil radial.
	"""

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
	edges = cv2.Canny(blur, 40, 120)

	height, width = edges.shape
	image_center = (width // 2, height // 2)

	def evaluate_center(edge_map: np.ndarray, center: tuple[int, int], radius_spacing: int) -> tuple[float, list[tuple[int, float]]] | None:
		map_height, map_width = edge_map.shape
		max_radius = min(center[0], center[1], map_width - center[0], map_height - center[1]) - 5
		if max_radius <= 20:
			return None

		polar = cv2.warpPolar(edge_map, (max_radius, 360), center, max_radius, cv2.WARP_POLAR_LINEAR)
		profile = polar.mean(axis=0).astype(np.float32)
		smoothed_profile = np.convolve(profile, np.ones(17) / 17, mode="same")

		candidates: list[tuple[int, float]] = []
		for radius_index in range(8, len(smoothed_profile) - 8):
			window = smoothed_profile[radius_index - 3 : radius_index + 4]
			if smoothed_profile[radius_index] == window.max():
				candidates.append((radius_index, float(smoothed_profile[radius_index])))

		candidates.sort(key=lambda item: item[1], reverse=True)
		selected: list[tuple[int, float]] = []
		for radius_index, score in candidates:
			if all(abs(radius_index - selected_radius) >= radius_spacing for selected_radius, _ in selected):
				selected.append((radius_index, score))
			if len(selected) == expected_count:
				break

		if not selected:
			return None

		return sum(score for _, score in selected), selected

	# Recherche grossière sur l'image réduite pour localiser le centre.
	small_gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))
	small_edges = cv2.Canny(cv2.GaussianBlur(small_gray, (9, 9), 1.5), 40, 120)
	small_height, small_width = small_edges.shape
	small_center = (small_width // 2, small_height // 2)

	best_small_center: tuple[int, int] | None = None
	best_small_score = -1.0
	for offset_y in range(-40, 41, 4):
		for offset_x in range(-40, 41, 4):
			candidate_center = (small_center[0] + offset_x, small_center[1] + offset_y)
			if candidate_center[0] <= 20 or candidate_center[1] <= 20:
				continue
			if candidate_center[0] >= small_width - 20 or candidate_center[1] >= small_height - 20:
				continue

			candidate_result = evaluate_center(small_edges, candidate_center, 8)
			if candidate_result is None:
				continue

			score, _ = candidate_result
			if score > best_small_score:
				best_small_score = score
				best_small_center = candidate_center

	if best_small_center is None:
		best_small_center = small_center

	# Raffinement sur l'image originale.
	refined_center = (best_small_center[0] * 2, best_small_center[1] * 2)
	best_full_center: tuple[int, int] = refined_center
	best_full_radii: list[int] = []
	best_full_score = -1.0
	for offset_y in range(-12, 13, 2):
		for offset_x in range(-12, 13, 2):
			candidate_center = (refined_center[0] + offset_x, refined_center[1] + offset_y)
			if candidate_center[0] <= 20 or candidate_center[1] <= 20:
				continue
			if candidate_center[0] >= width - 20 or candidate_center[1] >= height - 20:
				continue

			candidate_result = evaluate_center(edges, candidate_center, 12)
			if candidate_result is None:
				continue

			score, selected_radii = candidate_result
			if score > best_full_score:
				best_full_score = score
				best_full_center = candidate_center
				best_full_radii = [radius for radius, _ in selected_radii]

	if len(best_full_radii) < expected_count:
		raise RuntimeError("Impossible d'extraire les 6 cercles concentriques de la cible.")

	return best_full_center, sorted(int(radius) for radius in best_full_radii[:expected_count])


def draw_concentric_circles(image: np.ndarray, center: tuple[int, int], radii: list[int]) -> np.ndarray:
	"""
	Dessine les cercles concentriques et le centre détecté sur l'image.
	"""

	annotated_image = image.copy()
	colors = [
		(255, 0, 0),
		(0, 128, 255),
		(0, 255, 255),
		(0, 255, 0),
		(255, 0, 255),
		(0, 0, 255),
	]

	for index, radius in enumerate(sorted(radii)):
		color = colors[index % len(colors)]
		cv2.circle(annotated_image, center, int(radius), color, 2)

	cv2.circle(annotated_image, center, 8, (0, 255, 255), thickness=-1)
	cv2.drawMarker(
		annotated_image,
		center,
		(0, 255, 255),
		markerType=cv2.MARKER_CROSS,
		markerSize=22,
		thickness=2,
	)
	cv2.putText(
		annotated_image,
		f"Centre ({center[0]}, {center[1]})",
		(center[0] + 15, center[1] - 15),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		(0, 255, 255),
		2,
		cv2.LINE_AA,
	)

	return annotated_image




def main() -> None:
	# On utilise Path(__file__) pour construire des chemins relatifs au script.
	# Cela évite les problèmes de chemins absolus quand le projet est déplacé.
	base_dir = Path(__file__).parent / "image_test" / "homographie_mur_fixe"
	camera_image_path = base_dir / "cam3_ref.jpg"
	reference_image_path = base_dir / 'ref_mathias.jpg'

	if not camera_image_path.exists():
		raise FileNotFoundError(
			f"Image avec caméra inclinée introuvable : {camera_image_path}. "
		)

	if not reference_image_path.exists():
		raise FileNotFoundError(
			f"Image de référence introuvable : {reference_image_path}. "
		)

	camera_image = cv2.imread(str(camera_image_path), cv2.IMREAD_COLOR)
	reference_image = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)

	if camera_image is None:
		raise RuntimeError(f"Impossible de lire l'image avec caméra inclinée : {camera_image_path}")

	if reference_image is None:
		raise RuntimeError(f"Impossible de lire l'image de référence : {reference_image_path}")

	print("Images chargées avec succès.")
	print(f"- Image avec caméra inclinée : {camera_image.shape[1]} x {camera_image.shape[0]} pixels")
	print(f"- Image de référence : {reference_image.shape[1]} x {reference_image.shape[0]} pixels")

	# Une homographie plane possède 8 degrés de liberté utiles, car la matrice
	# 3x3 est définie à un facteur d'échelle près. Il faut donc au minimum 4
	# correspondances de points non alignés pour calculer la transformation.

	# -----
	# Si j'augmente le nombre de correspondances de points, j'améliore la précision de l'homographie.
	# Par exemple, je peux ajouter toutes les positions des numéros visibles de la cible
	# dans les deux images (caméra inclinée et référence).
	# C'est-à-dire passer de 4 à 20 points de correspondance (à voir).
	# -----


	"""
	Position des points de la cible vue de la caméra 1 (caméra inclinée) :
	"""
	# pts_src contient les points mesurés dans l'image avec caméra inclinée. [x,y]
	# pts_src_cam1 = np.array(
	# 	[
	# 		[401.3, 499.7],  # coordonnées du point sur le bord supérieur du nombre visible 1 dans l'image avec caméra inclinée
	# 		[316.6, 447.1],  # coordonnées du point sur le bord supérieur du nombre visible 18 dans l'image avec caméra inclinée
	# 		[273.3, 382.2],  # coordonnées du point sur le bord supérieur du nombre visible 4 dans l'image avec caméra inclinée
	# 		[271.8, 317.7],  # coordonnées du point sur le bord supérieur du nombre visible 13 dans l'image avec caméra inclinée
	# 		[299.9, 259.7],  # coordonnées du point sur le bord supérieur du nombre visible 6 dans l'image avec caméra inclinée
	# 		[348.0, 211.9],  # coordonnées du point sur le bord supérieur du nombre visible 10 dans l'image avec caméra inclinée
	# 		[409.2, 176.5],  # coordonnées du point sur le bord supérieur du nombre visible 15 dans l'image avec caméra inclinée
	# 		[475.7, 152.4],  # coordonnées du point sur le bord supérieur du nombre visible 2 dans l'image avec caméra inclinée
	# 		[544.4, 138.9],  # coordonnées du point sur le bord supérieur du nombre visible 17 dans l'image avec caméra inclinée
	# 		[617.5, 135.3],  # coordonnées du point sur le bord supérieur du nombre visible 3 dans l'image avec caméra inclinée
	# 		[687.3, 142.4],  # coordonnées du point sur le bord supérieur du nombre visible 16 dans l'image avec caméra inclinée
	# 		[755.4, 159.7],  # coordonnées du point sur le bord supérieur du nombre visible 7 dans l'image avec caméra inclinée
	# 		[815.9, 189.0],  # coordonnées du point sur le bord supérieur du nombre visible 19 dans l'image avec caméra inclinée
	# 		[866.1, 230.5],  # coordonnées du point sur le bord supérieur du nombre visible 8 dans l'image avec caméra inclinée
	# 		[896.4, 284.1],  # coordonnées du point sur le bord supérieur du nombre visible 11 dans l'image avec caméra inclinée
	# 		[900.2, 348.8],  # coordonnées du point sur le bord supérieur du nombre visible 14 dans l'image avec caméra inclinée
	# 		[860.6, 419.4],  # coordonnées du point sur le bord supérieur du nombre visible 9 dans l'image avec caméra inclinée
	# 		[778.0, 481.7],  # coordonnées du point sur le bord supérieur du nombre visible 12 dans l'image avec caméra inclinée
	# 		[657.7, 522.0],  # coordonnées du point sur le bord supérieur du nombre visible 5 dans l'image avec caméra inclinée
	# 		[522.8, 528.2],  # coordonnées du point sur le bord supérieur du nombre visible 20 dans l'image avec caméra inclinée
	# 	],
	# 	dtype=np.float32,
	# )

	"""
	Position des points de la cible vue de la caméra 2 (caméra inclinée) :
	"""
	# pts_src contient les points mesurés dans l'image avec caméra inclinée. [x,y]
	# pts_src_cam2 = np.array(
	# 	[
	# 		[983.9, 300.5],  # coordonnées du point sur le bord supérieur du nombre visible 1 dans l'image de référence
	# 		[1003.9, 359.2], # coordonnées du point sur le bord supérieur du nombre visible 18 dans l'image de référence
	# 		[991.5, 424.7],  # coordonnées du point sur le bord supérieur du nombre visible 4 dans l'image de référence
	# 		[937.0, 490.0],  # coordonnées du point sur le bord supérieur du nombre visible 13 dans l'image de référence
	# 		[839.4, 539.6],  # coordonnées du point sur le bord supérieur du nombre visible 6 dans l'image de référence
	# 		[718.4, 562.0],  # coordonnées du point sur le bord supérieur du nombre visible 10 dans l'image de référence
	# 		[586.0, 552.0],  # coordonnées du point sur le bord supérieur du nombre visible 15 dans l'image de référence
	# 		[474.6, 511.0],  # coordonnées du point sur le bord supérieur du nombre visible 2 dans l'image de référence
	# 		[404.6, 449.5],  # coordonnées du point sur le bord supérieur du nombre visible 17 dans l'image de référence
	# 		[378.1, 383.3],  # coordonnées du point sur le bord supérieur du nombre visible 3 dans l'image de référence
	# 		[387.1, 321.3],  # coordonnées du point sur le bord supérieur du nombre visible 16 dans l'image de référence
	# 		[422.5, 268.3],  # coordonnées du point sur le bord supérieur du nombre visible 7 dans l'image de référence
	# 		[473.9, 228.5],  # coordonnées du point sur le bord supérieur du nombre visible 19 dans l'image de référence
	# 		[535.1, 200.2],  # coordonnées du point sur le bord supérieur du nombre visible 8 dans l'image de référence
	# 		[604.1, 182.9],  # coordonnées du point sur le bord supérieur du nombre visible 11 dans l'image de référence
	# 		[675.2, 175.4],  # coordonnées du point sur le bord supérieur du nombre visible 14 dans l'image de référence
	# 		[745.5, 179.6],  # coordonnées du point sur le bord supérieur du nombre visible 9 dans l'image de référence
	# 		[814.7, 192.0],  # coordonnées du point sur le bord supérieur du nombre visible 12 dans l'image de référence
	# 		[880.1, 215.5],  # coordonnées du point sur le bord supérieur du nombre visible 5 dans l'image de référence
	# 		[937.6, 249.4],  # coordonnées du point sur le bord supérieur du nombre visible 20 dans l'image de référence
	# 	],
	# 	dtype=np.float32,
	# )

	"""
	Position des points de la cible vue de la caméra 3 (caméra inclinée) :
	"""
	# pts_src contient les points mesurés dans l'image avec caméra inclinée. [x,y]
	pts_src_cam3 = np.array(
		[
			[581.8, 157.1],  # coordonnées du point sur le bord supérieur du nombre visible 1 dans l'image de référence
			[651.0, 141.9],  # coordonnées du point sur le bord supérieur du nombre visible 18 dans l'image de référence
			[721.8, 136.6],  # coordonnées du point sur le bord supérieur du nombre visible 4 dans l'image de référence
			[792.8, 142.2],  # coordonnées du point sur le bord supérieur du nombre visible 13 dans l'image de référence
			[862.0, 157.3],  # coordonnées du point sur le bord supérieur du nombre visible 6 dans l'image de référence
			[925.3, 183.7],  # coordonnées du point sur le bord supérieur du nombre visible 10 dans l'image de référence
			[980.7, 222.6],  # coordonnées du point sur le bord supérieur du nombre visible 15 dans l'image de référence
			[1016.8, 273.8],  # coordonnées du point sur le bord supérieur du nombre visible 2 dans l'image de référence
			[1029.0, 334.8],  # coordonnées du point sur le bord supérieur du nombre visible 17 dans l'image de référence
			[1004.0, 401.6],  # coordonnées du point sur le bord supérieur du nombre visible 3 dans l'image de référence
			[934.4, 463.4],  # coordonnées du point sur le bord supérieur du nombre visible 16 dans l'image de référence
			[824.8, 507.4],  # coordonnées du point sur le bord supérieur du nombre visible 7 dans l'image de référence
			[692.4, 520.6],  # coordonnées du point sur le bord supérieur du nombre visible 19 dans l'image de référence
			[566.1, 499.5],  # coordonnées du point sur le bord supérieur du nombre visible 8 dans l'image de référence
			[470.4, 451.3],  # coordonnées du point sur le bord supérieur du nombre visible 11 dans l'image de référence
			[416.1, 389.0],  # coordonnées du point sur le bord supérieur du nombre visible 14 dans l'image de référence
			[403.4, 324.5],  # coordonnées du point sur le bord supérieur du nombre visible 9 dans l'image de référence
			[421.6, 266.8],  # coordonnées du point sur le bord supérieur du nombre visible 12 dans l'image de référence
			[462.5, 218.9],  # coordonnées du point sur le bord supérieur du nombre visible 5 dans l'image de référence
			[518.0, 182.1],  # coordonnées du point sur le bord supérieur du nombre visible 20 dans l'image de référence
		],
		dtype=np.float32,
	)

	# pts_dst contient les points exactement correspondants dans la vue de face (reference prise avec cam de l'arceau, un peu inclinée)
	# A ne pas utiliser pour l'homographie
	# pts_dst = np.array(
	# 	[
	# 		[735.5, 113.9],  # position du point sur le bord supérieur du nombre visible 1 dans l'image de référence
	# 		[799.1, 155.2],  # position du point sur le bord supérieur du nombre visible 18 dans l'image de référence
	# 		[846.9, 216.4],  # position du point sur le bord supérieur du nombre visible 4 dans l'image de référence
	# 		[875.0, 288.9],  # position du point sur le bord supérieur du nombre visible 13 dans l'image de référence
	# 		[877.2, 367.6],  # position du point sur le bord supérieur du nombre visible 6 dans l'image de référence
	# 		[854.7, 443.6],  # position du point sur le bord supérieur du nombre visible 10 dans l'image de référence
	# 		[810.1, 508.9],  # position du point sur le bord supérieur du nombre visible 15 dans l'image de référence
	# 		[745.7, 556.3],  # position du point sur le bord supérieur du nombre visible 2 dans l'image de référence
	# 		[670.4, 580.7],  # position du point sur le bord supérieur du nombre visible 17 dans l'image de référence
	# 		[591.2, 580.4],  # position du point sur le bord supérieur du nombre visible 3 dans l'image de référence
	# 		[516.2, 554.3],  # position du point sur le bord supérieur du nombre visible 16 dans l'image de référence
	# 		[453.9, 506.8],  # position du point sur le bord supérieur du nombre visible 7 dans l'image de référence
	# 		[409.3, 444.0],  # position du point sur le bord supérieur du nombre visible 19 dans l'image de référence
	# 		[387.8, 371.3],  # position du point sur le bord supérieur du nombre visible 8 dans l'image de référence
	# 		[387.8, 295.3],  # position du point sur le bord supérieur du nombre visible 11 dans l'image de référence
	# 		[411.9, 223.0],  # position du point sur le bord supérieur du nombre visible 14 dans l'image de référence
	# 		[457.0, 162.3],  # position du point sur le bord supérieur du nombre visible 9 dans l'image de référence
	# 		[516.2, 117.4],  # position du point sur le bord supérieur du nombre visible 12 dans l'image de référence
	# 		[587.7, 93.8],   # position du point sur le bord supérieur du nombre visible 5 dans l'image de référence
	# 		[662.5, 91.6],   # position du point sur le bord supérieur du nombre visible 20 dans l'image de référence
	# 	],
	# 	dtype=np.float32,
	# )

	# pts_dst contient les points exactement correspondants dans la vue de face.(ref mathias iPhone)
	pts_ref_iphone = np.array(
		[
			[755.8, 131.9],  # position du point sur le bord supérieur du nombre visible 1 dans l'image de référence
			[820.0, 178.9],  # position du point sur le bord supérieur du nombre visible 18 dans l'image de référence
			[867.0, 241.9],  # position du point sur le bord supérieur du nombre visible 4 dans l'image de référence
			[891.0, 317.2],  # position du point sur le bord supérieur du nombre visible 13 dans l'image de référence
			[891.8, 395.8],  # position du point sur le bord supérieur du nombre visible 6 dans l'image de référence
			[868.0, 471.1],  # position du point sur le bord supérieur du nombre visible 10 dans l'image de référence
			[822.2, 534.8],  # position du point sur le bord supérieur du nombre visible 15 dans l'image de référence
			[758.0, 581.6],  # position du point sur le bord supérieur du nombre visible 2 dans l'image de référence
			[684.1, 607.0],  # position du point sur le bord supérieur du nombre visible 17 dans l'image de référence
			[605.9, 607.7],  # position du point sur le bord supérieur du nombre visible 3 dans l'image de référence
			[529.5, 583.5],  # position du point sur le bord supérieur du nombre visible 16 dans l'image de référence
			[465.5, 537.8],  # position du point sur le bord supérieur du nombre visible 7 dans l'image de référence
			[418.3, 473.9],  # position du point sur le bord supérieur du nombre visible 19 dans l'image de référence
			[393.0, 398.6],  # position du point sur le bord supérieur du nombre visible 8 dans l'image de référence
			[391.9, 320.6],  # position du point sur le bord supérieur du nombre visible 11 dans l'image de référence
			[416.0, 244.2],  # position du point sur le bord supérieur du nombre visible 14 dans l'image de référence
			[463.8, 179.7],  # position du point sur le bord supérieur du nombre visible 9 dans l'image de référence
			[526.8, 132.5],  # position du point sur le bord supérieur du nombre visible 12 dans l'image de référence
			[601.6, 107.8],  # position du point sur le bord supérieur du nombre visible 5 dans l'image de référence
			[680.7, 107.9],  # position du point sur le bord supérieur du nombre visible 20 dans l'image de référence
		],
		dtype=np.float32,
	)

	# print("\nPoints source (image avec caméra inclinée) :")
	# print(pts_src_cam1)
	# print("\nPoints desref_iphoneation (image de référence) :")
	# print(pts_dst)

	
	# findHomography cherche la matrice H qui vérifie, pour chaque point source
	# (x,y), la relation suivante : (x,y)' ~ H (x,y).
	# Le symbole ~ signifie "égal à un facteur d'échelle près".
	homography_matrix, _ = cv2.findHomography(pts_src_cam3, pts_ref_iphone)

	if homography_matrix is None:
		raise RuntimeError(
			"Impossible de calculer l'homographie. Vérifiez que les points "
			"ne sont pas alignés et qu'ils correspondent bien entre les images."
		)

	print("\nMatrice d'homographie H :")
	print(homography_matrix)

	# ---------- Test sur un point d'exemple ----------
	# Le point d'exemple représente ici une position détectée de la pointe de la fléchette
	# dans l'image prise avec caméra inclinée. On applique la transformation pour savoir où il
	# se projette dans la vue corrigée de face.
	test_point_camera = np.array([[[780, 400]]], dtype=np.float32)
	test_point_camera_x = test_point_camera[0, 0, 0]
	test_point_camera_y = test_point_camera[0, 0, 1]
	# On applique la transformation d'homographie au point test (ici sur un seul point)
	transformed_point = cv2.perspectiveTransform(test_point_camera, homography_matrix)
	# Point transformé brut (format homogène) : [[[x' y']]]
	
	transformed_x = float(transformed_point[0, 0, 0])
	transformed_y = float(transformed_point[0, 0, 1])

	print(f"\nPoint test dans l'image avec caméra inclinée : ({test_point_camera_x}, {test_point_camera_y})")
	print(
		"Point transformé dans l'image de référence : "
		f"({transformed_x:.2f}, {transformed_y:.2f})"
	)

	# warpPerspective applique H à tous les pixels de l'image avec caméra inclinée.
	# On obtient alors une image rectifiée, comme si la caméra regardait la
	# cible parfaitement de face.
	camera_image_height, camera_image_width = camera_image.shape[:2]
	# On applique la transformation d'homographie à l'image entière pour obtenir une image corrigée. (ici sur tous les points de l'image)
	corrected_image = cv2.warpPerspective(
		camera_image,
		homography_matrix,
		(camera_image_width, camera_image_height),
	)

	# -------------------------------------------------------------------------
	# Affichage des points choisis + sauvegarde des images traçage des points
	# Cela permet de contrôler visuellement que les correspondances sont bien
	# placées sur les numéros choisis.
	camera_image_with_points = draw_points(camera_image, pts_src_cam3, ["1", "18", "4", "13", "6", "10", "15", "2", "17", "3", "16", "7", "19", "8", "11", "14", "9", "12", "5", "20"])
	camera_image_with_points = draw_points(camera_image_with_points,test_point_camera.reshape(1, 2),["Point test"])
	cv2.imwrite(str(base_dir / 'cam3_avec_points.jpg'), camera_image_with_points)

	reference_image_with_points = draw_points(reference_image, pts_ref_iphone, ["1", "18", "4", "13", "6", "10", "15", "2", "17", "3", "16", "7", "19", "8", "11", "14", "9", "12", "5", "20"])
	# cv2.imwrite(str(base_dir / 'ref_iphone_avec_points.jpg'), reference_image_with_points)

	# enregistre l'image corrigée pour pouvoir la comparer avec les résultats du modèle IA
	circle_center, circle_radius = detect_concentric_circles(corrected_image)
	print("\nCentre des cercles détecté :", circle_center)
	print("Rayon détecté :", circle_radius)

	corrected_image = draw_points(corrected_image,transformed_point.reshape(1, 2),["Point test corrige"])
	# corrected_image = draw_concentric_circles(corrected_image, circle_center, circle_radius)
	cv2.imwrite(str(base_dir / 'cam3_homographie_avec_points.jpg'), corrected_image)


	# -------------------------------------------------------------------------
	# Visualisation des résultats

	# (pas obligatoire) mais je peux resize les images pour qu'elles tiennent mieux à l'écran
	# WIDTH_RESIZE, HEIGHT_RESIZE = 500, 500
	# rz_camera_image_with_points = cv2.resize(camera_image_with_points, (WIDTH_RESIZE, HEIGHT_RESIZE))
	# rz_reference_image_with_points = cv2.resize(reference_image_with_points, (WIDTH_RESIZE, HEIGHT_RESIZE))
	# rz_corrected_image = cv2.resize(corrected_image, (WIDTH_RESIZE, HEIGHT_RESIZE))

	cv2.imshow("Image originale - camera inclinee", camera_image_with_points)
	cv2.imshow("Image de reference", reference_image_with_points)
	# cv2.imshow("Image corrigee", rz_corrected_image)
	plt.title("Image corrigée (homographie)")
	plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
	plt.show()
	plt.pause(0.001)

	print("\nAppuyez sur n'importe quelle touche pour arrêter le programme.")
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()