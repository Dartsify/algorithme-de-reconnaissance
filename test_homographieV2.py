"""
Correction de perspective pour une caméra inclinée.
Ici on travaille avec la caméra 2 (cf. numéro sur l'arceau)

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
		cv2.circle(annotated_image, (x, y), 8, (0, 255, 0), thickness=-1)
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
	base_dir = Path(__file__).parent / "image_test"
	camera_image_path = base_dir / "image_camera2_inclinee.jpeg"
	reference_image_path = base_dir / "image_reference.jpeg"

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
	# Par exemple, je peux ajouter tous les points sur les bords supérieurs des numéros visibles dans les images 
	# C'est-à-dire passer de 4 à 20 points de correspondance (à voir).
	# -----
	
	# pts_src contient les points mesurés dans l'image avec caméra inclinée. [x,y]
	pts_src = np.array(
		[
			[65.9, 674.7],  # coordonnées du point sur le bord supérieur du nombre visible 7 dans l'image avec caméra inclinée
			[583.9, 374.3],  # coordonnées du point sur le bord supérieur du nombre visible 9 dans l'image avec caméra inclinée
			[563.8, 1204.6],  # coordonnées du point sur le bord supérieur du nombre visible 15 dans l'image avec caméra inclinée
			[1087.7, 690.3],  # coordonnées du point sur le bord supérieur du nombre visible 18 dans l'image avec caméra inclinée
		],
		dtype=np.float32,
	)

	# pts_dst contient les points exactement correspondants dans la vue de face.
	pts_dst = np.array(
		[
			[244.4, 1167.9],  # position du point sur le bord supérieur du nombre visible 7 dans l'image de référence
			[166.4, 415.0],  # position du point sur le bord supérieur du nombre visible 9 dans l'image de référence
			[1013.6, 1057.7],  # position du point sur le bord supérieur du nombre visible 15 dans l'image de référence
			[888.4, 316.3],  # position du point sur le bord supérieur du nombre visible 18 dans l'image de référence
		],
		dtype=np.float32,
	)

	print("\nPoints source (image avec caméra inclinée) :")
	print(pts_src)
	print("\nPoints destination (image de référence) :")
	print(pts_dst)

	
	# findHomography cherche la matrice H qui vérifie, pour chaque point source
	# (x,y), la relation suivante : (x,y)' ~ H (x,y).
	# Le symbole ~ signifie "égal à un facteur d'échelle près".
	homography_matrix, _ = cv2.findHomography(pts_src, pts_dst)

	if homography_matrix is None:
		raise RuntimeError(
			"Impossible de calculer l'homographie. Vérifiez que les 4 points "
			"ne sont pas alignés et qu'ils correspondent bien entre les images."
		)

	print("\nMatrice d'homographie H :")
	print(homography_matrix)

	# ---------- Test sur un point d'exemple ----------
	# Le point (394.9, 768.9) représente ici une position détectée de la pointe de la fléchette
	# d'une fléchette dans l'image avec caméra inclinée. On applique la transformation pour savoir où il
	# se projette dans la vue corrigée de face.
	test_point_camera = np.array([[[394.9, 768.9]]], dtype=np.float32)
	# On applique la transformation d'homographie au point test (ici sur un seul point)
	transformed_point = cv2.perspectiveTransform(test_point_camera, homography_matrix)
	# Point transformé brut (format homogène) : [[[x' y']]]
	
	transformed_x = float(transformed_point[0, 0, 0])
	transformed_y = float(transformed_point[0, 0, 1])

	print("\nPoint test dans l'image avec caméra  inclinée : (394.9, 768.9)")
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
	camera_image_with_points = draw_points(camera_image, pts_src, ["7", "9", "15", "18"])
	camera_image_with_points = draw_points(camera_image_with_points,test_point_camera.reshape(1, 2),["Point test"])
	# cv2.imwrite(str(base_dir / "image_camera2_with_points.jpeg"), camera_image_with_points)

	reference_image_with_points = draw_points(reference_image, pts_dst, ["7", "9", "15", "18"])
	# cv2.imwrite(str(base_dir / "image_reference_with_points.jpeg"), reference_image_with_points)

	# enregistre l'image corrigée pour pouvoir la comparer avec les résultats du modèle IA
	circle_center, circle_radius = detect_concentric_circles(corrected_image)
	print("\nCentre des cercles détecté :", circle_center)
	print("Rayon détecté :", circle_radius)

	corrected_image = draw_points(corrected_image,transformed_point.reshape(1, 2),["Point test corrige"])
	corrected_image = draw_concentric_circles(corrected_image, (582,727), circle_radius)



	# -------------------------------------------------------------------------
	# Visualisation des résultats

	# (pas obligatoire) mais je resize les images pour qu'elles tiennent mieux à l'écran
	WIDTH_RESIZE, HEIGHT_RESIZE = 500, 500
	rz_camera_image_with_points = cv2.resize(camera_image_with_points, (WIDTH_RESIZE, HEIGHT_RESIZE))
	rz_reference_image_with_points = cv2.resize(reference_image_with_points, (WIDTH_RESIZE, HEIGHT_RESIZE))
	rz_corrected_image = cv2.resize(corrected_image, (1152, 1536))

	# cv2.imwrite(str(base_dir / "image_cam2_corrigee1152x1536_with_circles.jpeg"), rz_corrected_image)

	# cv2.imshow("Image originale - camera inclinee", rz_camera_image_with_points)
	# cv2.imshow("Image de reference", rz_reference_image_with_points)
	# cv2.imshow("Image corrigee", rz_corrected_image)
	plt.title("Image corrigée (homographie)")
	plt.imshow(cv2.cvtColor(rz_corrected_image, cv2.COLOR_BGR2RGB))
	plt.show()
	# plt.pause(0.001)

	print("\nAppuyez sur n'importe quelle touche pour arrêter le programme.")
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()