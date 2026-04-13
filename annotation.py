from pathlib import Path
import json

import cv2
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button

"""
Voici un label d'exemple pour une image dans le fichier "label.csv":
{
	"objects":
	[
		{
			"point":
			{
				"x":100,
				"y":200
			}
		},
		{
			"point":
			{
				"x":150,
				"y":250
			}
		},
		{
			"point":
			{
				"x":200,
				"y":300
			}
		}
	]
}
"""
ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "Dartsify" / "dataset"
LABEL_PATH = ROOT / "Dartsify" / "label.csv"


def list_images(image_dir: Path):
	return sorted(
		path for path in image_dir.iterdir()
		if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
	)


def load_image(image_path: Path):
	img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if img_bgr is None:
		raise FileNotFoundError(f"Image not found: {image_path}")
	return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def parse_position(position_text: str):
	text = position_text.strip()
	if not text or text == '""':
		return []
	if text.startswith('"') and text.endswith('"'):
		text = text[1:-1]
	data = json.loads(text)
	points = []
	for item in data.get("objects", []):
		point = item.get("point")
		if isinstance(point, dict) and {"x", "y"} <= point.keys():
			points.append((float(point["x"]), float(point["y"])))
			continue
		for value in item.values():
			if isinstance(value, dict) and {"x", "y"} <= value.keys():
				points.append((float(value["x"]), float(value["y"])))
				break
	return points


def format_position(points):
	objects = [{"point": {"x": round(x, 1), "y": round(y, 1)}} for x, y in points]
	return json.dumps({"objects": objects}, separators=(",", ":"))


def read_label_rows(label_path: Path):
	if not label_path.exists():
		return []

	rows = []
	with label_path.open("r", encoding="utf-8") as csv_file:
		lines = csv_file.read().splitlines()

	for line in lines[1:]:
		if not line.strip():
			continue
		image_id, _, position_text = line.partition(",")
		rows.append((image_id.strip().strip('"'), position_text.strip()))
	return rows


def load_points_for_image(label_path: Path, image_name: str):
	for existing_id, position_text in read_label_rows(label_path):
		if existing_id == image_name:
			return parse_position(position_text)
	return []


def write_label(label_path: Path, image_id: str, points):
	rows = read_label_rows(label_path)
	updated_lines = ['"ID","Position"']
	target_line = f'"{image_id}",{format_position(points)}'
	updated = False

	for existing_id, position_text in rows:
		if existing_id == image_id:
			updated_lines.append(target_line)
			updated = True
		else:
			updated_lines.append(f'"{existing_id}",{position_text}')

	if not updated:
		updated_lines.append(target_line)

	label_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def main():
	images = list_images(IMAGE_DIR)
	if not images:
		raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

	current_index = 0
	current_image = images[current_index]
	img = load_image(current_image)
	current_points = load_points_for_image(LABEL_PATH, current_image.name)

	fig, ax = plt.subplots()
	fig.subplots_adjust(bottom=0.2)
	ax.imshow(img)
	ax.set_title("")
	ax.set_xlabel("x")
	ax.set_ylabel("y")

	button_prev_ax = fig.add_axes([0.28, 0.05, 0.18, 0.08])
	button_next_ax = fig.add_axes([0.54, 0.05, 0.18, 0.08])
	button_prev = Button(button_prev_ax, "Previous")
	button_next = Button(button_next_ax, "Next")

	def redraw():
		ax.clear()
		ax.imshow(img)
		ax.set_title(
			f"{current_image.name} ({current_index + 1}/{len(images)}) - "
			"clic droit pour ajouter un point, u pour annuler, c pour effacer, q pour quitter"
		)
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		for index, (x, y) in enumerate(current_points, start=1):
			ax.scatter([x], [y], c="red", s=30)
			ax.text(x + 5, y + 5, str(index), color="yellow", fontsize=9)
		fig.canvas.draw_idle()

	def save():
		write_label(LABEL_PATH, current_image.name, current_points)
		print(f"Saved {len(current_points)} point(s) for {current_image.name} to {LABEL_PATH}")

	def load_image_at(index: int):
		nonlocal current_index, current_image, img, current_points
		current_index = index
		current_image = images[current_index]
		img = load_image(current_image)
		current_points = load_points_for_image(LABEL_PATH, current_image.name)
		redraw()

	def move_to(index: int):
		if index < 0 or index >= len(images):
			return
		save()
		load_image_at(index)

	def on_click(event):
		if event.inaxes != ax or event.button != MouseButton.RIGHT:
			return
		if event.xdata is None or event.ydata is None:
			return

		current_points.append((float(event.xdata), float(event.ydata)))
		redraw()
		save()

	def on_previous(event):
		_ = event
		move_to(current_index - 1)

	def on_next(event):
		_ = event
		move_to(current_index + 1)

	def on_key(event):
		if event.key == "u" and current_points:
			current_points.pop()
			redraw()
			save()
		elif event.key == "c":
			current_points.clear()
			redraw()
			save()
		elif event.key == "left":
			move_to(current_index - 1)
		elif event.key == "right":
			move_to(current_index + 1)
		elif event.key in {"q", "escape"}:
			plt.close(fig)

	fig.canvas.mpl_connect("button_press_event", on_click)
	fig.canvas.mpl_connect("key_press_event", on_key)
	button_prev.on_clicked(on_previous)
	button_next.on_clicked(on_next)
	redraw()
	plt.show()


if __name__ == "__main__":
	main()