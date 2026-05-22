import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
image_dir = BASE_DIR / "dataset"

for image_path in image_dir.glob("*.jpg"):

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    cv2.imshow("Obstacle Detection", edges)

    cv2.waitKey(0)

cv2.destroyAllWindows()
