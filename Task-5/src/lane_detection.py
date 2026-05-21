from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

image_dir = BASE_DIR / "data" / "raw"

image_paths = list(image_dir.glob("*.jpg"))

for image_path in image_paths:

    image = cv2.imread(str(image_path))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        2,
        np.pi/180,
        100,
        minLineLength=40,
        maxLineGap=5
    )

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line.reshape(4)

            cv2.line(
                image,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                5
            )

    cv2.imshow("Lane Detection", image)

    cv2.waitKey(0)

cv2.destroyAllWindows()