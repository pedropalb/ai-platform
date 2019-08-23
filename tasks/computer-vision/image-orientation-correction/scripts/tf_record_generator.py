from pathlib import Path
import cv2


def generate(input_dir, output_dir, output_filename, index_path):
    input_dir = Path(input_dir).resolve()

    with open(index_path, 'r') as f:
        for line in f:
            filename = line.strip()
            file_path = input_dir / filename

            image = cv2.imread(str(file_path))
