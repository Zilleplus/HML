from pathlib import Path
import os
import tensorflow as tf  # type: ignore

p = Path(__file__).parent / "cats_vs_dogs_small"
img_paths = os.walk(p)

bad_paths = []

files: list[str] = []
for root, dirs, files in img_paths:
    root_path = Path(root)
    for file in files:
        file_path: Path = Path(root) / file
        try:
            print(f"reading {file_path}")
            img_bytes = tf.io.read_file(str(file_path))
            decoded_img = tf.io.decode_image(img_bytes)
        except tf.errors.InvalidArgumentError as e:
            print(f"Found bad path {file_path}...{e}")
            bad_paths.append(file_path)

if len(bad_paths) > 0:
    print("BAD PATHS:")
    for bad_path in bad_paths:
        print(f"{bad_path}")
else:
    print("All ok")
