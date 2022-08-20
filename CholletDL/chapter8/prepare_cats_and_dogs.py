import os, shutil, pathlib
from tracemalloc import start

original_dir = pathlib.Path("/home/zilleplus/Downloads/cats_vs_dogs/PetImages")
new_base_dir = pathlib.Path(f"{pathlib.Path(__file__).parent}/cats_vs_dogs_small")

def make_subset(subset_name, start_index, end_index):
    for catergory in ("Cat", "Dog"):
        dir: pathlib.Path = new_base_dir / subset_name / catergory
        if not dir.exists():
            os.makedirs(dir)

        fnames = [f"{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir/catergory/fname, dst=dir/fname)


make_subset("train", start_index=0, end_index=1000)
make_subset("validation", start_index=1000, end_index=1500)
make_subset("test", start_index=1500, end_index=2500)