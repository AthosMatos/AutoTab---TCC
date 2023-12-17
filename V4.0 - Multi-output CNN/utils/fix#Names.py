from utils.paths import CUSTOM_DATASETS
import os

# some of the folders in the dataset contain the hash
# symbol the wrong way, like this (♯) instead of this (#)
# this script fixes that

for DIRS, folders, files in os.walk(CUSTOM_DATASETS.path):
    for folder in folders:
        if "♯" in folder:
            print(f"Renaming {folder}")
            os.rename(
                os.path.join(DIRS, folder), os.path.join(DIRS, folder.replace("♯", "#"))
            )
