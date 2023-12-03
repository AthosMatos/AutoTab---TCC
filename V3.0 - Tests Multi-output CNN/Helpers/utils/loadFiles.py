import os
import numpy as np


def getFilesPATHS(path: str, extension=".wav"):
    paths = []
    for DIRS, _, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                paths.append(os.path.join(DIRS, file))
    return np.array(paths)


def findFilePath(
    filename,
    path,
):
    extension = filename.split(".")[-1]
    for root, _, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            if filepath.endswith(extension):
                # print(filepath)
                # print(filename)
                if filepath.split("\\")[-1] == filename:
                    return str(filepath)

    return None
