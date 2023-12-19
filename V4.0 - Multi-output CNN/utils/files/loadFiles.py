import os
import numpy as np


def getFilesPATHS(
    path: str,
    ignores: list[str] = None,
    extension=".wav",
    randomize=False,
    maxFiles=None,
):
    paths = []
    for DIRS, _, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                if ignores is not None and ignores.__len__() > 0:
                    if any(x in DIRS for x in ignores):
                        continue
                paths.append(os.path.join(DIRS, file))

    if randomize:
        np.random.shuffle(paths)

    if maxFiles is not None:
        paths = paths[:maxFiles]

    return np.array(paths)


def findFilePath(filename, path, pathCustomEnd=None):
    extension = filename.split(".")[-1]
    pathone = ""
    pathtwo = ""

    for root, _, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            if filepath.endswith(extension):
                """if filepath.split("\\")[-2] == "audio_mono-pickup_mix":
                print(filepath)"""
                # print(filepath.split("\\")[-1])
                # print(filename)
                # print()
                filePathName = filepath.split("\\")[-1]
                if pathCustomEnd is not None:
                    for p in pathCustomEnd:
                        if filePathName.split(p).__len__() > 1:
                            # remove just the p and not the extension
                            filePathName = filePathName.split(p)[0] + "." + extension
                            break

                    if filePathName == filename and pathone == "":
                        pathone = str(filepath)
                    elif filePathName == filename and pathtwo == "":
                        pathtwo = str(filepath)
                else:
                    if filePathName == filename:
                        return str(filepath)

    if pathone != "" and pathtwo != "":
        return pathone, pathtwo
    return None
