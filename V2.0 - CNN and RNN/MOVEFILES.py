import os

notes_class = (
    "A3",
    "A4",
    "C4",
    "C5",
    "D3",
    "D4",
    "D5",
    "E3",
    "E4",
    "E5",
    "G3",
    "G4",
    "noise",
)
# move wav files to folders with their name

folders = []
files = []
ds_path = os.path.join(os.getcwd(), "dataset")


for root, dirs, _ in os.walk(os.path.join(ds_path, "training")):
    for dir in dirs:
        if dir in notes_class:
            folderpath = os.path.join(root, dir)
            folders.append([folderpath, dir])


for root, dirs, _ in os.walk(os.path.join(ds_path)):
    for dir in dirs:
        for file in os.listdir(os.path.join(root, dir)):
            if file.endswith("wav"):
                filepath = os.path.join(root, dir, file)
                filename = file.split(".")[0].split("-")[0].strip()
                files.append([filepath, filename])


for folder in folders:
    for file in files:
        # check if the string file is in the string folder, they are not exactly the same
        if file[1] == folder[1]:
            print("moving file: ", file[1])
            # check if exists a file with the same name
            if os.path.exists(os.path.join(folder[0], os.path.basename(file[0]))):
                print("file already exists")
                # if exists, add a number to the end of the file
                os.rename(
                    file[0],
                    os.path.join(
                        folder[0],
                        os.path.basename(file[0]).split(".")[0]
                        + "-1"
                        + "."
                        + os.path.basename(file[0]).split(".")[1],
                    ),
                )
            else:
                os.rename(file[0], os.path.join(folder[0], os.path.basename(file[0])))
