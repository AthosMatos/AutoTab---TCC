import os
import numpy as np
import os
from Helpers.consts.paths import training_ds_path

labels = []
""" labels_ignore = ["Major", "Minor"] """

for root, dirs, files in os.walk(training_ds_path):
    for dir in dirs:
        """if any(label in dir for label in labels_ignore):
        continue"""
        for file in os.listdir(os.path.join(root, dir)):
            if file.endswith("wav"):
                labels.append(dir)


labels = np.unique(labels)
