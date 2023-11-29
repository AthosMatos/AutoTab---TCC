import os


MAIN_PATH = os.getcwd().split("V4.0 - Multi-output CNN")[0]
DATASETS = os.path.join(MAIN_PATH, "Datasets")
NUMPY_DATASETS = os.path.join(DATASETS, "Numpy")
MODEL_PATH = os.path.join("autotabModels", "model-out6.h5")


class CUSTOM_DATASETS:
    path = os.path.join(DATASETS, "Custom")
    none = os.path.join(DATASETS, "Custom", "none")

    class IDMT_SMT_GUITAR_V2:
        path = os.path.join(DATASETS, "Custom", "IDMT-SMT-GUITAR_V2")
        chords = os.path.join(DATASETS, "Custom", "IDMT-SMT-GUITAR_V2", "chords")
        notes = os.path.join(DATASETS, "Custom", "IDMT-SMT-GUITAR_V2", "notes")

    class GuitarSet:
        path = os.path.join(DATASETS, "Custom", "GuitarSet")
        chords = os.path.join(DATASETS, "Custom", "GuitarSet", "chords")
        notes = os.path.join(DATASETS, "Custom", "GuitarSet", "notes")

    """ chords = os.path.join(DATASETS, "Custom", "chords")
    notes = os.path.join(DATASETS, "Custom", "notes")
    none = os.path.join(DATASETS, "Custom", "none") """


class RAW_DATASETS:
    path = os.path.join(DATASETS, "Raw")
    AthosSet = os.path.join(DATASETS, "Raw", "AthosSet")
    GuitarSet = os.path.join(DATASETS, "Raw", "GuitarSet")
    IDMT_SMT_GUITAR_V2 = os.path.join(DATASETS, "Raw", "IDMT-SMT-GUITAR_V2")
