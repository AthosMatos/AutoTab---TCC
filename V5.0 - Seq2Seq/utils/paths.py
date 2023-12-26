import os


MAIN_PATH = os.getcwd().split("V5.0 - Seq2Seq")[0]
DATASETS = os.path.join(MAIN_PATH, "Datasets")
NUMPY_DATASETS = os.path.join(DATASETS, "Numpy")


class DATASET_UTILS:
    def createPath(path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    def getDatasetsNames():
        return ["IDMT-SMT-GUITAR_V2", "GuitarSet", "AthosSet"]


class CUSTOM_DATASETS:
    path = os.path.join(DATASETS, "Custom")
    none = os.path.join(DATASETS, "Custom", "none")

    class AUGMENTED:
        path = os.path.join(DATASETS, "Custom", "Augmented")

        class IDMT_SMT_GUITAR_V2:
            path = os.path.join(DATASETS, "Custom", "Augmented", "IDMT-SMT-GUITAR_V2")
            chords = os.path.join(
                DATASETS, "Custom", "Augmented", "IDMT-SMT-GUITAR_V2", "chords"
            )
            notes = os.path.join(
                DATASETS, "Custom", "Augmented", "IDMT-SMT-GUITAR_V2", "notes"
            )

        class GuitarSet:
            path = os.path.join(DATASETS, "Custom", "Augmented", "GuitarSet")
            chords = os.path.join(
                DATASETS, "Custom", "Augmented", "GuitarSet", "chords"
            )
            notes = os.path.join(DATASETS, "Custom", "Augmented", "GuitarSet", "notes")

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


class SEQ2SEQ:
    path = os.path.join(DATASETS, "Seq2Seq")

    class Frag:
        path = os.path.join(DATASETS, "Seq2Seq", "Frag")

    class Full:
        path = os.path.join(DATASETS, "Seq2Seq", "Full")


class RAW_DATASETS:
    path = os.path.join(DATASETS, "Raw")
    AthosSet = os.path.join(DATASETS, "Raw", "AthosSet")
    GuitarSet = os.path.join(DATASETS, "Raw", "GuitarSet")
    IDMT_SMT_GUITAR_V2 = os.path.join(DATASETS, "Raw", "IDMT-SMT-GUITAR_V2")
