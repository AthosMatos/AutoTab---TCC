from utils.audio.load_prepare_CURRENT import load
from utils.paths import RAW_DATASETS
from historic.getOnsets import get_onsets
from historic.AudioWindow_mix import audio_window

""" 
obs:
so far the best model dataset has been the one without the notes augmentation, tryng now
with augmentation
"""


def main():
    # MUSICTESTPATH = RAW_DATASETS.path + "/musics/g chord.wav"
    # MUSICTESTPATH = RAW_DATASETS.path + "/musics/beach house - clean.wav"
    # MUSICTESTPATH = RAW_DATASETS.path + "/musics/my bron-yr-aur.mp3"
    # MUSICTESTPATH = RAW_DATASETS.path + "/musics/TEST SONG.wav"
    # MUSICTESTPATH = RAW_DATASETS.path + "/musics/simple notes test.wav"
    # MUSICTESTPATH = RAW_DATASETS.path + "/musics/fastnotestSeq.wav"
    MUSICTESTPATH = RAW_DATASETS.path + "/musics/riffs test.wav"
    # MUSICTESTPATH = RAW_DATASETS.path + "/musics/riff test 3 notes.wav"
    # MUSICTESTPATH = "output.wav"

    """
    riff test notes in riff
    1- A2 E3 A3
    2- D3 A3 D4
    3 - E3 B3 E4
    
    beach notes 
    D3 G3 A3 C4 A3 C4 D4 E4 G4 C4 D4 D4 D#4 A3 C4 A3 D3 G3 A3 C4 A3
    """

    SR = 44100
    AUDIO, _ = load(MUSICTESTPATH, sample_rate=SR)
    ONSETS_SECS, ONSETS_SRS = get_onsets(AUDIO, SR)
    audio_window(
        AUDIO,
        SR,
        (ONSETS_SECS, ONSETS_SRS),
        MaxSteps=20,  # experimental, incrases the end of the audio to get more context
        # justNotes=True,
    )


if __name__ == "__main__":
    main()
