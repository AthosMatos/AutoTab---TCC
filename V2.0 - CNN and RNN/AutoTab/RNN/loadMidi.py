import mido
from mido import MidiFile, MidiTrack, Message
import os 

mid = MidiFile(os.path.dirname(__file__) + "/newDS/miditest.midi")

for i, track in enumerate(mid.tracks):
    print(f"Track {i}: {track.name}")
    for msg in track:
        print(msg)
