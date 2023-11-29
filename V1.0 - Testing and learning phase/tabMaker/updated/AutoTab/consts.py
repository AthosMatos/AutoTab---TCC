notes_class = ("A3", "A4", "C4", "C5", "D3",
               "D4", "D5", "E3", "E4", "E5", "G3", "G4", "noise")

gain_class = ("clean", "distortion", "none")
amps_class = ("fender", "marshall", "mesa", "none")  # "orange"
playin_class = ('pluck', 'running')  # 'strum'
playin_class2 = ('notes', 'none')  # 'chords',
# mix all the classes
all_classes = amps_class + playin_class + gain_class + playin_class2
