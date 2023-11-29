from utils.notes import genNotes

# 30 frets because the dataset has harmonics that go beyond the 22 frets
GUITAR_NOTES, _, GUITAR_NOTES_INDEXES = genNotes(indexes=True, FRETS=30)

print(GUITAR_NOTES)

""" 
[
    'A2', 'A3', 'A4', 'A5', 'A6', 'A♯2', 'A♯3', 
    'A♯4', 'A♯5', 'B2', 'B3', 'B4', 'B5', 'C3', 
    'C4', 'C5', 'C6', 'C♯3', 'C♯4', 'C♯5', 'C♯6', 
    'D3', 'D4', 'D5', 'D6', 'D♯3', 'D♯4', 'D♯5',
    'D♯6', 'E2', 'E3', 'E4', 'E5', 'E6', 'F2', 
    'F3', 'F4', 'F5', 'F6', 'F♯2', 'F♯3', 'F♯4', 
    'F♯5', 'F♯6', 'G2', 'G3', 'G4', 'G5', 'G6', 
    'G♯2', 'G♯3', 'G♯4', 'G♯5', 'G♯6', 'none'
]
"""
