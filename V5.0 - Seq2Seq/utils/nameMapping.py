import numpy as np
from utils.notes import genNotes_v2
from keras.preprocessing.text import Tokenizer

""" 
Have this as a file to avoid unescessary processing
"""
GUITAR_NOTES = genNotes_v2("F#1", "A6")
note_tokenizer = Tokenizer(lower=False, filters="")
note_tokenizer.fit_on_texts(GUITAR_NOTES)
note_tokenizer = note_tokenizer.word_index


codeToName = {
    -1:'Previously played',-2:'None',5:'Note keeps sounding'
}

def convert(code:int):
    if code ==-1: return codeToName[-1]
    elif code ==-2: return codeToName[-2]
    elif code == 5: return codeToName[5]
    else: return code
    
def printTimes(arrays):
    for timesteps in arrays:
        for times in timesteps:
            timein = times[0]
            timeout = times[1]
            print(f"{convert(timein)} - {convert(timeout)}")
            
def printNotesCategorical(arrays):
    for timesteps in arrays:
        for notes in timesteps:
            noteIndex = np.argmax(notes)
            if noteIndex!=0:
                print(list(note_tokenizer.keys())[
                    list(note_tokenizer.values()).index(noteIndex)
                ])
            else:
                print('None')
            