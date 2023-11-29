from TrainVars import notes_class,gain_class,amps_class
from keras.models import load_model
from audioUtils import loadAndPrepare
import os

model = load_model('test_modelSmallMultiClass.h5')

windowStart = 0.0
windowEnd = windowStart+0.5

audio, _ = loadAndPrepare(os.path.dirname(
    __file__) + '/dataset/musics/beach house - distortion.wav', (windowStart, windowEnd))
y_pred = model.predict(audio.reshape(1, audio.shape[0], audio.shape[1]))
os.system('cls')
print('|| Audio Window: ', windowStart, ' - ', windowEnd, ' ||')


# print('max: ', "{:.4f}".format(np.max(y_pred)*100), '%')  # confidence
# print('np.argmax(y_pred): ', np.argmax(y_pred))  # index
# print('Predicted note: ', all_notes[np.argmax(y_pred)])  # note
# each pred notes confidende

def printPred(NN_pred,pred_class,Name):
    print(Name)

    for index, pred in enumerate(NN_pred):
        print(pred_class[index], ': ', "{:.4f}%".format(pred*100))


printPred(y_pred[0][0],notes_class,'|| Notes ||')
printPred(y_pred[1][0],gain_class,'|| Gain ||')
printPred(y_pred[2][0],amps_class,'|| Amps ||')
