
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from loadAudio import LoadAudio
from librosa.display import specshow

audioClass = LoadAudio('my bron-yr-aur.wav')
audio_spec = audioClass.getStft()
nfft = audioClass.getNftt()
hop_length = audioClass.getHopLength()

sr = audioClass.getSr()
print('audio_spec shape: ', audio_spec.shape)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

audio_window = 1024
jump_window = 64
plt.xlabel('Time Seconds')
plt.ylabel('Frequency')
plt.title('Live Audio Spectrogram')
plt.figtext(0.5, 0.92, 'audio window: ' + str(audio_window) + ' samples' +
            '\n jump window: ' + str(jump_window) + ' samples', ha='center', va='center')

img = specshow(audio_spec[:, :audio_window], x_axis='time', y_axis='log',
               sr=sr, ax=ax1, n_fft=nfft, hop_length=hop_length)


def animate(i):
    i = i * jump_window
    ys = audio_spec[:, i:audio_window + i]
    # ax1.clear()
    img.set_array(ys)

    return img,


anim = FuncAnimation(
    fig, animate, frames=audio_spec.shape[1], interval=100, blit=True)

plt.show()
