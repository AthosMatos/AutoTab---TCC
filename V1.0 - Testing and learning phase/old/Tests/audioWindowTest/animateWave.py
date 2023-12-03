
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from loadAudio import LoadAudio

audioClass = LoadAudio('my bron-yr-aur.wav')
audio = audioClass.getAudio()
sr = audioClass.getSr()
plot_time = audioClass.getAudioTime()
samples = audioClass.getSamplesAmount()

plt.style.use('seaborn')
fig = plt.figure()
# creating a subplot
ax1 = fig.add_subplot(1, 1, 1)

audio_window = 8192
jump_window = 2048
# ax1.set_xbound(plot_time.min(), plot_time.max())
plt.xlabel('Time Seconds')
plt.ylabel('Amplitude')
plt.title('Live Audio Waveform')

plt.figtext(0.5, 0.92, 'audio window: ' + str(audio_window) + ' samples - ' +
            '{:.5f}'.format(audio_window / sr) + '(s)'
            '\n jump window: ' +
            str(jump_window) + ' samples - ' +
            '{:.5f}'.format(jump_window / sr) + '(s)',
            ha='center', va='center')


def animate(i):
    i = i * jump_window
    xs = plot_time[i:audio_window + i]
    ys = audio[i:audio_window + i]
    ax1.clear()
    ax1.plot(xs, ys)
    ax1.set_ybound(audio.min(), audio.max())

    # print('start: ', i, 'end: ', audio_window + i)


anim = FuncAnimation(fig, animate, frames=samples, interval=5)


""" anim.save('sine_wave.gif', writer='imagemagick') """

plt.show()
