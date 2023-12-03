from Helpers.consts.paths import ds_path
import matplotlib.pyplot as plt
from Helpers.DataPreparation.audioUtils import load

path = ds_path + "/musics/beach house - clean.wav"
audio, sr = load(path)


plt.figure(figsize=(400, 5), facecolor="none")
# remove the x and y ticks
plt.xticks([])
plt.yticks([])

# plot the wavefor
plt.plot(audio)
plt.axis("off")
plt.box(False)

plt.tight_layout()

# save the plot
plt.savefig("waveform.png")

""" plt.show() """
