from prepareTrain import audio_inputs, notes_outputs,amps_outputs,gain_outputs,playS_outputs,playS2_outputs
from consts import gain_class,amps_class,all_classes,notes_class,playin_class,playin_class2

trainDataAmount = len(audio_inputs)

x_train = audio_inputs
y_train = notes_outputs