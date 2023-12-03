import os
import soundfile as sf
import numpy as np


def convert_24bit_to_16bit(audio_data):
    # Ensure that the audio data is in the range [-1.0, 1.0]
    audio_data = np.clip(audio_data, -1.0, 1.0)

    # Convert 24-bit audio (range: [-2^23, 2^23 - 1]) to 16-bit audio (range: [-2^15, 2^15 - 1])
    audio_data = np.round(audio_data * 32767).astype(np.int16)

    return audio_data


def convert_audio_files(input_folder):
    # List all files in the input folder
    audio_files = []
    # walk through all files in the folder
    for dirpath, dirnames, filenames in os.walk(input_folder):
        # print(f"Found directory: {dirpath}")
        for file_name in filenames:
            wav_file = os.path.join(dirpath, file_name)
            audio_files.append(wav_file)

    for audio_file in audio_files:
        input_path = os.path.join(input_folder, audio_file)
        output_path = input_path

        # Load the 24-bit audio
        audio_data, sample_rate = sf.read(input_path)

        # check if the file is already 16bit
        print(f"Converting {audio_file}...")

        # Convert to 16-bit
        audio_data = convert_24bit_to_16bit(audio_data)

        # Save as a 16-bit WAV file
        sf.write(output_path, audio_data, sample_rate, subtype="PCM_16")
