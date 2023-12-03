import os
import soundfile as sf
import numpy as np
from Helpers.consts.paths import datasetpath, convertDatasetpath


def convert_24bit_to_16bit(audio_data):
    # Ensure that the audio data is in the range [-1.0, 1.0]
    audio_data = np.clip(audio_data, -1.0, 1.0)

    # Convert 24-bit audio (range: [-2^23, 2^23 - 1]) to 16-bit audio (range: [-2^15, 2^15 - 1])
    audio_data = np.round(audio_data * 32767).astype(np.int16)

    return audio_data


def convert_audio_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    audio_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

    print(
        f"Converting {len(audio_files)} audio files from {input_folder} to {output_folder}..."
    )

    for audio_file in audio_files:
        input_path = os.path.join(input_folder, audio_file)
        output_path = os.path.join(output_folder, audio_file)

        # Load the 24-bit audio
        audio_data, sample_rate = sf.read(input_path, dtype="int32")

        # Convert to 16-bit
        audio_data = convert_24bit_to_16bit(audio_data)

        # Save as a 16-bit WAV file
        sf.write(output_path, audio_data, sample_rate, subtype="PCM_16")

        print(f"Converted and saved: {output_path}")
