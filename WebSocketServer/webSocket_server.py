import asyncio
from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosedOK
import json
import numpy as np
import matplotlib.pyplot as plt
from historic.getOnsets import get_onsets
from historic.AudioWindow import audio_window
from keras.models import load_model
import base64
import librosa
import soundfile as sf
import wave
import io
from sklearn.preprocessing import minmax_scale


class Msg:
    data: any

    def __init__(self, data):
        # dat is a list of strings representing a int8bit audio
        self.data = data["data"]


# params with respectives types
async def handler(websocket: WebSocketServerProtocol, path: str):
    def loadJson(data):
        dataJson: str = json.loads(data)
        msg: Msg = Msg(dataJson)
        return msg

    def is_blob(data):
        return isinstance(data, bytes)

    def handle_base64_audio(base64_data, output_filename="output.wav"):
        # Decode the base64 string to obtain binary data
        binary_data = base64.b64decode(base64_data)

        # Convert the binary data to a BytesIO object
        audio_io = io.BytesIO(binary_data)

        # Save the audio data as a WAV file
        save_audio_as_wav(audio_io, output_filename)

    def save_audio_as_wav(
        audio_io,
        output_filename="output.wav",
        channels=1,
        sample_width=4,
        frame_rate=48000,
    ):
        with wave.open(output_filename, "wb") as wave_file:
            wave_file.setnchannels(channels)
            wave_file.setsampwidth(sample_width)
            wave_file.setframerate(frame_rate)
            wave_file.writeframes(audio_io.getvalue())

    try:
        async for message in websocket:
            # Process the received message

            AUDIO = np.array(loadJson(message).data)

            # normalize between -1 and 1
            # AUDIO = minmax_scale(AUDIO, feature_range=(-1, 1))

            plt.plot(AUDIO)
            plt.ylim(-1, 1)
            plt.show()
            # handle_base64_audio(AUDIO)

            """  # Convert the received data to a NumPy array
            audio_array = np.frombuffer(AUDIO, dtype=np.int16)

            print(len(audio_array))

            # Create a Wave_write object
            wave_file = wave.open(file_name, "w")
            wave_file.setnchannels(channels)
            wave_file.setsampwidth(sample_width)
            wave_file.setframerate(sample_rate)
            wave_file.writeframes(AUDIO)

            wave_file.close() """

            return
            # print(AUDIO)

            audio, sr = librosa.load(file_name, sr=44100, mono=True)
            audio = audio[20:]
            print("Sample rate:", sr, "\tSamples:", len(audio))
            plt.plot(audio)
            plt.ylim(-1, 1)
            plt.show()

            """ ONSETS_SECS, ONSETS_SRS = get_onsets(audio, sr)
            print("ONSETS_SECS")
            audio_window(
                audio,
                sr,
                (ONSETS_SECS, ONSETS_SRS),
                MaxSteps=40,
                giveMoreAudioContext=False,  # experimental, incrases the end of the audio to get more context
                justNotes=False,
            ) """

            """  # Respond to the client (optional)
            response_message = f"Server received: {message}"
            await websocket.send(response_message) """

        # You can perform any additional processing or logic here
        # For example, you might want to send a response back to the client

        # plt.show()

        """ reply = f"Data recieved as:  {data}!"
        
        await websocket.send(reply) """
    except ConnectionClosedOK as e:
        print(f"WebSocket connection closed with status code {e.code}")
    finally:
        await websocket.close()


start_server = serve(handler, "localhost", max_size=None, port=50007)

print(f"WebSocket server running at url: ws://localhost:50007")

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
