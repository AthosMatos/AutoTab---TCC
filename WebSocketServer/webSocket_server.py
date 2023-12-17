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

    try:
        async for message in websocket:
            # Process the received message

            # dataJson = loadJson(message)
            sample_width = 2  # 16-bit PCM audio
            channels = 1  # Mono audio
            sample_rate = 48000  # Sample rate in Hz
            file_name = "output.wav"
            AUDIO = message

            # Convert the received data to a NumPy array
            audio_array = np.frombuffer(AUDIO, dtype=np.int16)

            print(len(audio_array))

            # Create a Wave_write object
            wave_file = wave.open(file_name, "w")
            wave_file.setnchannels(channels)
            wave_file.setsampwidth(4)
            wave_file.setframerate(sample_rate)
            wave_file.writeframes(audio_array.tobytes())

            wave_file.close()

            # print(AUDIO)

            audio, sr = librosa.load(file_name, sr=44100, mono=True)
            audio = audio[20:]
            print("Sample rate:", sr, "\tSamples:", len(audio))
            plt.plot(audio)

            plt.show()

            ONSETS_SECS, ONSETS_SRS = get_onsets(audio, sr)
            # model = load_model("Models/model-out-6-Adam-bigDS.h5")
            model = load_model("Models/model-out-6-Adam-new.h5")
            audio_window(audio, (ONSETS_SECS, ONSETS_SRS), sr, model, MaxSteps=20)

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


start_server = serve(handler, "localhost", 50007)

print(f"WebSocket server running at url: ws://localhost:50007")

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
