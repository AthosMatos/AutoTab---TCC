import asyncio
from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosedOK
import json
import numpy as np
import matplotlib.pyplot as plt
from historic.getOnsets import get_onsets
from historic.AudioWindow import audio_window
from keras.models import load_model


class Msg:
    data: np.ndarray

    def __init__(self, data):
        # dat is a list of strings representing a int8bit audio
        self.data = np.array(data["data"])


# params with respectives types
async def handler(websocket: WebSocketServerProtocol, path: str):
    def loadJson(data):
        dataJson: str = json.loads(data)
        msg: Msg = Msg(dataJson)
        return msg

    try:
        msgs = 0
        audio = []
        async for message in websocket:
            # Process the received message
            dataJson = loadJson(message)
            AUDIO = dataJson.data
            audio.extend(AUDIO)
            msgs += 1
            if msgs == 4:
                print("4")

                audio = np.array(audio)
                print(audio.shape)
                audio = np.array(audio)
                print(audio.shape)
                SR = 44100
                # plot audio with max limit being 1 and min limit being -1
                plt.plot(audio)
                plt.ylim(-1, 1)
                plt.show()

                ONSETS_SECS, ONSETS_SRS = get_onsets(audio, SR)
                # model = load_model("Models/model-out-6-Adam-bigDS.h5")
                model = load_model("Models/model.h5")
                audio_window(audio, (ONSETS_SECS, ONSETS_SRS), SR, model, MaxSteps=20)

            """  # Respond to the client (optional)
            response_message = f"Server received: {message}"
            await websocket.send(response_message) """

        # You can perform any additional processing or logic here
        # For example, you might want to send a response back to the client

        # plt.show()

    except ConnectionClosedOK as e:
        print(f"WebSocket connection closed with status code {e.code}")
    finally:
        await websocket.close()


start_server = serve(handler, "localhost", 50007)

print(f"WebSocket server running at url: ws://localhost:50007")

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
