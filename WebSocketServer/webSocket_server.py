import asyncio
from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosedOK

# create handler for each connection


# params with respectives types
async def handler(websocket: WebSocketServerProtocol, path: str):
    try:
        data = await websocket.recv()
        print(data)
        """ reply = f"Data recieved as:  {data}!"
        
        await websocket.send(reply) """
    except ConnectionClosedOK as e:
        print(f"WebSocket connection closed with status code {e.code}")
    finally:
        await websocket.close()


start_server = serve(handler, "localhost", 50007)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
