import React, { useEffect } from 'react';
import {
  ArrayQueue,
  ConstantBackoff,
  Websocket,
  WebsocketBuilder,
  WebsocketEvent,
} from "websocket-ts";
import toast, { Toaster } from 'react-hot-toast';

function App() {
  const ws = new WebsocketBuilder("ws://localhost:50007")
    .withBuffer(new ArrayQueue())           // buffer messages when disconnected
    .withBackoff(new ConstantBackoff(1000)) // retry every 1s
    .build();


  useEffect(() => {
    //async promise function
    const connectToWebSocket = () =>
      new Promise(
        async (
          resolve: (res: string) => void,
          reject: (res: string) => void
        ) => {
          ws.addEventListener(WebsocketEvent.open, () => {
            //ws.send("Hello World!");
            resolve("success");

          })
        })
    toast.promise(
      connectToWebSocket(),
      {
        loading: 'Conectando ao servidor de processamento...',
        success: <b>Conectado ao servidor!</b>,
        error: <b>Erro ao conectar!</b>,
      }
    );

  }, [ws])


  return (
    <div className="App">
      <Toaster
        position="top-center"
        reverseOrder={false}
      />
      <button onClick={() => ws.send("Hello World!")}>Send</button>

    </div>
  );
}

export default App;
