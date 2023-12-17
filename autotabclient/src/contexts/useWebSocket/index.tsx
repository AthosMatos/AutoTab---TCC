import { createContext, useContext, useEffect, useState } from "react";
import { ArrayQueue, ConstantBackoff, Websocket, WebsocketBuilder, WebsocketEvent } from "websocket-ts";
import toast from "react-hot-toast";


interface WebsocketProviderI {
	children: React.ReactNode;
}
interface WebsocketContextI {
	connect: (url: string) => void;
	getWebSocket: () => Websocket;
	send: (message: MsgT) => void;
	isConnected: boolean;
}

interface MsgT {
	data: any;
}

const WebsocketContext = createContext<WebsocketContextI>({} as any);

export const WebsocketProvider = ({ children }: WebsocketProviderI) => {
	const [ws, set_ws] = useState<Websocket>();
	const [isConnected, setIsConnected] = useState(false);

	function connect(url: string) {

		const checkIfConnected = (webSocket: Websocket) =>
			new Promise(async (resolve: (res: string) => void, reject: (res: string) => void) => {
				webSocket.addEventListener(WebsocketEvent.open, () => {
					//webSocket.send("Hello World!");
					resolve("success");
					setIsConnected(true);
				});
			});

		//check if url is a valid url
		if (!url || !url.includes("ws://")) {
			toast("Web socket URL inválida!", {
				style: {
					background: '#b94b4b',
					color: '#fff',
				},
			})
			return;
		}

		const webS = new WebsocketBuilder(url)
			.withBuffer(new ArrayQueue()) // buffer messages when disconnected
			.withBackoff(new ConstantBackoff(1000)) // retry every 1s
			.build()
		set_ws(webS);

		toast.promise(checkIfConnected(webS), {
			loading: "Conectando ao servidor de processamento...",
			success: <b>Conectado ao servidor!</b>,
			error: <b>Erro ao conectar!</b>,
		});


	}

	function getWebSocket() {
		if (!isConnected || !ws) {
			throw new Error("WebSocket is not connected");
		}
		return ws;
	}

	function send(message: MsgT, asJson?: boolean) {
		if (!isConnected || !ws) {
			console.log("WebSocket is not connected");
			return;
		}
		if (asJson) {
			const msg = JSON.stringify(message);
			ws.send(msg);
			return;
		}

		ws.send(message.data);
	}

	return (
		<WebsocketContext.Provider value={{ connect, getWebSocket, isConnected, send }}>
			{children}
		</WebsocketContext.Provider>
	);
};

export const useWebSocket = () => {
	const context = useContext(WebsocketContext);
	if (!context) {
		throw new Error(
			"useSpecificContext must be used within a SpecificProvider"
		);
	}
	return context;
};

