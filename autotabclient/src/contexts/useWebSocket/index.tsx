import { createContext, useContext, useEffect, useState } from "react";
import { ArrayQueue, ConstantBackoff, Websocket, WebsocketBuilder, WebsocketEvent } from "websocket-ts";
import toast, { Toaster } from "react-hot-toast";


interface WebsocketProviderI {
	children: React.ReactNode;
}
interface WebsocketContextI {
	connect: (url: string) => void;
	getWebSocket: () => Websocket;
	isConnected: boolean;
}
const WebsocketContext = createContext<WebsocketContextI>({} as any);

export const WebsocketProvider = ({ children }: WebsocketProviderI) => {
	const [ws, set_ws] = useState<Websocket>();
	const [isConnected, setIsConnected] = useState(false);

	function connect(url: string) {

		const checkIfConnected = (webSocket: Websocket) =>
			new Promise(async (resolve: (res: string) => void, reject: (res: string) => void) => {
				webSocket.addEventListener(WebsocketEvent.open, () => {
					//ws.send("Hello World!");
					resolve("success");
					setIsConnected(true);
				});
			});
		function ToastLoading(webSocket: Websocket) {
			toast.promise(checkIfConnected(webSocket), {
				loading: "Conectando ao servidor de processamento...",
				success: <b>Conectado ao servidor!</b>,
				error: <b>Erro ao conectar!</b>,
			});
		}

		const webS = new WebsocketBuilder(url)
			.withBuffer(new ArrayQueue()) // buffer messages when disconnected
			.withBackoff(new ConstantBackoff(1000)) // retry every 1s
			.build()
		set_ws(webS);
		ToastLoading(webS);


	}

	function getWebSocket() {
		if (!isConnected || !ws) {
			throw new Error("WebSocket is not connected");
		}
		return ws;
	}

	return (
		<WebsocketContext.Provider value={{ connect, getWebSocket, isConnected }}>
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

