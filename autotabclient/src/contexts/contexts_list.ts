import { PlaybackProvider } from "./PlaybackContext/usePlaybackContext";
import { TabViewProvider } from "./TabViewContext/useTabViewContext";
import { WebsocketProvider } from "./useWebSocket";

export const contextsList = [WebsocketProvider, TabViewProvider, PlaybackProvider];
