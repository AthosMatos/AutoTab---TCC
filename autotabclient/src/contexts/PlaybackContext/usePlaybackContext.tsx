import { createContext, useContext, useState } from "react";


interface PlaybackContextI {
    Predict: PredictT;
    playPredict: () => void;
    togglePredict: () => void;
    toggleShow: () => void;
    toggleLoop: () => void;
    stopPredict: () => void;
    updatePredictSpeed: (speed: number) => void;
}
const TabViewContext = createContext<PlaybackContextI>({} as any);

interface PredictT {
    playing?: boolean;
    show?: boolean;
    loop?: boolean;
    updateSpeed?: number;
}

export const PlaybackProvider = (props: any) => {
    const [Predict, setPredict] = useState<PredictT>({
        playing: false,
        show: true,
        loop: false,
        updateSpeed: 1000
    });

    function playPredict() {
        setPredict({ ...Predict, playing: true })
    }
    function stopPredict() {
        setPredict({ ...Predict, playing: false })
    }
    function togglePredict() {
        setPredict({ ...Predict, playing: !Predict.playing })
    }

    function toggleShow() {
        setPredict({ ...Predict, show: !Predict.show })
    }

    function toggleLoop() {
        setPredict({ ...Predict, loop: !Predict.loop })
    }

    function updatePredictSpeed(speed: number) {
        setPredict({ ...Predict, updateSpeed: speed })
    }

    return (
        <TabViewContext.Provider value={{
            Predict,
            playPredict,
            toggleShow,
            togglePredict,
            toggleLoop,
            stopPredict,
            updatePredictSpeed
        }}>
            {props.children}
        </TabViewContext.Provider>
    );
};

export const usePlaybackContext = () => {

    const context = useContext(TabViewContext);
    if (!context) {
        throw new Error(
            "useSpecificContext must be used within a SpecificProvider"
        );
    }
    return context;
};
