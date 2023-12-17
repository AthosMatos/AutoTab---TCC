import { useEffect, useReducer, useState } from "react";
import Tab, { SpecificNotes } from "./TabCanvas";
import { Container } from "./styled";
import { useTabViewContext } from "../../contexts/TabViewContext/useTabViewContext";
import { usePlaybackContext } from "../../contexts/PlaybackContext/usePlaybackContext";


interface TabViewI {
    windowSize: { width: number, height: number }
}

const TabView = (props: TabViewI) => {
    const { windowSize } = props
    const { activeAudioEvents } = useTabViewContext()
    const [timedActivatedChords, setTimedActivatedChords] = useState<SpecificNotes[]>([])
    const [aniIndex, setAnimationIndex] = useState<number>(0)
    const { stopPredict, Predict, } = usePlaybackContext()


    useEffect(() => {
        if (activeAudioEvents.length > 0) {
            Predict.show ?
                setTimedActivatedChords(activeAudioEvents[aniIndex ? aniIndex - 1 : 0].indexes)
                :
                setTimedActivatedChords([])
        }
    }, [Predict.show, activeAudioEvents.length])

    useEffect(() => {
        if (Predict.playing) {
            let animationIndex = aniIndex
            if (aniIndex == activeAudioEvents.length) {
                setAnimationIndex(0)
                animationIndex = 0
            }

            setTimeout(() => {
                const activatedChord = activeAudioEvents[animationIndex]
                if (animationIndex === activeAudioEvents.length - 1) {

                    if (Predict.loop) setAnimationIndex(0)
                    else {
                        stopPredict()
                    }
                }

                setTimedActivatedChords([])
                setTimeout(() => {
                    setTimedActivatedChords(activatedChord.indexes)
                }, 100)

                setAnimationIndex(animationIndex + 1)
            }, Predict.updateSpeed);
        }
    }, [Predict.playing, aniIndex])

    return (
        <Container>
            <Tab
                windowSize={{
                    width: windowSize.width
                }}
                findBy="index"
                activatedNotes={timedActivatedChords} />
        </Container>
    );
};

export default TabView;
