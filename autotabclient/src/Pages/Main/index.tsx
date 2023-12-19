import { useEffect, useRef, useState } from "react"
import { useWebSocket } from "../../contexts/useWebSocket"
import { AudioEventAnalyser } from "../../utils/path"
import useWindowResize from "../../hooks/useWindowResize"
import { useTabViewContext } from "../../contexts/TabViewContext/useTabViewContext"
import TabView from "../../components/TabView"
import { usePlaybackContext } from "../../contexts/PlaybackContext/usePlaybackContext"
import { PlayButton } from "../../components/Buttons/Play"
import { PauseButton } from "../../components/Buttons/Stop"
import { ShowButton } from "../../components/Buttons/Show"
import { LoopButton } from "../../components/Buttons/Loop"
import { PredictButtonsContainer } from "../../components/Buttons"
import Recorder from "../../Recorder/src/recorder"
import useAudioRecorder from "../../hooks/useAudioRecorder"

const SeqTest3 = [
    "D6",
    ["D3", "A3", "D4", "F#4"],//DMAJOR
    "D4", "D4",
    ["A2", "E3", "A3", "C#4", "E4"],//AMAJOR 
    ["F2", "C3", "F3", "A3", "C4", "F4"], //FMAJOR
    "F4", "C3", "C3",
    ["G4", "A4", "B4"],
]





interface AudioDataI {
    blobObject: Blob | null,
    startTime: number | null,
}
const MainPage = () => {
    const windowSize = useWindowResize(0.9)
    const { allNotesFromFrets, frets, updateActivatedChords } = useTabViewContext()
    const chords = AudioEventAnalyser(SeqTest3, allNotesFromFrets, frets)
    const { updatePredictSpeed } = usePlaybackContext()
    const { connect, isConnected, send } = useWebSocket()
    const { startRecording, stopRecording, audioData } = useAudioRecorder()




    /* useEffect(() => {
        if (audioBlob && wavArrayBuffer) {
            // convert to base64
            function arrayBufferToBase64(arrayBuffer: ArrayBuffer): string {
                // Convert the ArrayBuffer to a binary string
                let binaryString = '';
                const bytes = new Uint8Array(arrayBuffer);
                const length = bytes.byteLength;

                for (let i = 0; i < length; i++) {
                    binaryString += String.fromCharCode(bytes[i]);
                }

                // Convert the binary string to a Base64 string
                return btoa(binaryString);
            }

            // Example usage
            const base64String = arrayBufferToBase64(wavArrayBuffer);


            send({ data: base64String })

            let url = URL.createObjectURL(audioBlob);
            let a = document.createElement('a');
            a.href = url;
            a.download = 'test.wav';
            a.click();
        }
    }, [audioBlob, wavArrayBuffer]); */

    useEffect(() => {
        connect('ws://localhost:50007')

        updatePredictSpeed(500)
        updateActivatedChords(chords.map((posi) => {
            const notes = posi.map((posi) => {
                return posi.note
            })
            return {
                notes: notes,
                indexes: posi.map((posi) => {
                    const pos = posi.pos
                    return {
                        string: pos.string,
                        fret: pos.fret
                    }
                })
            }
        }))
    }, [])

    /* useEffect(() => {
        isConnected && send({ data: 'chords' })
    }, [isConnected]) */

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
        }}>

            <button onClick={startRecording}>Start Recording</button>
            <button onClick={() => {
                stopRecording()
                const audioFlat: any[] = []
                audioData?.forEach((data) => {
                    data.forEach((d: any) => {
                        audioFlat.push(d)
                    })
                })

                //console.log(timer)
                console.log(audioFlat.length / 44100)
                send({ data: audioFlat }, true)
                console.log(audioFlat)
            }}>Stop Recording</button>
            {/* <button onClick={playAudio} disabled={!audioUrl}>
                Play Audio
            </button> */}

            <TabView windowSize={windowSize} />
            <PredictButtonsContainer>
                <PlayButton />
                <ShowButton />
                <LoopButton />
                <PauseButton />
            </PredictButtonsContainer>
        </div>

    )
}

export default MainPage
