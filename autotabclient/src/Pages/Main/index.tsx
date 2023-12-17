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

    const startRecording = async () => {
        const micStream = await navigator.mediaDevices.getUserMedia({ audio: true, });
        const audioContext = new (window.AudioContext)({
            ///sampleRate: 48000,
            latencyHint: 'interactive'
        });
        const gainNode = audioContext.createGain();
        gainNode.gain.value = 2;

        //compressor.connect(audioContext.destination);
        let source = audioContext.createMediaStreamSource(micStream);
        source.channelInterpretation = 'discrete';
        source.channelCountMode = 'explicit';
        source.channelCount = 2;
        source.connect(gainNode);

        // Create a Recorder object
        let recorder = new Recorder(source);

        recorder.record();

        // Stop recording after 1 second
        setTimeout(function () {
            recorder.stop();

            // Export the recording to a WAV file
            recorder.exportWAV(async function (blob: Blob) {


                // Convert the Blob to ArrayBuffer
                const arrayBuffer = await blob.arrayBuffer();

                // Convert the ArrayBuffer to base64
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

                const base64String = arrayBufferToBase64(arrayBuffer);




                // Send the data through the WebSocket
                // Replace 'your-websocket-server-url' with your actual WebSocket server URL

                send({ data: blob })

                let url = URL.createObjectURL(blob);
                let a = document.createElement('a');
                a.href = url;
                a.download = 'test.wav';
                a.click();
            });
        }, 3500);

    };



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
            {/*  <button onClick={stopRecording}>Stop Recording</button>
            <button onClick={playAudio} disabled={!audioUrl}>
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
