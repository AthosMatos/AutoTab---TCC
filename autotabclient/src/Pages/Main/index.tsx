import { useEffect } from "react"
import { useWebSocket } from "../../contexts/useWebSocket"
import { getPosbySeq, getPosbySeq2 } from "../../utils/path"
import useWindowResize from "../../hooks/useWindowResize"
import { useTabViewContext } from "../../contexts/TabViewContext/useTabViewContext"
import TabView from "../../components/TabView"

const MainPage = () => {
    const windowSize = useWindowResize(0.9)
    const SeqTest = ["C4", "D4", "E4", "F4"];
    const SeqTest2 = [
        ["D3", "A3", "D4", "F#4"],//DMAJOR
        ["A2", "E3", "A3", "C#4", "E4"],//AMAJOR
        ["C4", "D4", "E4", "F4", "A4", "B4"],
        ["G4", "A4", "B4"],
    ]
    const { activatedNotes, allNotesFromFrets, frets, updateActivatedNotes } = useTabViewContext()
    const positions = getPosbySeq2(SeqTest2, allNotesFromFrets, frets)

    /* 
    useEffect(() => {
        updateActivatedNotes({
            ...activatedNotes,
            notes: SeqTest,
            indexes: positions.map((posi) => {
                const pos = posi[1]
                return {
                    string: pos.string,
                    fret: pos.fret
                }
            })
        })
    }, []) */

    useEffect(() => {
        updateActivatedNotes({
            ...activatedNotes,
            notes: SeqTest,
            indexes: positions[1].map((posi) => {
                const pos = posi[1]
                return {
                    string: pos.string,
                    fret: pos.fret
                }
            })
        })
    }, [])

    return (
        <div>
            <button onClick={() => {
                updateActivatedNotes({
                    ...activatedNotes,
                    animate: true,
                    loop: false,
                })
            }}>Run Predict</button>
            <button onClick={() => {
                updateActivatedNotes({
                    ...activatedNotes,
                    animate: false,
                    loop: false,
                    show: true
                })
            }}>Show Predict</button>
            <button onClick={() => {
                updateActivatedNotes({
                    ...activatedNotes,
                    animate: false,
                    loop: false,
                    show: false
                })
            }}>Hide Predict</button>
            <TabView windowSize={windowSize} />
        </div>

    )
}

export default MainPage
