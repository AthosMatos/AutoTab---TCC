import { useEffect, useState } from "react"
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
        ["F2", "C3", "F3", "A3", "C4", "F4"], //FMAJOR
        ["G4", "A4", "B4"],
    ]
    /* 
    prepare for the next step
    const SeqTest2 = [
        "C4", "D4", "E4", "F4"
        ["D3", "A3", "D4", "F#4"],//DMAJOR
        ["A2", "E3", "A3", "C#4", "E4"],//AMAJOR
        ["F2", "C3", "F3", "A3", "C4", "F4"], //FMAJOR
        ["G4", "A4", "B4"],
    ]
    */
    const { activatedChords, allNotesFromFrets, frets, updateActivatedChords, updateUpdateSpeed, updateActivatedNotes, setShowNotes, showNotes } = useTabViewContext()
    const chords = getPosbySeq2(SeqTest2, allNotesFromFrets, frets)
    const notes = getPosbySeq(SeqTest, allNotesFromFrets, frets)

    useEffect(() => {
        updateActivatedNotes({
            notes: SeqTest,
            indexes: notes.map((posi) => {
                const pos = posi[1]
                return {
                    string: pos.string,
                    fret: pos.fret
                }
            })
        })


    }, [])

    useEffect(() => {
        updateUpdateSpeed(500)
        updateActivatedChords({
            notesData: chords.map((posi) => {
                return {
                    notes: SeqTest,
                    indexes: posi.map((posi) => {
                        const pos = posi[1]
                        return {
                            string: pos.string,
                            fret: pos.fret
                        }
                    })
                }
            })

        })
    }, [])

    return (
        <div>
            <button onClick={() => {
                showNotes ?
                    updateActivatedNotes({
                        animate: true,
                        loop: false,
                        animationStart: true,
                    })
                    :
                    updateActivatedChords({
                        animate: true,
                        loop: false,
                        animationStart: true,
                    })
                /* 
                    updateActivatedChords({
                    animate: true,
                    loop: false,
                    animationStart: true,
                })
                */
            }}>Run Predict</button>
            <button onClick={() => {
                showNotes ?
                    updateActivatedNotes({
                        animate: false,
                        loop: false,
                        show: true
                    })
                    :
                    updateActivatedChords({
                        animate: false,
                        loop: false,
                        show: true
                    })
            }}>Show Predict</button>
            <button onClick={() => {
                showNotes ?
                    updateActivatedNotes({
                        animate: false,
                        loop: false,
                        show: false
                    })
                    :
                    updateActivatedChords({
                        animate: false,
                        loop: false,
                        show: false
                    })
            }}>Hide Predict</button>
            <button onClick={() => {
                setShowNotes(!showNotes)
            }}>Toggle between Notes/Chords</button>
            <TabView windowSize={windowSize} />
        </div>

    )
}

export default MainPage
