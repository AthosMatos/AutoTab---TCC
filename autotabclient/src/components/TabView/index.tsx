import { useEffect, useReducer, useState } from "react";
import Tab, { SpecificNotes } from "./TabCanvas";
import { Container } from "./styled";
import { useTabViewContext } from "../../contexts/TabViewContext/useTabViewContext";

interface TabViewI {
    windowSize: { width: number, height: number }
}

const TabView = (props: TabViewI) => {
    const { windowSize } = props
    const { activatedNotes, frets, updateActivatedNotes, activatedChords, updateActivatedChords, updateSpeed, showNotes } = useTabViewContext()
    const [timedActivatedNotes, setTimedActivatedNotes] = useState<SpecificNotes[]>([])
    const [timedActivatedChords, setTimedActivatedChords] = useState<SpecificNotes[]>([])

    useEffect(() => {
        //console.log(activatedNotes)
        if (activatedNotes.animationStart) {
            updateActivatedNotes({
                animationIndex: 0,
                animationStart: false
            })

            return
        }

        if (activatedNotes.animate && activatedNotes.indexes.length) {
            if (activatedNotes.animationIndex && activatedNotes.animationIndex >= activatedNotes.indexes.length) {
                updateActivatedNotes({

                    animate: activatedNotes.loop ?? false,
                    //show: true,
                })
            }
            else {
                setTimedActivatedNotes([activatedNotes.indexes[activatedNotes.animationIndex ?? 0]])
                setTimeout(() => {
                    updateActivatedNotes({
                        animationIndex: (activatedNotes.animationIndex ?? 0) + 1,
                        //show: true,
                    })
                }, updateSpeed);
            }

        }
        else if (activatedNotes.show) {
            setTimedActivatedNotes(activatedNotes.indexes)
        }
        else {
            setTimedActivatedNotes([])
        }
    }, [activatedNotes])

    useEffect(() => {
        const activatedChord = activatedChords.notesData[activatedChords.animationIndex ?? 0]

        if (activatedChords.animationStart) {
            updateActivatedChords({
                animationIndex: 0,
                animationStart: false
            })

            return
        }
        if (activatedChords.animate && activatedChord.indexes.length) {

            if (activatedChords.animationIndex && activatedChords.animationIndex >= activatedChord.indexes.length) {
                updateActivatedChords({
                    animate: activatedChords.loop ?? false,
                    show: true,
                })
            }
            else {
                setTimedActivatedChords(activatedChord.indexes)
                setTimeout(() => {
                    updateActivatedChords({
                        animationIndex: (activatedChords.animationIndex ?? 0) + 1,
                        show: true,
                    })
                }, updateSpeed);
            }


        }
        else if (activatedChords.show) {
            setTimedActivatedChords(activatedChord.indexes)
        }
        else {
            setTimedActivatedChords([])
        }


    }, [activatedChords])

    return (
        <Container>
            <Tab
                windowSize={{
                    width: windowSize.width
                }}
                findBy="index"
                activatedNotes={showNotes ? timedActivatedNotes : timedActivatedChords} />
        </Container>
    );
};

export default TabView;
