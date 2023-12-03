import { useEffect, useReducer, useState } from "react";
import Tab, { SpecificNotes } from "./TabCanvas";
import { Container } from "./styled";
import { useTabViewContext } from "../../contexts/TabViewContext/useTabViewContext";

interface TabViewI {
    windowSize: { width: number, height: number }
}

const TabView = (props: TabViewI) => {
    const { windowSize } = props
    const { activatedNotes, frets, updateActivatedNotes } = useTabViewContext()
    const [timedActivatedNotes, setTimedActivatedNotes] = useState<SpecificNotes[]>([])

    useEffect(() => {
        if (activatedNotes.animate && activatedNotes.indexes.length) {
            if (activatedNotes.animationIndex && activatedNotes.animationIndex >= activatedNotes.indexes.length) {
                updateActivatedNotes({
                    ...activatedNotes,
                    animate: activatedNotes.loop ?? false,
                    animationIndex: 0
                })
            }
            else {
                setTimedActivatedNotes([activatedNotes.indexes[activatedNotes.animationIndex ?? 0]])
                setTimeout(() => {
                    updateActivatedNotes({
                        ...activatedNotes,
                        animationIndex: (activatedNotes.animationIndex ?? 0) + 1
                    })
                }, 200);
            }

        }
    }, [activatedNotes])


    return (
        <Container>
            <Tab
                windowSize={{
                    width: windowSize.width
                }}
                findBy="index"
                activatedNotes={timedActivatedNotes} />
        </Container>
    );
};

export default TabView;
