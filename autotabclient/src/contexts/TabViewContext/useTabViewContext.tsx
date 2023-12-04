import { createContext, useContext, useState } from "react";
import { SpecificNotes } from "../../components/TabView/TabCanvas";
import { genNotes } from "../../utils/notes";


type SequenceT = {
    sequence: SpecificNotes[];
    loop: boolean;
    animate: boolean;
};

type ActivateNotesT = {
    notes: string[];
    indexes: SpecificNotes[];
    loop?: boolean;
    animate?: boolean;
    animationIndex?: number;
    show?: boolean;
};

interface TabViewContextI {
    sequence: SequenceT;
    updateSequence: (newSequence: SequenceT) => void;
    activatedNotes: ActivateNotesT;
    updateActivatedNotes: (newActivatedNotes: ActivateNotesT) => void;
    frets: number;
    allNotesFromFrets: string[][];

}

const TabViewContext = createContext<TabViewContextI>({} as any);



export const TabViewProvider = (props: any) => {
    const [sequence, setSequence] = useState<SequenceT>({
        sequence: [],
        loop: false,
        animate: false,
    });
    const [frets, setFrets] = useState<number>(24);
    const [activatedNotes, setActivatedNotes] = useState<ActivateNotesT>({
        notes: [],
        indexes: [],
        loop: false,
        animate: false,
        show: false,
    })
    const allNotesFromFrets = genNotes({
        FRETS: frets,
    })[1];


    return (
        <TabViewContext.Provider value={{
            sequence,
            updateSequence: setSequence,
            activatedNotes,
            updateActivatedNotes: setActivatedNotes,
            frets,
            allNotesFromFrets
        }}>
            {props.children}
        </TabViewContext.Provider>
    );
};

export const useTabViewContext = () => {

    const context = useContext(TabViewContext);
    if (!context) {
        throw new Error(
            "useSpecificContext must be used within a SpecificProvider"
        );
    }
    return context;
};
