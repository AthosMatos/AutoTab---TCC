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
    animationStart?: boolean;
};

type ActivateNotesUpdateT = {
    notes?: string[];
    indexes?: SpecificNotes[];
    loop?: boolean;
    animate?: boolean;
    animationIndex?: number;
    show?: boolean;
    animationStart?: boolean;
};

type ActivateChordsT = {
    notesData: {
        notes: string[];
        indexes: SpecificNotes[];
    }[];
    loop?: boolean;
    animate?: boolean;
    animationIndex?: number;
    show?: boolean;
    animationStart?: boolean;
};
type ActivateChordsUpdateT = {
    notesData?: {
        notes: string[];
        indexes: SpecificNotes[];
    }[];
    loop?: boolean;
    animate?: boolean;
    animationIndex?: number;
    show?: boolean;
    animationStart?: boolean;
};
interface TabViewContextI {
    sequence: SequenceT;
    updateSequence: (newSequence: SequenceT) => void;
    activatedNotes: ActivateNotesT;
    updateActivatedNotes: (newActivatedNotes: ActivateNotesUpdateT) => void;
    frets: number;
    allNotesFromFrets: string[][];
    activatedChords: ActivateChordsT;
    updateActivatedChords: (newActivatedChords: ActivateChordsUpdateT) => void;
    updateSpeed: number;
    updateUpdateSpeed: (newUpdateSpeed: number) => void;
    showNotes: boolean;
    setShowNotes: (newShowNotes: boolean) => void;
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
    const [activatedChords, setActivatedChords] = useState<ActivateChordsT>({
        notesData: [],
        loop: false,
        animate: false,
        show: false,
    })
    const allNotesFromFrets = genNotes({
        FRETS: frets,
    })[1];

    const [updateSpeed, setUpdateSpeed] = useState<number>(200);
    const [showNotes, setShowNotes] = useState<boolean>(false);


    function updateActivatedChords(params: ActivateChordsUpdateT) {
        setActivatedChords(
            {
                ...activatedChords,
                ...params
            }
        )
    }
    function updateActivatedNotes(params: ActivateNotesUpdateT) {
        setActivatedNotes(
            {
                ...activatedNotes,
                ...params
            }
        )
    }

    return (
        <TabViewContext.Provider value={{
            sequence,
            updateSequence: setSequence,
            activatedNotes,
            activatedChords,
            updateActivatedChords,
            updateActivatedNotes,
            frets,
            allNotesFromFrets,
            updateSpeed,
            updateUpdateSpeed: setUpdateSpeed,
            showNotes,
            setShowNotes
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
