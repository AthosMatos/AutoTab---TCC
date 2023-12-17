import { createContext, useContext, useState } from "react";
import { SpecificNotes } from "../../components/TabView/TabCanvas";
import { genNotes } from "../../utils/notes";


type ActiveAudioEventsT = {
    notes: (string | string[])[];
    indexes: SpecificNotes[];
};

interface TabViewContextI {
    frets: number;
    allNotesFromFrets: string[][];
    activeAudioEvents: ActiveAudioEventsT[];
    updateActivatedChords: (newActivatedChords: ActiveAudioEventsT[]) => void;
}

const TabViewContext = createContext<TabViewContextI>({} as any);


export const TabViewProvider = (props: any) => {
    const [frets, setFrets] = useState<number>(24);
    const [activeAudioEvents, setActiveAudioEvents] = useState<ActiveAudioEventsT[]>([])
    const allNotesFromFrets = genNotes({
        FRETS: frets,
    })[1];

    function updateActivatedChords(newActivatedChords: ActiveAudioEventsT[]) {
        setActiveAudioEvents(newActivatedChords)
    }

    return (
        <TabViewContext.Provider value={{
            updateActivatedChords,
            activeAudioEvents,
            frets,
            allNotesFromFrets,
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
