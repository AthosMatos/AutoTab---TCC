import smoothColors from "../../../colors/smoothColors";
import { Fret, Node, NodeText, TabWrapper } from "./styled";
import { useTabViewContext } from "../../../contexts/TabViewContext/useTabViewContext";
import { useRef } from "react";

export interface SpecificNotes {
    string: number,
    fret: number
}

type TabCanvasIndexI = {
    findBy: 'index'
    activatedNotes?: SpecificNotes[]
}
type TabCanvasNameI = {
    findBy: 'name'
    activatedNotes?: string[]
}
interface TabI {
    windowSize: { width?: number, height?: number }
}
type TabCanvasI = (TabCanvasIndexI | TabCanvasNameI) & TabI

const Tab = (props: TabCanvasI) => {

    const { windowSize } = props
    const { allNotesFromFrets, frets } = useTabViewContext()
    const NodeRef = useRef<HTMLDivElement>(null)

    function defineColor(note: string) {
        let nodeColor = '#fff'
        let textColor = '#fff'
        note = note.toString().charAt(0)

        switch (note) {
            case "E":
                nodeColor = smoothColors.blue
                break;
            case "A":
                nodeColor = smoothColors.orange
                break;
            case "D":
                nodeColor = smoothColors.purple
                break;
            case "G":
                nodeColor = smoothColors.yellow
                textColor = '#202020';
                break;
            case "B":
                nodeColor = smoothColors.gray
                break;
            case "C":
                nodeColor = smoothColors.green
                break;
            default:
                nodeColor = smoothColors.red
                break;
        }

        return { nodeColor, textColor }
    }

    const allNotesFromFretsConverted = allNotesFromFrets[0].map((col, i) => allNotesFromFrets.map(row => row[i]))

    const TabW = windowSize.width ? windowSize.width : 400
    const TabH = windowSize.height ? windowSize.height : 0

    return (
        <TabWrapper
            height={TabH}
            width={TabW}
        >
            {allNotesFromFretsConverted.map((stringsOnFret, fretIndex) => {

                return (
                    <Fret
                        frets={frets}
                        TabW={TabW}
                        key={fretIndex}
                    >
                        <Node
                            activated={false}
                            TabW={TabW}
                            frets={frets}
                            isButton={false}

                            color={'transparent'}
                            key={fretIndex}>
                            <NodeText
                                increaseText={1.4}
                                TabW={TabW}
                                color={'white'}>
                                {fretIndex}
                            </NodeText>
                        </Node>

                        {
                            stringsOnFret.map((note, stringIndex) => {
                                const { nodeColor, textColor } = defineColor(note)
                                //activate node if it is in the activatedNodes array
                                let activated = false
                                switch (props.findBy) {
                                    case 'index':
                                        const foundNote = props.activatedNotes && props.activatedNotes.find((note: SpecificNotes) => note.string === stringIndex && note.fret === fretIndex)
                                        activated = foundNote ? true : false
                                        break
                                    case "name":
                                        activated = props.activatedNotes ? props.activatedNotes.includes(note) : false
                                        break
                                }


                                return (
                                    <Node
                                        color={nodeColor}
                                        activated={activated}
                                        TabW={TabW}
                                        frets={frets}
                                        isButton
                                        key={stringIndex}
                                    >
                                        <NodeText
                                            TabW={TabW}
                                            color={textColor}>
                                            {note}
                                        </NodeText>
                                    </Node>
                                )
                            })
                        }
                    </Fret>
                )
            })}
        </TabWrapper >
    );
};

export default Tab;
