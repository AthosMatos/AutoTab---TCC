import smoothColors from "../../../colors/smoothColors";
import { Fret, Node, NodeText, TabWrapper } from "./styled";
import { useTabViewContext } from "../../../contexts/TabViewContext/useTabViewContext";

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

    /* 
    const nodeSizeValue = signal(interpolate(24, 111.5, 44.5, 162.33, windowSize.value.width / frets.value))
    const fontSizeValue = signal(interpolate(36, 8, 159.33, 41.5, windowSize.value.width / frets.value))
    const gap = signal(interpolate(8, 0.44, 159, 41.5, windowSize.value.width / frets.value))
    const paddingHeight = signal(interpolate(1.2, 0.43, 159, 41.5, windowSize.value.width / frets.value)) 
    */

    /* const nodeSizeValue = windowSize.width / 36
    const fontSizeValue = windowSize.width / 124
    const fretNumberSize = windowSize.width / 70
    const gap = windowSize.width / 170
    const paddingHeight = windowSize.width / 130
    const tabBorderRadius = windowSize.width / 130 */

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
    //convert allNotesFromFrets from [string][fret] to [fret][string]
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
