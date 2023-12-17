import { FaPause } from "react-icons/fa6";
import { usePlaybackContext } from "../../../contexts/PlaybackContext/usePlaybackContext";
import styled from "styled-components";
import Colors from "../../../colors/Colors";

const StyledPauseButton = styled(FaPause) <{ isActivated?: boolean, isClicable?: boolean }>`
    background-color: #e7715c78;
    border-radius:50%;
    padding: 12px;
    //margin: 10px;
    cursor: pointer;
    font-size: 1rem;
    color: ${Colors.backColor};
    box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.75);
    transition: scale 0.2s ease, background-color 0.4s ease;

    &:hover {
        scale: 1.1;
    }

    //trigger is activated
    ${({ isActivated }) => isActivated && `
        background-color: #e7af5c;
        scale: 1.1;
    `}
    ${({ isClicable }) => isClicable && `
        &:active {
        //background-color: #5ce796;
        scale: 0.9;
    }
    `}
    
`;

interface PauseButtonProps {
    onClick?: () => void;
}

export const PauseButton = ({ onClick }: PauseButtonProps) => {
    const { Predict, stopPredict } = usePlaybackContext()
    return (
        <StyledPauseButton onClick={() => {
            Predict.playing && stopPredict()
            onClick && onClick()
        }} isActivated={!Predict.playing} />
    )
}