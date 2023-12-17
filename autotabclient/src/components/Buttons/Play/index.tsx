import styled from "styled-components";
import Colors from "../../../colors/Colors";
import { usePlaybackContext } from "../../../contexts/PlaybackContext/usePlaybackContext";
import { FaPlay } from "react-icons/fa6";

const StyledPlayButton = styled(FaPlay) <{ isActivated?: boolean }>`
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
        background-color: #5ce796;
        scale: 1.1;
    `}

    &:active {
        //background-color: #5ce796;
        scale: 0.9;
    }
`;

interface PlayButtonProps {
    onClick?: () => void;
}

export const PlayButton = ({ onClick }: PlayButtonProps) => {
    const { Predict, togglePredict } = usePlaybackContext()
    return (
        <StyledPlayButton onClick={() => {
            togglePredict()
            onClick && onClick()
        }} isActivated={Predict.playing} />
    )
}