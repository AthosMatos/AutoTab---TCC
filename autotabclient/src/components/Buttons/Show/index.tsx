import { usePlaybackContext } from "../../../contexts/PlaybackContext/usePlaybackContext";
import styled from "styled-components";
import Colors from "../../../colors/Colors";
import { IoEyeSharp, IoEyeOffSharp } from "react-icons/io5";

const StyledShowButton = styled.div  <{ isActivated?: boolean }>`
    background-color: #e7715c78;
    border-radius:50%;
    padding: 9px;
    cursor: pointer;
    font-size: 1.4rem;
    color: ${Colors.backColor};
    box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.75);
    transition: scale 0.2s ease, background-color 0.4s ease;
    display: flex;
    justify-content: center;
    align-items: center;
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



interface ShowButtonProps {
    onClick?: () => void;

}

export const ShowButton = ({ onClick }: ShowButtonProps) => {
    const { Predict, toggleShow } = usePlaybackContext()
    return (
        <StyledShowButton onClick={() => {
            toggleShow()
            onClick && onClick()
        }} isActivated={Predict.show} >
            {Predict.show ? <IoEyeSharp style={{ margin: 0 }} /> : <IoEyeOffSharp style={{ margin: 0 }} />}
        </StyledShowButton>

    )
}