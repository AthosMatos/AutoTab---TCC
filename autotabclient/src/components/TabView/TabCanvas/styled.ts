import styled from "styled-components";

export const Node = styled.div<{ color: string; isButton: boolean; frets: number; TabW: number; activated: boolean }>`
	width: ${(props) => (props.TabW / props.frets) * 0.68}px;
	height: ${(props) => (props.TabW / props.frets) * 0.68}px;
	border-radius: 50%;
	background-color: ${(props) => props.color};
	display: flex;
	justify-content: center;
	align-items: center;
	//transition to only transform, filter and box-shadow
	transition: transform 0.2s ease, filter 0.2s ease, box-shadow 0.2s ease;

	user-select: none;
	filter: brightness(0.4);
	cursor: ${(props) => (props.isButton ? "pointer" : "default")};
	box-shadow: ${(props) => (props.isButton ? "5px 0px 10px 0px rgba(0,0,0,0.35)" : "none")};

	${(props) =>
		props.isButton
			? props.activated
				? `
			filter: brightness(1);
			box-shadow: 5px 0px 10px 0px rgba(0,0,0,0.75);
			transform: scale(1.15);
		`
				: `&:hover {
				//brighten the color with effects
				filter: brightness(1);
				box-shadow: 5px 0px 10px 0px rgba(0,0,0,0.75);
				transform: scale(1.15);
			}
		`
			: ``}
	${(props) =>
		props.isButton &&
		`&:active {
				transform: scale(0.9);
			}
		`}


	//animation opacity with delay
	animation-fill-mode: forwards;
	animation-duration: 0.6s;
	animation-timing-function: ease;
	animation-iteration-count: 1;
	animation-direction: normal;
	animation-play-state: running;
	animation-name: opacity;

	@keyframes opacity {
		from {
			opacity: 0;
		}
		to {
			opacity: 1;
		}
	}
`;

export const NodeText = styled.p<{ color: string; TabW: number; increaseText?: number }>`
	font-size: ${(props) => (props.TabW / 100) * (props.increaseText ? props.increaseText : 1)}px;
	font-weight: bold;
	color: ${(props) => props.color};
`;

export const Fret = styled.div<{ frets: number; TabW: number }>`
	display: flex;
	flex-direction: column;
	width: ${(props) => props.TabW / props.frets}px;
	justify-content: space-around;
	align-items: center;
	//background-color: blue;
	border: 1px solid #5b6872;
	transition: border-color 0.2s ease, background-color 0.1s ease;
	&:hover {
		//border-color: #ffcb05;
		background-color: rgba(255, 255, 255, 0.1);
	}
`;

export const TabWrapper = styled.div<{
	width: number;
	height: number;
}>`
	display: flex;
	flex-direction: row;
	border: 2px solid #5b6872;
	width: ${(props) => props.width}px;
	height: ${(props) => (props.width / props.height > 4 ? props.width / 4 : props.height)}px;
	border-radius: ${(props) => (props.width / (props.height ? props.height : props.width / 4)) * 2}px;
	overflow: hidden;
	//background-color: red;

	animation-fill-mode: forwards;
	animation-duration: 0.8s;
	animation-timing-function: ease;
	animation-iteration-count: 1;
	animation-direction: normal;
	animation-play-state: running;
	animation-name: width;

	@keyframes width {
		from {
			width: 0px;
		}
		to {
			width: ${(props) => props.width}px;
		}
	}
`;
