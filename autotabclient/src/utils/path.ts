export type PosI = {
	fret: number;
	string: number;
};

export type Node = {
	fret: number; //x
	string: number; //y
	note: string; //name
	nodeIndex?: number; //index
};

/* const Seq = [
    ["C4", "D4", "E4", "F4"],
    ["G4", "A4", "B4"],
    ["C4", "D4", "E4", "F4","A4", "B4"],
    ["G4", "A4", "B4"],
] */

export function getPosbySeq2(Seq: string[][], allNotesFromFrets: string[][], frets: number) {
	function fillEmptyStrings() {
		const strings: number[] = [];
		for (let i = 0; i < 6; i++) {
			strings.push(i);
		}

		return strings;
	}

	const SeqAsPos: { pos: PosI[]; note: string }[][] = [];
	Seq.forEach((simultaneousNotes, simulIndex) => {
		const notesPos: { pos: PosI[]; note: string }[] = [];
		simultaneousNotes.forEach((notefromSimultaneous, noteIndex) => {
			const notePos: PosI[] = [];
			allNotesFromFrets.forEach((notesFromString, stringIndex) => {
				const fretIndex = notesFromString.findIndex((noteFromString) => noteFromString === notefromSimultaneous);
				if (fretIndex !== -1) notePos.push({ fret: fretIndex, string: stringIndex });
			});
			notesPos.push({ pos: notePos, note: notefromSimultaneous });
		});
		SeqAsPos.push(notesPos);
	});

	console.log("SeqAsPos", SeqAsPos);

	SeqAsPos.forEach((simultaneousNotes, simulIndex) => {
		let firstRun = true;
		let currPoss: PosI = { fret: 0, string: 0 };
		const NotUsedStrings = fillEmptyStrings();

		for (let currNoteIndex = 0, nextNoteIndex = 1; currNoteIndex < simultaneousNotes.length - 1; currNoteIndex++, nextNoteIndex++) {
			const currNotefromSimultaneous = simultaneousNotes[currNoteIndex];
			//console.log("currNotefromSimultaneous", currNotefromSimultaneous.note);
			const nextNoteFromSimultaneous = simultaneousNotes[nextNoteIndex];
			const { pos: currPositions } = currNotefromSimultaneous;
			const { pos: nextPositions } = nextNoteFromSimultaneous;

			let bestWeight = -1;
			let bestNextPos: PosI = { fret: 0, string: 0 };
			let firstPostoUse: PosI = { fret: 0, string: 0 };

			if (firstRun) {
				currPositions.forEach((currPos, index) => {
					nextPositions.forEach((nextPos, nextIndex) => {
						const { lowestWeight, nextPosNote } = vasculhar2(currPos, nextNoteFromSimultaneous.note, allNotesFromFrets, frets, NotUsedStrings);
						if (bestWeight === -1) {
							bestWeight = lowestWeight;
							bestNextPos = nextPosNote;
							firstPostoUse = currPos;
							const compensatedIndex = NotUsedStrings.findIndex((string) => string === currPos.string);
							//console.log("NotUsedStrings", NotUsedStrings, "usedstring", currPos.string, "compensatedIndex", compensatedIndex);
							if (compensatedIndex !== -1) NotUsedStrings.splice(compensatedIndex, 1);
						} else if (lowestWeight < bestWeight) {
							bestWeight = lowestWeight;
							bestNextPos = nextPosNote;
							firstPostoUse = currPos;
							const compensatedIndex = NotUsedStrings.findIndex((string) => string === currPos.string);
							if (compensatedIndex !== -1) NotUsedStrings.splice(compensatedIndex, 1);
						}
					});
				});
			} else {
				const currPos = currPoss;
				nextPositions.forEach((nextPos, nextIndex) => {
					const { lowestWeight, nextPosNote } = vasculhar2(currPos, nextNoteFromSimultaneous.note, allNotesFromFrets, frets, NotUsedStrings);
					if (bestWeight === -1) {
						bestWeight = lowestWeight;
						bestNextPos = nextPosNote;
						firstPostoUse = currPos;
						const compensatedIndex = NotUsedStrings.findIndex((string) => string === currPos.string);
						//console.log("NotUsedStrings", NotUsedStrings, "usedstring", currPos.string, "compensatedIndex", compensatedIndex);
						if (compensatedIndex !== -1) NotUsedStrings.splice(compensatedIndex, 1);
					} else if (lowestWeight < bestWeight) {
						bestWeight = lowestWeight;
						bestNextPos = nextPosNote;
						firstPostoUse = currPos;
						const compensatedIndex = NotUsedStrings.findIndex((string) => string === currPos.string);
						if (compensatedIndex !== -1) NotUsedStrings.splice(compensatedIndex, 1);
					}
				});
			}

			console.log({
				curr: { pos: firstPostoUse, note: currNotefromSimultaneous.note },
				next: { pos: bestNextPos, note: nextNoteFromSimultaneous.note },
				bestWeight: bestWeight,
			});
			firstRun = false;
			currPoss = bestNextPos;
		}
	});
}

export function getPosbySeq(Seq: string[], allNotesFromFrets: string[][], frets: number) {
	const NotesGrouped: Node[][] = [];

	Seq.forEach((noteFromSeq, index) => {
		const node: Node[] = [];
		allNotesFromFrets.forEach((notesFromString, stringIndex) => {
			notesFromString.forEach((noteFromString, fretIndex) => {
				if (noteFromString === noteFromSeq) {
					node.push({ fret: fretIndex, string: stringIndex, note: noteFromString });
				}
			});
		});
		NotesGrouped.push(node);
	});

	const Positions: [string, PosI][] = [];
	let noteInAnalysis: PosI;

	for (let i = 0; i < NotesGrouped.length - 1; i++) {
		if (NotesGrouped.length < 2) break;
		const Notes = NotesGrouped[i];
		let nextNoteName = NotesGrouped[i + 1][0].note;
		if (i === 0) {
			Positions.push([Notes[0].note, Notes[0]]);
			noteInAnalysis = { string: Notes[0].string, fret: Notes[0].fret };
		}

		const { eachStringWeight, biggetWeight, lowestWeight, nextPosNote } = vasculhar(noteInAnalysis!, nextNoteName, allNotesFromFrets, frets);
		noteInAnalysis = nextPosNote;
		Positions.push([nextNoteName, nextPosNote]);
	}

	return Positions;
}

function vasculhar2(currPos: PosI, nextNote: string, allNotesFromFrets: string[][], frets: number, NotUsedStrings: number[]) {
	let lowestWeight = 0;
	let nextNoteSmallestWeight = -1;
	let nextPosNote: PosI = { string: 0, fret: 0 };

	for (let str = 0; str < 6; str++) {
		for (let frt = 0; frt < frets; frt++) {
			let dist = Math.pow(Math.abs(currPos.string - str), 2) + Math.abs(currPos.fret - frt);
			if (frt === 0) dist = 3;

			if (allNotesFromFrets[str][frt] === nextNote && str !== currPos.string) {
				if (nextNoteSmallestWeight === -1) {
					nextNoteSmallestWeight = dist;
					nextPosNote = { string: str, fret: frt };
					lowestWeight = dist;
				} else if (dist < nextNoteSmallestWeight) {
					nextNoteSmallestWeight = dist;
					nextPosNote = { string: str, fret: frt };
					lowestWeight = dist;
				}
			}
		}
	}

	return { lowestWeight, nextPosNote };
}

function vasculhar(pos: PosI, note: string, allNotesFromFrets: string[][], frets: number) {
	const eachStringWeight: number[][] = [];
	let biggetWeight = 0;
	let lowestWeight = 0;
	let nextPosNote: PosI = { string: 0, fret: 0 };
	let nextPosNotes: PosI[] = [];
	let nextNoteSmallestWeight = -1;

	for (let str = 0; str < 6; str++) {
		const fretWeight: number[] = [];
		for (let frt = 0; frt < frets; frt++) {
			let dist = Math.pow(Math.abs(pos.string - str), 2) + Math.abs(pos.fret - frt);
			if (frt === 0) dist = 3;

			fretWeight.push(dist);
			if (dist > biggetWeight) {
				biggetWeight = dist;
			}
			if (allNotesFromFrets[str][frt] === note) {
				if (nextNoteSmallestWeight === -1) {
					nextNoteSmallestWeight = dist;
					nextPosNote = { string: str, fret: frt };
					lowestWeight = dist;
				} else if (dist < nextNoteSmallestWeight) {
					nextNoteSmallestWeight = dist;
					nextPosNote = { string: str, fret: frt };
					lowestWeight = dist;
				}
				nextPosNotes.push({ string: str, fret: frt });
			}
		}
		eachStringWeight.push(fretWeight);
	}

	//the complexity of this function is O(n^2) because of the nested for loops
	return { eachStringWeight, biggetWeight, lowestWeight, nextPosNote, nextPosNotes };
}
