export type PosI = {
	fret: number;
	string: number;
	note?: string;
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
	const Chords: Node[][][] = [];
	Seq.forEach((simultaneousNotes, simulIndex) => {
		const notesPos: Node[][] = [];
		simultaneousNotes.forEach((notefromSimultaneous, noteIndex) => {
			const notePos: Node[] = [];
			allNotesFromFrets.forEach((notesFromString, stringIndex) => {
				const fretIndex = notesFromString.findIndex((noteFromString) => noteFromString === notefromSimultaneous);
				if (fretIndex !== -1) notePos.push({ fret: fretIndex, string: stringIndex, note: notefromSimultaneous });
			});
			notesPos.push(notePos);
		});
		Chords.push(notesPos);
	});

	const ChordsPositions: [string, PosI][][] = [];

	Chords.forEach((simultaneousNotes, simulIndex) => {
		const Positions: [string, PosI][] = [];
		let noteInAnalysis: PosI;
		const stringsUsed: number[] = [];
		for (let i = 0; i < simultaneousNotes.length - 1; i++) {
			if (simultaneousNotes.length < 2) break;
			const Notes = simultaneousNotes[i];
			let nextNoteName = simultaneousNotes[i + 1][0].note;
			if (i === 0) {
				Positions.push([Notes[0].note, Notes[0]]);
				noteInAnalysis = { string: Notes[0].string, fret: Notes[0].fret };
				stringsUsed.push(noteInAnalysis.string);
			}

			const { eachStringWeight, biggetWeight, lowestWeight, nextPosNote } = vasculhar2(noteInAnalysis!, nextNoteName, allNotesFromFrets, frets, stringsUsed);
			stringsUsed.push(nextPosNote.string);
			noteInAnalysis = nextPosNote;
			Positions.push([nextNoteName, nextPosNote]);
		}
		ChordsPositions.push(Positions);
	});

	console.log(ChordsPositions);
	return ChordsPositions;
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

function vasculhar2(pos: PosI, note: string, allNotesFromFrets: string[][], frets: number, stringsUsed: number[]) {
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
			if (allNotesFromFrets[str][frt] === note && !stringsUsed.includes(str)) {
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
