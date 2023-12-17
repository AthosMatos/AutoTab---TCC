import React, { useState, useEffect, useRef } from 'react';



const useAudioRecorder = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [stream, setStream] = useState<MediaStream | null>(null);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [audioWave, setAudioWave] = useState<Uint8Array>();
    var mediaRecorder: MediaRecorder | undefined = undefined
    const audioRef = useRef<HTMLAudioElement>(null)
    const [analyser, setAnalyser] = useState<AnalyserNode | null>(null)
    const sampleRate = 44100
    const startRecording = async (id: string) => {
        const chunks: Blob[] = [];
        console.log("Recording started");
        try {
            const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            /*  const speakerStream = await (navigator as any).mediaDevices.getDisplayMedia({
                 audio: true,
                 video: false,
             }); */
            setStream(audioStream);
            mediaRecorder = new MediaRecorder(audioStream, {
                audioBitsPerSecond: 128000,
            });
            const audioCtx = new AudioContext({
                sampleRate: sampleRate,
            });
            const analyser = audioCtx.createAnalyser();
            //analyser.fftSize = 512;
            audioCtx.createMediaStreamSource(audioStream).connect(analyser);
            setAnalyser(analyser);
            setIsRecording(true);

            const mimeTypes = ["audio/mp4", "audio/webm", 'audio/wav'].filter((type) =>
                MediaRecorder.isTypeSupported(type)
            );

            if (mimeTypes.length === 0) {
                return alert("Browser not supported");
            }

            mediaRecorder.ondataavailable = async (event) => {
                if (event.data.size > 0) {
                    chunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const blob = new Blob(chunks, { type: 'audio/wav' });
                setAudioBlob(blob);

                const arrayBuffer = await blob.arrayBuffer();
                const audio = new Uint8Array(arrayBuffer);
                setAudioWave(audio);
            };


            /*  setTimerInterval(
                 setInterval(() => {
                     analyser?.getByteTimeDomainData(audioWave)
                     audio.push(new Uint8Array(audioWave))
                     //console.log(AudioWave)
                 }, 10)
             ); */

            /*  recorder = new MediaRecorder(audioStream, {
                 mimeType: mimeTypes[0],
             }); */

            /* recorder.addEventListener("dataavailable", async (event) => {
                //console.log(event.data)
                //convert to int8array
                const blob = await event.data.arrayBuffer()
                const view = new Int8Array(blob)
                // console.log(event.timeStamp)
                //console.log(event.timecode)

                //print currnt time and the time of the event on a formated string showing it in standard time
                //const date = new Date(event.timeStamp)
                //console.log(date.toLocaleTimeString())

                //console.log(new Date().toISOString().slice(14, -5))
                console.log(new Date(event.timeStamp).toISOString().slice(14, -5))
                //print it as a seconds as number
                console.log(event.timeStamp / 1000)



                dataArray.push(...view)
                audio.push(event.data)
            });

            recorder.start(100); */
            mediaRecorder.start();
            setIsRecording(true);
        } catch (error) {
            console.error("Error accessing media devices:", error);
        }
    };

    const stopRecording = () => {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
        }
        setIsRecording(false);
        mediaRecorder?.stop();

        //console.log(AudioWave.length / 44100)

        console.log("Recording stopped");
    };

    const playAudio = () => {
        if (audioBlob) {

            const audioURL = URL.createObjectURL(audioBlob);
            if (!audioRef.current) return
            audioRef.current.src = audioURL;
            console.log('play')
            //audioRef.current.play();
        }
    };

    useEffect(() => {
        startRecording('test')
        setTimeout(() => {
            stopRecording()
        }, 1000)
    }, [])

    useEffect(() => {
        if (audioBlob) playAudio()
    }, [audioBlob])


    return { audioRef, audioWave, analyser, sampleRate };
}

export default useAudioRecorder