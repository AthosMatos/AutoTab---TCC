import React, { useState, useEffect, useRef } from 'react';



const useAudioRecorder = () => {
    const context = useRef<AudioContext | null>(null);
    const processor = useRef<ScriptProcessorNode | null>(null);
    const input = useRef<MediaStreamAudioSourceNode | null>(null);
    const globalStream = useRef<MediaStream | null>(null);
    const [streamStreaming, setStreamStreaming] = useState(false);
    const [audioData, setAudioData] = useState<any[]>()
    const bufferSize = 4096;
    const [timer, setTimer] = useState(0)
    const [intervalId, setIntervalId] = useState<NodeJS.Timeout | null>(null)

    useEffect(() => {

        if (streamStreaming) {
            const interval = setInterval(() => {
                setTimer((prev) => prev + 0.1)
            }, 100);
            setIntervalId(interval);
        } else {

            intervalId && clearInterval(intervalId);
        }

        //return () => clearInterval(intervalId);
    }, [streamStreaming]);


    const handleSuccess = (stream: MediaStream) => {
        globalStream.current = stream;
        input.current = context.current!.createMediaStreamSource(stream);
        input.current.connect(processor.current!);
        processor.current!.onaudioprocess = (e) => {
            const left = e.inputBuffer.getChannelData(0);
            setAudioData((prev) => {
                if (prev) {
                    return [...prev, left]
                } else {
                    return [left]
                }
            })

            //const left16 = downsampleBuffer(left, 44100, 16000);
        };
    };

    const startRecording = () => {
        setStreamStreaming(true);
        context.current = new (window.AudioContext)({ latencyHint: 'interactive' });
        processor.current = context.current.createScriptProcessor(bufferSize, 1, 1);
        processor.current.connect(context.current.destination);
        context.current.resume();

        //inicialize timer



        navigator.mediaDevices.getUserMedia({ audio: true, video: false }).then(handleSuccess);
    };



    const stopRecording = () => {
        setStreamStreaming(false);
        const track = globalStream.current!.getTracks()[0];
        track.stop();
        input.current!.disconnect(processor.current!);
        processor.current!.disconnect(context.current!.destination);
        context.current!.close().then(() => {
            input.current = null;
            processor.current = null;
            context.current = null;
        });


    };


    return { startRecording, stopRecording, audioData, streamStreaming }


}

export default useAudioRecorder