import React, { useState, useEffect, useRef } from 'react';



const useAudioRecorder = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [stream, setStream] = useState<MediaStream | null>(null);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [wavArrayBuffer, setWavArrayBuffer] = useState<ArrayBuffer | null>(null);
    //var mediaRecorder: MediaRecorder | undefined = undefined
    const audioRef = useRef<HTMLAudioElement>(null)
    const sampleRate = 44100
    const [input, setInput] = useState<MediaStreamAudioSourceNode | null>(null);
    const [processor, setProcessor] = useState<ScriptProcessorNode | null>(null);
    const [context, setContext] = useState<AudioContext | null>(null);
    const chunks: any[] = [];

    /* function downsampleBuffer (buffer, sampleRate, outSampleRate) {
        if (outSampleRate == sampleRate) {
        return buffer;
        }
        if (outSampleRate > sampleRate) {
        throw 'downsampling rate show be smaller than original sample rate';
        }
        var sampleRateRatio = sampleRate / outSampleRate;
        var newLength = Math.round(buffer.length / sampleRateRatio);
        var result = new Int16Array(newLength);
        var offsetResult = 0;
        var offsetBuffer = 0;
        while (offsetResult < result.length) {
        var nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        var accum = 0,
            count = 0;
        for (var i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }
    
        result[offsetResult] = Math.min(1, accum / count) * 0x7fff;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
        }
        return result.buffer;
    }  */


    function audioBufferToWavBlob(audioBuffer: AudioBuffer): Blob {
        const interleaved = interleaveChannels(audioBuffer);
        const buffer = createWavBuffer(interleaved, audioBuffer.sampleRate);
        return new Blob([buffer], { type: 'audio/wav' });
    }

    function interleaveChannels(audioBuffer: AudioBuffer): Float32Array {
        const interleaved = new Float32Array(audioBuffer.length * audioBuffer.numberOfChannels);
        const channelData = [];

        for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
            channelData.push(audioBuffer.getChannelData(channel));
        }

        for (let i = 0; i < audioBuffer.length; i++) {
            for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
                interleaved[i * audioBuffer.numberOfChannels + channel] = channelData[channel][i];
            }
        }

        return interleaved;
    }

    function createWavBuffer(interleaved: Float32Array, sampleRate: number): ArrayBuffer {
        const numberOfChannels = 1;
        const bytesPerSample = 4; // 32-bit floating-point

        const dataSize = interleaved.length * bytesPerSample;
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);

        // RIFF identifier
        writeString(view, 0, 'RIFF');
        // file length
        view.setUint32(4, 36 + dataSize, true);
        // RIFF type
        writeString(view, 8, 'WAVE');
        // format chunk identifier
        writeString(view, 12, 'fmt ');
        // format chunk length
        view.setUint32(16, 16, true);
        // sample format (floating point)
        view.setUint16(20, 3, true);
        // channel count
        view.setUint16(22, numberOfChannels, true);
        // sample rate
        view.setUint32(24, sampleRate, true);
        // byte rate (sample rate * block align)
        view.setUint32(28, sampleRate * numberOfChannels * bytesPerSample, true);
        // block align (channel count * bytes per sample)
        view.setUint16(32, numberOfChannels * bytesPerSample, true);
        // bits per sample
        view.setUint16(34, bytesPerSample * 8, true);
        // data chunk identifier
        writeString(view, 36, 'data');
        // data chunk length
        view.setUint32(40, dataSize, true);

        // write the PCM samples
        for (let offset = 0; offset < interleaved.length; offset += 1) {
            view.setFloat32(44 + offset * bytesPerSample, interleaved[offset], true);
        }

        return buffer;
    }

    function writeString(view: DataView, offset: number, value: string) {
        for (let i = 0; i < value.length; i++) {
            view.setUint8(offset + i, value.charCodeAt(i));
        }
    }






    const startRecording = async (id: string) => {

        console.log("Recording started");
        try {

            const context = new AudioContext({
                // if Non-interactive, use 'playback' or 'balanced' // https://developer.mozilla.org/en-US/docs/Web/API/AudioContextLatencyCategory
                latencyHint: 'interactive',
            });
            const processor = context.createScriptProcessor(4096, 1, 1);
            processor.connect(context.destination);
            setProcessor(processor);
            context.resume();
            setContext(context);

            var handleSuccess = function (stream: MediaStream) {
                setStream(stream);
                const input = context.createMediaStreamSource(stream);
                setInput(input);
                input.connect(processor);

                processor.onaudioprocess = function (e) {
                    var left = e.inputBuffer.getChannelData(0);
                    //var left16 = downsampleBuffer(left, 44100, 16000);
                    chunks.push(left);

                };
                //on stop
            };

            navigator.mediaDevices.getUserMedia({ audio: true, video: false }).then(handleSuccess);



            /* mediaRecorder = new MediaRecorder(audioStream, {
                audioBitsPerSecond: 1000000,
                //bitsPerSecond: 44100,
                mimeType: mimeTypes[0]

            }); */
            /* const audioCtx = new AudioContext({
                sampleRate: sampleRate,
            });
            const analyser = audioCtx.createAnalyser();
            //analyser.fftSize = 512;
            audioCtx.createMediaStreamSource(audioStream).connect(analyser);
            setAnalyser(analyser); */
            setIsRecording(true);



            /* mediaRecorder.ondataavailable = async (event) => {
                if (event.data.size > 0) {
                    chunks.push(event.data);
                }
            }; */

            /* mediaRecorder.onstop = async () => {
                const blob = new Blob(chunks, { type: 'audio/wav' });
                const arrayBuffer = await blob.arrayBuffer();
                const audioBuffer = await (new AudioContext()).decodeAudioData(arrayBuffer);
                const audioBlob = audioBufferToWavBlob(audioBuffer);

                //convert blob to arraybuffer
                const wavArrayBuffer = await audioBlob.arrayBuffer()

                setWavArrayBuffer(wavArrayBuffer)

                setAudioBlob(audioBlob);

                //const arrayBuffer = await blob.arrayBuffer();
                //const audio = new Uint8Array(arrayBuffer);
                //setAudioWave(audio);
            }; */


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
            //mediaRecorder.start();
            setIsRecording(true);
        } catch (error) {
            console.error("Error accessing media devices:", error);
        }
    };

    const stopRecording = () => {


        if (stream && input && processor && context) {
            stream.getTracks().forEach((track) => track.stop());
            input.disconnect(processor);
            processor.disconnect(context.destination);
            context.close().then(function () {
                console.log(chunks)
            });
            setIsRecording(false);
            //mediaRecorder?.stop();

            //console.log(AudioWave.length / 44100)

            console.log("Recording stopped");
        }


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

    function startRecord() {
        startRecording('test')

    }
    useEffect(() => {
        /* if (stream && input && processor && context) {
            setTimeout(() => {

            }, 1000)
        } */

    }, [stream, input, processor, context])


    return { audioRef, sampleRate, startRecord, audioBlob, wavArrayBuffer, stopRecording };
}

export default useAudioRecorder