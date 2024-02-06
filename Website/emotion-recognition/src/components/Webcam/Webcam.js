import React, { useState } from 'react';
import * as faceapi from 'face-api.js';
import PhotoAlbumComponent from '../PhotoAlbum/PhotoAlbum';

function WebcamComponent() {

    const [modelsLoaded, setModelsLoaded] = React.useState(false);
    const [captureVideo, setCaptureVideo] = React.useState(false);

    const [savedImages, setSavedImages] = React.useState([]);
    const [detectedFaceBox, setDetectedFaceBox] = useState(null);

    const videoRef = React.useRef();
    const videoHeight = 480;
    const videoWidth = 640;
    const canvasRef = React.useRef();

    React.useEffect(() => {
        const loadModels = async () => {
            const MODEL_URL = process.env.PUBLIC_URL + '/models';

            Promise.all([
                faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
                faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
                faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
                faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
            ]).then(setModelsLoaded(true));
        }
        loadModels();
    }, []);

    const startVideo = () => {
        setCaptureVideo(true);
        navigator.mediaDevices
            .getUserMedia({ video: { width: 300 } })
            .then(stream => {
                let video = videoRef.current;
                video.srcObject = stream;
                video.play();
            })
            .catch(err => {
                console.error("error:", err);
            });
    }

    const handleVideoOnPlay = () => {
        setInterval(async () => {
            if (canvasRef && canvasRef.current) {
                canvasRef.current.innerHTML = faceapi.createCanvasFromMedia(videoRef.current);
                const displaySize = {
                    width: videoWidth,
                    height: videoHeight
                }

                faceapi.matchDimensions(canvasRef.current, displaySize);

                const detections = await faceapi.detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions();

                const resizedDetections = faceapi.resizeResults(detections, displaySize);

                canvasRef && canvasRef.current && canvasRef.current.getContext('2d').clearRect(0, 0, videoWidth, videoHeight);
                canvasRef && canvasRef.current && faceapi.draw.drawDetections(canvasRef.current, resizedDetections);
                canvasRef && canvasRef.current && faceapi.draw.drawFaceLandmarks(canvasRef.current, resizedDetections);
                //canvasRef && canvasRef.current && faceapi.draw.drawFaceExpressions(canvasRef.current, resizedDetections);

                // Send to microservice
                if (resizedDetections && resizedDetections.length > 0) {
                    const faceBox = resizedDetections[0].detection.box;
                    setDetectedFaceBox(faceBox);
                }
            }

        }, 1000)
    }

    const captureImageFromVideo = (faceBox) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        const scale = videoRef.current.videoWidth / videoRef.current.offsetWidth;
        canvas.width = faceBox.width * scale;
        canvas.height = faceBox.height * scale;
        ctx.drawImage(
            videoRef.current,
            faceBox.x * scale, faceBox.y * scale, faceBox.width * scale, faceBox.height * scale,
            0, 0, canvas.width, canvas.height
        );
        return canvas.toDataURL();
    };

    const handleSaveImage = async () => {
        const base64Image = captureImageFromVideo(detectedFaceBox);

        try {
            const emotionReturned = await sendImageToServer(base64Image);
            console.log(emotionReturned);
            const currentTime = new Date();
            const imageSrc = canvasRef.current.toDataURL('image/jpeg');

            // Yeni imajı kaydet
            setSavedImages(prevImages => [...prevImages, {
                src: imageSrc,
                srcRealPhoto: base64Image,
                width: 4,
                height: 3,
                emotion: emotionReturned.emotion,
                // time: currentTime.toLocaleString()
                time: currentTime.getHours() + ':' + currentTime.getMinutes() + ':' + currentTime.getSeconds()
            }]);
        } catch (error) {
            console.error('Error:', error);
        }
    };

    const sendImageToServer = async (base64Image) => {
        const response = await fetch('http://127.0.0.1:5500/predict-emotion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: base64Image }),
        });
        return await response.json();
    };

    const closeWebcam = () => {
        videoRef.current.pause();
        videoRef.current.srcObject.getTracks()[0].stop();
        setCaptureVideo(false);
    }

    return (
        <div>
            <div style={{ textAlign: 'center', padding: '10px' }}>
                {
                    captureVideo && modelsLoaded ?
                        <button onClick={closeWebcam} style={{ cursor: 'pointer', backgroundColor: 'green', color: 'white', padding: '15px', fontSize: '25px', border: 'none', borderRadius: '10px' }}>
                            Close Webcam
                        </button>
                        :
                        <button onClick={startVideo} style={{ cursor: 'pointer', backgroundColor: 'green', color: 'white', padding: '15px', fontSize: '25px', border: 'none', borderRadius: '10px' }}>
                            Open Webcam
                        </button>
                }
            </div>
            {
                captureVideo ?
                    modelsLoaded ?
                        <div>
                            <div style={{ display: 'flex', justifyContent: 'center', padding: '10px' }}>
                                <video ref={videoRef} height={videoHeight} width={videoWidth} onPlay={handleVideoOnPlay} style={{ borderRadius: '10px' }} />
                                <canvas ref={canvasRef} style={{ position: 'absolute' }} />
                            </div>
                        </div>
                        :
                        <div>loading...</div>
                    :
                    <>
                    </>
            }
            <div onClick={handleSaveImage} className='cursor-pointer w-1/2 flex justify-center items-center shadow-[0_20px_50px_rgba(8,_112,_184,_0.7)] bg-black text-white rounded-md m-2 p-2'>
                <span>Save Image</span>
            </div>

            <PhotoAlbumComponent savedImages={savedImages} />
        </div>
    );
}

export default WebcamComponent;