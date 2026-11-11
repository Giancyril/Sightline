import React, { useEffect, useRef, useState } from "react";
import { Camera, CameraOff, RefreshCw } from "lucide-react";

interface WebcamCaptureProps {
  onCapture: (blob: Blob) => void;
  isProcessing: boolean;
}

export const WebcamCapture: React.FC<WebcamCaptureProps> = ({ onCapture, isProcessing }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [streamActive, setStreamActive] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const startCamera = async () => {
    try {
      setErrorMsg(null);
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play();
        setStreamActive(true);
      }
    } catch (err) {
      setErrorMsg(`Unable to access camera: ${(err as Error).message}`);
      setStreamActive(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setStreamActive(false);
    }
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const captureFrame = () => {
    const video = videoRef.current;
    if (!video || !streamActive) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (blob) {
          onCapture(blob);
        }
      },
      "image/jpeg",
      0.9
    );
  };

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      <div className="relative w-full max-w-lg aspect-video bg-slate-900 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
        <video
          ref={videoRef}
          playsInline
          muted
          className={`w-full h-full object-cover ${streamActive ? "block" : "hidden"}`}
        />
        {!streamActive && (
          <div className="text-center p-6">
            <Camera className="w-12 h-12 text-slate-600 mx-auto mb-2" />
            <p className="text-sm text-slate-400">Webcam is currently inactive</p>
          </div>
        )}

        {isProcessing && (
          <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm flex items-center justify-center">
            <div className="flex items-center gap-2 text-sky-400 font-medium">
              <RefreshCw className="w-5 h-5 animate-spin" />
              <span>Running Inference...</span>
            </div>
          </div>
        )}
      </div>

      {errorMsg && <p className="text-sm text-red-400">{errorMsg}</p>}

      <div className="flex items-center gap-3">
        {!streamActive ? (
          <button
            type="button"
            onClick={startCamera}
            className="flex items-center gap-2 px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-sm font-medium transition"
          >
            <Camera className="w-4 h-4" />
            <span>Enable Webcam</span>
          </button>
        ) : (
          <>
            <button
              type="button"
              onClick={captureFrame}
              disabled={isProcessing}
              className="flex items-center gap-2 px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg text-sm font-semibold shadow-lg shadow-emerald-950 transition"
            >
              <Camera className="w-4 h-4" />
              <span>Capture & Detect</span>
            </button>
            <button
              type="button"
              onClick={stopCamera}
              className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm font-medium transition"
            >
              <CameraOff className="w-4 h-4" />
              <span>Turn Off</span>
            </button>
          </>
        )}
      </div>
    </div>
  );
};

