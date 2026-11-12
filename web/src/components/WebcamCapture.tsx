import React, { useEffect, useRef, useState } from "react";
import { Camera, CameraOff } from "lucide-react";
import { Button } from "./ui/Button";

interface WebcamCaptureProps {
  onCapture: (blob: Blob) => void;
  isProcessing: boolean;
}

export const WebcamCapture: React.FC<WebcamCaptureProps> = ({
  onCapture,
  isProcessing,
}) => {
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
        await videoRef.current.play();
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
    <div className="flex flex-col items-center justify-center gap-4 w-full max-w-lg mx-auto">
      <div className="relative w-full aspect-video bg-canvas rounded-panel overflow-hidden border border-border-subtle flex items-center justify-center">
        <video
          ref={videoRef}
          playsInline
          muted
          className={`w-full h-full object-cover ${streamActive ? "block" : "hidden"}`}
        />
        {!streamActive && (
          <div className="text-center p-6">
            <Camera className="w-10 h-10 text-text-tertiary mx-auto mb-2 opacity-60" />
            <p className="text-xs text-text-secondary">Camera stream inactive</p>
          </div>
        )}

        {isProcessing && (
          <div className="absolute inset-0 bg-canvas/80 backdrop-blur-sm flex items-center justify-center">
            <div className="flex items-center gap-2 text-xs font-medium text-brand">
              <span className="w-2 h-2 rounded-full bg-brand animate-ping" />
              <span>Running inference…</span>
            </div>
          </div>
        )}
      </div>

      {errorMsg && (
        <p className="text-xs text-status-danger font-medium">{errorMsg}</p>
      )}

      <div className="flex items-center gap-2">
        {!streamActive ? (
          <Button
            variant="primary"
            size="sm"
            onClick={startCamera}
            leftIcon={<Camera className="w-3.5 h-3.5" />}
          >
            Start camera
          </Button>
        ) : (
          <>
            <Button
              variant="primary"
              size="sm"
              onClick={captureFrame}
              isLoading={isProcessing}
              leftIcon={<Camera className="w-3.5 h-3.5" />}
            >
              Capture frame
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={stopCamera}
              leftIcon={<CameraOff className="w-3.5 h-3.5" />}
            >
              Stop
            </Button>
          </>
        )}
      </div>
    </div>
  );
};
