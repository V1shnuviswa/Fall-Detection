/**
 * Camera Feed Component
 * 
 * Displays video feed and detection status.
 */

import React, { useRef, useEffect, useState } from 'react';
import { Camera, CameraOff, Activity } from 'lucide-react';
import cameraService from '../services/camera';

const CameraFeed = ({ isActive, onFrame }) => {
  const videoRef = useRef(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [error, setError] = useState(null);
  const frameCountRef = useRef(0);

  useEffect(() => {
    if (isActive) {
      startCamera();
    } else {
      stopCamera();
    }

    return () => {
      stopCamera();
    };
  }, [isActive]);

  useEffect(() => {
    if (!isCameraReady || !isActive) return;

    const captureLoop = () => {
      if (!isActive) return;

      // Check if enough time has passed for next frame
      if (cameraService.shouldCaptureFrame()) {
        const frame = cameraService.captureFrame();
        
        if (frame && onFrame) {
          frameCountRef.current++;
          onFrame(frame, Date.now() / 1000, frameCountRef.current);
        }
      }

      requestAnimationFrame(captureLoop);
    };

    captureLoop();
  }, [isCameraReady, isActive, onFrame]);

  const startCamera = async () => {
    try {
      setError(null);
      await cameraService.startCamera(videoRef.current, {
        width: 640,
        height: 480,
        facingMode: 'user'
      });
      setIsCameraReady(true);
    } catch (err) {
      console.error('Camera error:', err);
      setError('Failed to access camera. Please check permissions.');
      setIsCameraReady(false);
    }
  };

  const stopCamera = () => {
    cameraService.stopCamera();
    setIsCameraReady(false);
    frameCountRef.current = 0;
  };

  return (
    <div className="camera-feed">
      <div className="camera-container">
        {error && (
          <div className="camera-error">
            <CameraOff size={48} />
            <p>{error}</p>
            <button onClick={startCamera} className="btn btn-primary">
              Retry
            </button>
          </div>
        )}

        {!error && (
          <>
            <video
              ref={videoRef}
              className="camera-video"
              autoPlay
              playsInline
              muted
            />

            {isCameraReady && (
              <div className="camera-overlay">
                <div className="camera-status">
                  <Activity size={16} />
                  <span>Monitoring Active</span>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default CameraFeed;
