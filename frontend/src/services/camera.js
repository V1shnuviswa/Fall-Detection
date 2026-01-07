/**
 * Camera Service
 * 
 * Handles device camera access and frame capture.
 */

class CameraService {
  constructor() {
    this.stream = null;
    this.videoElement = null;
    this.canvas = null;
    this.context = null;
    this.isStreaming = false;
    this.targetFPS = 10;
    this.frameInterval = 1000 / this.targetFPS;
    this.lastFrameTime = 0;
  }

  /**
   * Start camera stream
   */
  async startCamera(videoElement, options = {}) {
    try {
      const constraints = {
        video: {
          width: { ideal: options.width || 640 },
          height: { ideal: options.height || 480 },
          frameRate: { ideal: this.targetFPS },
          facingMode: options.facingMode || 'user'
        },
        audio: false
      };

      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.videoElement = videoElement;
      this.videoElement.srcObject = this.stream;
      
      await new Promise((resolve) => {
        this.videoElement.onloadedmetadata = () => {
          this.videoElement.play();
          resolve();
        };
      });

      // Create canvas for frame capture
      this.canvas = document.createElement('canvas');
      this.canvas.width = this.videoElement.videoWidth;
      this.canvas.height = this.videoElement.videoHeight;
      this.context = this.canvas.getContext('2d');

      this.isStreaming = true;
      
      console.log('Camera started successfully');
      console.log(`Resolution: ${this.canvas.width}x${this.canvas.height}`);
      
      return true;
    } catch (error) {
      console.error('Error starting camera:', error);
      throw error;
    }
  }

  /**
   * Capture frame from video
   */
  captureFrame() {
    if (!this.isStreaming || !this.videoElement || !this.canvas) {
      return null;
    }

    try {
      // Draw current video frame to canvas
      this.context.drawImage(
        this.videoElement,
        0, 0,
        this.canvas.width,
        this.canvas.height
      );

      // Convert to base64 JPEG
      const dataURL = this.canvas.toDataURL('image/jpeg', 0.85);
      
      return dataURL;
    } catch (error) {
      console.error('Error capturing frame:', error);
      return null;
    }
  }

  /**
   * Check if enough time has passed for next frame
   */
  shouldCaptureFrame() {
    const now = performance.now();
    if (now - this.lastFrameTime >= this.frameInterval) {
      this.lastFrameTime = now;
      return true;
    }
    return false;
  }

  /**
   * Stop camera stream
   */
  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    if (this.videoElement) {
      this.videoElement.srcObject = null;
      this.videoElement = null;
    }

    this.canvas = null;
    this.context = null;
    this.isStreaming = false;
    
    console.log('Camera stopped');
  }

  /**
   * Get camera constraints
   */
  getConstraints() {
    if (!this.stream) {
      return null;
    }

    const videoTrack = this.stream.getVideoTracks()[0];
    return videoTrack.getSettings();
  }

  /**
   * Switch camera (front/back)
   */
  async switchCamera() {
    const currentFacingMode = this.getConstraints()?.facingMode || 'user';
    const newFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    
    this.stopCamera();
    
    await this.startCamera(this.videoElement, {
      facingMode: newFacingMode
    });
  }
}

export default new CameraService();
