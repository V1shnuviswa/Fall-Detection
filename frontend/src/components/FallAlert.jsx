/**
 * Fall Alert Component
 * 
 * Displays alert modal when fall is detected.
 * Includes human confirmation loop.
 */

import React, { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle, Phone } from 'lucide-react';

const FallAlert = ({ onResponse, timeout = 15 }) => {
  const [timeLeft, setTimeLeft] = useState(timeout);
  const [responded, setResponded] = useState(false);

  useEffect(() => {
    if (responded) return;

    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          // Timeout - trigger help
          if (!responded) {
            handleResponse('timeout');
          }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    // Play alert sound
    playAlertSound();

    return () => clearInterval(timer);
  }, [responded]);

  const playAlertSound = () => {
    // Create audio context for alert sound
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);

    // Repeat 3 times
    setTimeout(() => playAlertSound(), 600);
  };

  const handleResponse = (response) => {
    if (responded) return;
    setResponded(true);
    onResponse(response);
  };

  return (
    <div className="alert-overlay">
      <div className="alert-modal">
        <div className="alert-header">
          <AlertTriangle className="alert-icon" size={48} />
          <h2>Fall Detected!</h2>
        </div>

        <div className="alert-body">
          <p className="alert-message">
            Our system detected a potential fall. Are you okay?
          </p>

          {!responded && (
            <div className="countdown">
              <p>Responding in:</p>
              <div className="countdown-timer">{timeLeft}s</div>
              <div className="countdown-progress">
                <div
                  className="countdown-progress-bar"
                  style={{ width: `${(timeLeft / timeout) * 100}%` }}
                />
              </div>
            </div>
          )}

          {responded && (
            <div className="response-confirmation">
              <CheckCircle size={32} />
              <p>Response recorded</p>
            </div>
          )}
        </div>

        {!responded && (
          <div className="alert-actions">
            <button
              className="btn btn-ok"
              onClick={() => handleResponse('ok')}
            >
              <CheckCircle size={20} />
              I'm Okay
            </button>

            <button
              className="btn btn-help"
              onClick={() => handleResponse('help')}
            >
              <Phone size={20} />
              I Need Help
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default FallAlert;
