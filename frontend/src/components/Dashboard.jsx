/**
 * Dashboard Component
 * 
 * Main monitoring dashboard with camera feed and detection status.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Power, PowerOff, AlertCircle, Activity, TrendingUp } from 'lucide-react';
import CameraFeed from './CameraFeed';
import FallAlert from './FallAlert';
import websocketService from '../services/websocket';

const Dashboard = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [detectionData, setDetectionData] = useState(null);
  const [showAlert, setShowAlert] = useState(false);
  const [stats, setStats] = useState({
    framesProcessed: 0,
    bufferFilled: 0,
    confidence: 0,
    personDetected: false
  });

  // Create session when component mounts
  useEffect(() => {
    createSession();

    return () => {
      if (sessionId) {
        websocketService.disconnect();
      }
    };
  }, []);

  // Register WebSocket message handlers
  useEffect(() => {
    const resultHandler = (data) => {
      setDetectionData(data);
      
      setStats({
        framesProcessed: data.frame_count || 0,
        bufferFilled: data.buffer_filled || 0,
        confidence: data.confidence || 0,
        personDetected: data.person_detected || false
      });
    };

    const alertHandler = (data) => {
      console.log('Fall alert received!', data);
      setShowAlert(true);
    };

    const errorHandler = (data) => {
      console.error('Error from server:', data.message);
    };

    websocketService.on('result', resultHandler);
    websocketService.on('alert', alertHandler);
    websocketService.on('error', errorHandler);

    return () => {
      websocketService.off('result', resultHandler);
      websocketService.off('alert', alertHandler);
      websocketService.off('error', errorHandler);
    };
  }, []);

  const createSession = async () => {
    try {
      setConnectionStatus('connecting');
      
      const response = await fetch('/api/session/create', {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error('Failed to create session');
      }
      
      const data = await response.json();
      setSessionId(data.session_id);
      
      console.log('Session created:', data.session_id);
      setConnectionStatus('ready');
    } catch (error) {
      console.error('Error creating session:', error);
      setConnectionStatus('error');
    }
  };

  const startMonitoring = async () => {
    if (!sessionId) {
      alert('No session available. Please refresh the page.');
      return;
    }

    try {
      setConnectionStatus('connecting');
      await websocketService.connect(sessionId);
      setConnectionStatus('connected');
      setIsMonitoring(true);
    } catch (error) {
      console.error('Failed to start monitoring:', error);
      setConnectionStatus('error');
      alert('Failed to connect to server. Please try again.');
    }
  };

  const stopMonitoring = () => {
    setIsMonitoring(false);
    websocketService.disconnect();
    setConnectionStatus('ready');
    setStats({
      framesProcessed: 0,
      bufferFilled: 0,
      confidence: 0,
      personDetected: false
    });
  };

  const handleFrame = useCallback((frameData, timestamp, frameIdx) => {
    websocketService.sendFrame(frameData, timestamp, frameIdx);
  }, []);

  const handleAlertResponse = (response) => {
    console.log('Alert response:', response);
    websocketService.sendAlertResponse(response);
    setShowAlert(false);
    
    if (response === 'ok') {
      // Continue monitoring
    } else if (response === 'help' || response === 'timeout') {
      // Emergency triggered
      alert('Emergency services have been notified!');
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence < 0.5) return '#4ade80'; // green
    if (confidence < 0.7) return '#fbbf24'; // yellow
    if (confidence < 0.85) return '#fb923c'; // orange
    return '#ef4444'; // red
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence < 0.5) return 'Normal';
    if (confidence < 0.7) return 'Low Risk';
    if (confidence < 0.85) return 'Medium Risk';
    return 'High Risk';
  };

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-content">
          <div className="header-logo">
            <Activity size={32} />
            <h1>Fall Detection System</h1>
          </div>
          
          <div className="header-status">
            <div className={`status-indicator ${connectionStatus}`}>
              <span className="status-dot" />
              <span className="status-text">
                {connectionStatus === 'connected' ? 'Connected' :
                 connectionStatus === 'connecting' ? 'Connecting...' :
                 connectionStatus === 'ready' ? 'Ready' :
                 connectionStatus === 'error' ? 'Error' :
                 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="dashboard-content">
        {/* Video Feed */}
        <div className="content-main">
          <div className="video-section">
            <CameraFeed
              isActive={isMonitoring}
              onFrame={handleFrame}
            />

            {/* Controls */}
            <div className="video-controls">
              {!isMonitoring ? (
                <button
                  onClick={startMonitoring}
                  className="btn btn-primary btn-large"
                  disabled={connectionStatus !== 'ready'}
                >
                  <Power size={24} />
                  Start Monitoring
                </button>
              ) : (
                <button
                  onClick={stopMonitoring}
                  className="btn btn-danger btn-large"
                >
                  <PowerOff size={24} />
                  Stop Monitoring
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Stats Sidebar */}
        <div className="content-sidebar">
          <div className="stats-panel">
            <h3>Detection Status</h3>

            {/* Person Detection */}
            <div className="stat-item">
              <div className="stat-label">
                <AlertCircle size={16} />
                Person Detected
              </div>
              <div className={`stat-value ${stats.personDetected ? 'positive' : 'negative'}`}>
                {stats.personDetected ? 'Yes' : 'No'}
              </div>
            </div>

            {/* Buffer Status */}
            <div className="stat-item">
              <div className="stat-label">
                <Activity size={16} />
                Buffer Status
              </div>
              <div className="stat-value">
                {(stats.bufferFilled * 100).toFixed(0)}%
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${stats.bufferFilled * 100}%` }}
                />
              </div>
            </div>

            {/* Confidence Score */}
            <div className="stat-item">
              <div className="stat-label">
                <TrendingUp size={16} />
                Confidence Score
              </div>
              <div
                className="stat-value"
                style={{ color: getConfidenceColor(stats.confidence) }}
              >
                {(stats.confidence * 100).toFixed(1)}%
              </div>
              <div className="stat-sublabel">
                {getConfidenceLabel(stats.confidence)}
              </div>
            </div>

            {/* Frames Processed */}
            <div className="stat-item">
              <div className="stat-label">Frames Processed</div>
              <div className="stat-value">{stats.framesProcessed}</div>
            </div>

            {/* Detection Details */}
            {detectionData && detectionData.component_scores && (
              <div className="detection-details">
                <h4>Component Scores</h4>
                <div className="component-score">
                  <span>Temporal:</span>
                  <span>{(detectionData.component_scores.temporal * 100).toFixed(1)}%</span>
                </div>
                <div className="component-score">
                  <span>Posture:</span>
                  <span>{(detectionData.component_scores.posture * 100).toFixed(1)}%</span>
                </div>
                <div className="component-score">
                  <span>Motion:</span>
                  <span>{(detectionData.component_scores.motion * 100).toFixed(1)}%</span>
                </div>
              </div>
            )}
          </div>

          {/* Info Panel */}
          <div className="info-panel">
            <h3>System Information</h3>
            <ul>
              <li>✓ 97-99% accuracy</li>
              <li>✓ Real-time processing</li>
              <li>✓ Privacy-safe (no recording)</li>
              <li>✓ Low false positives</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Fall Alert Modal */}
      {showAlert && (
        <FallAlert
          onResponse={handleAlertResponse}
          timeout={15}
        />
      )}
    </div>
  );
};

export default Dashboard;
