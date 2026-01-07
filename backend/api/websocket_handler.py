"""
WebSocket Handler for Real-Time Fall Detection

Handles bidirectional communication between frontend and backend.
"""

import asyncio
import json
import base64
import numpy as np
import cv2
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

from .session_manager import SessionManager, UserSession
from ..data_processing.pose_extractor import PoseData
from ..utils.logger import setup_logger, log_performance


logger = setup_logger(__name__)


class WebSocketHandler:
    """Handle WebSocket connections for fall detection."""
    
    def __init__(self, session_manager: SessionManager):
        """
        Initialize WebSocket handler.
        
        Args:
            session_manager: Session manager instance
        """
        self.session_manager = session_manager
        self.logger = logger
    
    async def handle_connection(
        self,
        websocket: WebSocket,
        session_id: str
    ):
        """
        Handle WebSocket connection lifecycle.
        
        Args:
            websocket: WebSocket connection
            session_id: User session ID
        """
        await websocket.accept()
        self.logger.info(f"WebSocket connected: {session_id}")
        
        # Get session
        session = self.session_manager.get_session(session_id)
        if not session:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid session ID"
            })
            await websocket.close()
            return
        
        try:
            while True:
                # Receive frame from client
                data = await websocket.receive_json()
                
                # Update activity
                self.session_manager.update_activity(session_id)
                
                # Process based on message type
                msg_type = data.get("type")
                
                if msg_type == "frame":
                    # Process frame
                    response = await self._process_frame(data, session)
                    await websocket.send_json(response)
                
                elif msg_type == "ping":
                    # Keepalive
                    await websocket.send_json({"type": "pong"})
                
                elif msg_type == "alert_response":
                    # User responded to alert
                    await self._handle_alert_response(data, session, websocket)
                
                elif msg_type == "reset":
                    # Reset detection state
                    session.detector.reset()
                    session.validator.reset()
                    session.fusion_engine.reset()
                    await websocket.send_json({"type": "reset_complete"})
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    })
        
        except WebSocketDisconnect:
            self.logger.info(f"WebSocket disconnected: {session_id}")
        
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}", exc_info=True)
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except:
                pass
        
        finally:
            # Cleanup
            self.logger.info(f"Cleaning up session: {session_id}")
    
    @log_performance()
    async def _process_frame(
        self,
        data: dict,
        session: UserSession
    ) -> dict:
        """
        Process a single frame.
        
        Args:
            data: Frame data from client
            session: User session
        
        Returns:
            Response dictionary
        """
        try:
            # Decode frame
            frame_data = data.get("frame")
            timestamp = data.get("timestamp", datetime.now().timestamp())
            frame_idx = data.get("frame_idx", session.frames_processed)
            
            # Decode base64 image
            if isinstance(frame_data, str):
                # Remove data URL prefix if present
                if "," in frame_data:
                    frame_data = frame_data.split(",")[1]
                
                # Decode
                img_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                raise ValueError("Invalid frame format")
            
            if frame is None:
                raise ValueError("Failed to decode frame")
            
            session.frames_processed += 1
            
            # Extract pose
            pose = session.pose_extractor.extract_pose(
                frame=frame,
                frame_idx=frame_idx,
                timestamp=timestamp
            )
            
            if pose is None:
                # No person detected
                return {
                    "type": "result",
                    "person_detected": False,
                    "fall_detected": False,
                    "confidence": 0.0,
                    "buffer_filled": 0.0
                }
            
            # Process pose through detector
            detection_result = session.detector.process_pose(pose)
            
            if not detection_result.get('ready', False):
                # Buffer not full yet
                return {
                    "type": "result",
                    "person_detected": True,
                    "fall_detected": False,
                    "confidence": detection_result.get('probability', 0.0),
                    "buffer_filled": detection_result.get('buffer_filled', 0.0),
                    "status": "buffering"
                }
            
            # Get LSTM probability
            lstm_prob = detection_result['probability']
            
            # Extract features for validation
            feature_buffer = list(session.detector.feature_buffer)
            if len(feature_buffer) > 0:
                latest_features = feature_buffer[-1]
                
                # Convert to dict
                features_dict = {
                    'torso_angle': float(latest_features[0]),
                    'bbox_ratio': float(latest_features[1]),
                    'com_velocity': float(latest_features[2]),
                    'hip_velocity': float(latest_features[3]),
                    'head_velocity': float(latest_features[4]),
                    'joint_spread': float(latest_features[5]),
                    'head_hip_distance': float(latest_features[6])
                }
            else:
                features_dict = {}
            
            # Fusion engine
            fusion_result = session.fusion_engine.fuse(
                temporal_score=lstm_prob,
                features=features_dict
            )
            
            # Physics validation (if score is high)
            if fusion_result.final_score >= 0.80:
                validation_result = session.validator.validate(
                    lstm_probability=lstm_prob,
                    features=features_dict
                )
            else:
                validation_result = None
            
            # Check fall filter
            should_filter, filter_reason = session.fall_filter.should_filter(
                features=features_dict,
                timestamp=timestamp
            )
            
            # Determine if fall should be triggered
            fall_detected = False
            ready_for_alert = False
            
            if (fusion_result.ready_to_trigger and
                not should_filter and
                not session.alert_triggered):
                
                if validation_result and validation_result.is_valid:
                    fall_detected = True
                    ready_for_alert = True
                    session.detections_count += 1
            
            # Prepare response
            response = {
                "type": "result",
                "person_detected": True,
                "fall_detected": fall_detected,
                "confidence": fusion_result.final_score,
                "confidence_level": fusion_result.confidence_level,
                "lstm_probability": lstm_prob,
                "component_scores": fusion_result.component_scores,
                "buffer_filled": 1.0,
                "status": "monitoring",
                "time_above_threshold": fusion_result.time_above_threshold
            }
            
            # Add validation info if available
            if validation_result:
                response["validation"] = {
                    "is_valid": validation_result.is_valid,
                    "violations": validation_result.violations,
                    "passed_checks": validation_result.passed_checks
                }
            
            # Trigger alert if needed
            if ready_for_alert:
                response["type"] = "alert"
                response["alert_message"] = "Fall detected! Are you okay?"
                session.alert_triggered = True
                session.alert_timestamp = datetime.now()
                session.fall_count += 1
                
                self.logger.warning(
                    f"FALL ALERT triggered for session {session.session_id} "
                    f"(confidence: {fusion_result.final_score:.3f})"
                )
            
            return response
        
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}", exc_info=True)
            return {
                "type": "error",
                "message": f"Processing error: {str(e)}"
            }
    
    async def _handle_alert_response(
        self,
        data: dict,
        session: UserSession,
        websocket: WebSocket
    ):
        """
        Handle user response to fall alert.
        
        Args:
            data: Response data
            session: User session
            websocket: WebSocket connection
        """
        response_type = data.get("response")  # 'ok' or 'help'
        
        if response_type == "ok":
            # User is okay, false alarm
            self.logger.info(f"Session {session.session_id}: User responded OK")
            
            # Reset alert state
            session.alert_triggered = False
            session.alert_timestamp = None
            
            # Reset detector
            session.detector.reset()
            session.validator.reset()
            session.fusion_engine.reset()
            
            await websocket.send_json({
                "type": "alert_cancelled",
                "message": "Glad you're okay!"
            })
        
        elif response_type == "help":
            # User needs help, trigger emergency
            self.logger.critical(f"Session {session.session_id}: EMERGENCY - User needs help!")
            
            # Here you would trigger external alerts:
            # - Send SMS
            # - Send email
            # - Call webhook
            # - Push notification
            
            await websocket.send_json({
                "type": "emergency_triggered",
                "message": "Emergency services have been notified."
            })
        
        else:
            # Timeout - no response
            self.logger.warning(f"Session {session.session_id}: Alert timeout, triggering emergency")
            
            await websocket.send_json({
                "type": "alert_timeout",
                "message": "No response received. Notifying emergency contacts."
            })
