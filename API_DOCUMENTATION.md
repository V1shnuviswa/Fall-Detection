# 📡 API Documentation

## Base URL

```
Development: http://localhost:8000
Production: https://your-domain.com/api
```

## Authentication

Currently, the API doesn't require authentication for development. For production, implement token-based authentication.

---

## REST Endpoints

### Health Check

#### GET `/health`

Check if the service is healthy and operational.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "active_sessions": 3
}
```

---

### Get Statistics

#### GET `/stats`

Get system-wide statistics.

**Response:**
```json
{
  "active_sessions": 5,
  "total_frames": 15420,
  "total_detections": 234,
  "total_falls": 3
}
```

---

### Session Management

#### POST `/session/create`

Create a new detection session.

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "websocket_url": "/ws/550e8400-e29b-41d4-a716-446655440000"
}
```

#### GET `/session/{session_id}`

Get session information.

**Parameters:**
- `session_id` (path): Session UUID

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-06T10:30:00",
  "last_activity": "2024-01-06T10:35:23",
  "frames_processed": 1542,
  "detections_count": 23,
  "fall_count": 0,
  "alert_triggered": false
}
```

#### DELETE `/session/{session_id}`

Delete a session.

**Parameters:**
- `session_id` (path): Session UUID

**Response:**
```json
{
  "message": "Session deleted successfully"
}
```

---

### Configuration

#### GET `/config`

Get system configuration.

**Response:**
```json
{
  "fall_threshold": 0.88,
  "confirmation_seconds": 2,
  "alert_timeout": 15,
  "target_fps": 10,
  "buffer_size": 30,
  "device": "cuda"
}
```

---

## WebSocket Protocol

### Connection

#### WS `/ws/{session_id}`

Connect to WebSocket for real-time fall detection.

**URL:** `ws://localhost:8000/ws/{session_id}`

---

### Client → Server Messages

#### 1. Send Frame

Send a video frame for processing.

**Message:**
```json
{
  "type": "frame",
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "timestamp": 1704537600.123,
  "frame_idx": 42
}
```

**Fields:**
- `type`: "frame"
- `frame`: Base64-encoded JPEG image (with or without data URL prefix)
- `timestamp`: Unix timestamp (float)
- `frame_idx`: Frame index (integer)

---

#### 2. Alert Response

Respond to fall alert.

**Message:**
```json
{
  "type": "alert_response",
  "response": "ok"
}
```

**Fields:**
- `type`: "alert_response"
- `response`: "ok" | "help"

**Response Values:**
- `"ok"`: User is okay, cancel alert
- `"help"`: User needs help, trigger emergency

---

#### 3. Reset

Reset detection state.

**Message:**
```json
{
  "type": "reset"
}
```

---

#### 4. Ping

Keepalive ping.

**Message:**
```json
{
  "type": "ping"
}
```

---

### Server → Client Messages

#### 1. Detection Result

Normal detection result (no fall).

**Message:**
```json
{
  "type": "result",
  "person_detected": true,
  "fall_detected": false,
  "confidence": 0.42,
  "confidence_level": "low",
  "lstm_probability": 0.38,
  "component_scores": {
    "temporal": 0.38,
    "posture": 0.25,
    "motion": 0.15
  },
  "buffer_filled": 1.0,
  "status": "monitoring",
  "time_above_threshold": 0.0
}
```

**Fields:**
- `type`: "result"
- `person_detected`: Boolean - person visible in frame
- `fall_detected`: Boolean - fall detected (not yet confirmed)
- `confidence`: Float [0-1] - final fused confidence score
- `confidence_level`: "low" | "medium" | "high" | "critical"
- `lstm_probability`: Float [0-1] - LSTM model output
- `component_scores`: Object with temporal, posture, motion scores
- `buffer_filled`: Float [0-1] - temporal buffer fill percentage
- `status`: "buffering" | "monitoring"
- `time_above_threshold`: Float - seconds above threshold

---

#### 2. Fall Alert

Fall confirmed, waiting for user response.

**Message:**
```json
{
  "type": "alert",
  "alert_message": "Fall detected! Are you okay?",
  "person_detected": true,
  "fall_detected": true,
  "confidence": 0.92,
  "confidence_level": "critical",
  "lstm_probability": 0.91,
  "component_scores": {
    "temporal": 0.91,
    "posture": 0.95,
    "motion": 0.88
  },
  "validation": {
    "is_valid": true,
    "violations": [],
    "passed_checks": [
      "LSTM probability: 0.910 >= 0.85",
      "Torso angle: 18.3° <= 30.0°",
      "Bbox ratio: 0.68 <= 1.0"
    ]
  },
  "buffer_filled": 1.0,
  "status": "monitoring",
  "time_above_threshold": 2.1
}
```

---

#### 3. Alert Cancelled

User responded "OK".

**Message:**
```json
{
  "type": "alert_cancelled",
  "message": "Glad you're okay!"
}
```

---

#### 4. Emergency Triggered

User responded "Help" or timeout occurred.

**Message:**
```json
{
  "type": "emergency_triggered",
  "message": "Emergency services have been notified."
}
```

---

#### 5. Alert Timeout

No response received within timeout period.

**Message:**
```json
{
  "type": "alert_timeout",
  "message": "No response received. Notifying emergency contacts."
}
```

---

#### 6. Reset Complete

Detection state reset successfully.

**Message:**
```json
{
  "type": "reset_complete"
}
```

---

#### 7. Pong

Response to ping.

**Message:**
```json
{
  "type": "pong"
}
```

---

#### 8. Error

Error occurred during processing.

**Message:**
```json
{
  "type": "error",
  "message": "Processing error: Invalid frame format"
}
```

---

## Example Usage

### Python Client

```python
import asyncio
import websockets
import json
import base64

async def fall_detection_client():
    uri = "ws://localhost:8000/ws/your-session-id"
    
    async with websockets.connect(uri) as websocket:
        # Send frame
        with open("frame.jpg", "rb") as f:
            frame_data = base64.b64encode(f.read()).decode()
        
        await websocket.send(json.dumps({
            "type": "frame",
            "frame": f"data:image/jpeg;base64,{frame_data}",
            "timestamp": time.time(),
            "frame_idx": 0
        }))
        
        # Receive result
        response = await websocket.recv()
        data = json.loads(response)
        
        if data["type"] == "alert":
            print("Fall detected!")
            # Send response
            await websocket.send(json.dumps({
                "type": "alert_response",
                "response": "ok"
            }))

asyncio.run(fall_detection_client())
```

---

### JavaScript Client

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/your-session-id');

ws.onopen = () => {
  console.log('Connected');
  
  // Send frame
  const canvas = document.getElementById('canvas');
  const frameData = canvas.toDataURL('image/jpeg', 0.85);
  
  ws.send(JSON.stringify({
    type: 'frame',
    frame: frameData,
    timestamp: Date.now() / 1000,
    frame_idx: 0
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'alert') {
    console.log('Fall detected!');
    
    // Send response
    ws.send(JSON.stringify({
      type: 'alert_response',
      response: 'ok'
    }));
  }
};
```

---

## Rate Limits

- **Frame submissions**: 10 FPS (recommended)
- **Session creation**: 10 per minute per IP
- **WebSocket connections**: 5 concurrent per IP

---

## Error Codes

| Code | Message | Description |
|------|---------|-------------|
| 400 | Bad Request | Invalid request format |
| 404 | Not Found | Session not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service not initialized |

---

## Best Practices

### Frame Submission

1. **Target 8-10 FPS** for optimal performance
2. **Resize frames** to 640×480 before sending
3. **Use JPEG** with 80-85% quality
4. **Send timestamp** with each frame for synchronization

### Connection Management

1. **Handle reconnection** with exponential backoff
2. **Implement ping/pong** for keepalive
3. **Close connection** when monitoring stops
4. **Create new session** after extended disconnection

### Alert Handling

1. **Display alert** immediately when received
2. **Start countdown timer** (15 seconds)
3. **Send response** as soon as user interacts
4. **Handle timeout** gracefully

---

## Monitoring

### Metrics Endpoint (Optional)

If Prometheus metrics are enabled:

```
GET /metrics
```

**Metrics:**
- `fall_detection_frames_total`: Total frames processed
- `fall_detection_detections_total`: Total detections
- `fall_detection_alerts_total`: Total alerts triggered
- `fall_detection_latency_seconds`: Processing latency histogram
- `fall_detection_active_sessions`: Current active sessions

---

## Support

For API issues or questions:
- Check logs: `logs/fall_detection.log`
- Enable DEBUG logging in `.env`
- Check API docs: `http://localhost:8000/docs`
