/**
 * WebSocket Service for Real-Time Fall Detection
 * 
 * Handles bidirectional communication with backend.
 */

class WebSocketService {
  constructor() {
    this.ws = null;
    this.sessionId = null;
    this.isConnected = false;
    this.messageHandlers = new Map();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  /**
   * Connect to WebSocket server
   */
  async connect(sessionId) {
    return new Promise((resolve, reject) => {
      this.sessionId = sessionId;
      
      // Construct WebSocket URL
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsHost = window.location.hostname;
      const wsPort = window.location.port || (wsProtocol === 'wss:' ? '443' : '80');
      const wsUrl = `${wsProtocol}//${wsHost}:${wsPort}/ws/${sessionId}`;
      
      console.log(`Connecting to WebSocket: ${wsUrl}`);
      
      try {
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          resolve();
        };
        
        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };
        
        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };
        
        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          this.isConnected = false;
          this.attemptReconnect();
        };
      } catch (error) {
        console.error('Failed to create WebSocket:', error);
        reject(error);
      }
    });
  }

  /**
   * Handle incoming messages
   */
  handleMessage(event) {
    try {
      const data = JSON.parse(event.data);
      const messageType = data.type;
      
      // Call registered handlers
      if (this.messageHandlers.has(messageType)) {
        const handlers = this.messageHandlers.get(messageType);
        handlers.forEach(handler => handler(data));
      }
      
      // Call general handler
      if (this.messageHandlers.has('*')) {
        const handlers = this.messageHandlers.get('*');
        handlers.forEach(handler => handler(data));
      }
    } catch (error) {
      console.error('Error handling message:', error);
    }
  }

  /**
   * Register message handler
   */
  on(messageType, handler) {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, []);
    }
    this.messageHandlers.get(messageType).push(handler);
  }

  /**
   * Unregister message handler
   */
  off(messageType, handler) {
    if (this.messageHandlers.has(messageType)) {
      const handlers = this.messageHandlers.get(messageType);
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  /**
   * Send frame to server
   */
  sendFrame(frameData, timestamp, frameIdx) {
    if (!this.isConnected || !this.ws) {
      console.warn('WebSocket not connected');
      return false;
    }
    
    try {
      const message = {
        type: 'frame',
        frame: frameData,
        timestamp: timestamp || Date.now() / 1000,
        frame_idx: frameIdx || 0
      };
      
      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error sending frame:', error);
      return false;
    }
  }

  /**
   * Send alert response
   */
  sendAlertResponse(response) {
    if (!this.isConnected || !this.ws) {
      console.warn('WebSocket not connected');
      return false;
    }
    
    try {
      const message = {
        type: 'alert_response',
        response: response  // 'ok' or 'help'
      };
      
      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error sending alert response:', error);
      return false;
    }
  }

  /**
   * Send reset command
   */
  reset() {
    if (!this.isConnected || !this.ws) {
      console.warn('WebSocket not connected');
      return false;
    }
    
    try {
      const message = {
        type: 'reset'
      };
      
      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error sending reset:', error);
      return false;
    }
  }

  /**
   * Attempt to reconnect
   */
  attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }
    
    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      if (this.sessionId) {
        this.connect(this.sessionId).catch(error => {
          console.error('Reconnection failed:', error);
        });
      }
    }, delay);
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
    this.sessionId = null;
  }
}

// Export singleton instance
export default new WebSocketService();
