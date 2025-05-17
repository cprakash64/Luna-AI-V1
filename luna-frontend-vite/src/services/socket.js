// src/services/socket.js
import { io } from 'socket.io-client';
// Determine socket URL from environment variables or config
const getSocketUrl = () => {
  if (import.meta.env && import.meta.env.VITE_SOCKET_URL) {
    return import.meta.env.VITE_SOCKET_URL;
  }
  
  // Modified fallback: Always use explicit URL format with port
  const host = window.location.hostname;
  // Use port 5000 for backend server, not the frontend port
  return `http://${host}:5000`;
};
// Get the base API URL
const getApiUrl = () => {
  if (import.meta.env && import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  
  // If no API URL defined, construct one based on hostname
  const host = window.location.hostname;
  return `http://${host}:5000`; // Use same port as socket
};
// Create socket service singleton
class SocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.isConnecting = false;
    this.connectionPromise = null;
    this.tabId = null;
    this.registeredTab = null;
    this.registeredClient = null;
    
    // Feature detection flags
    this.topicsSupported = true;
    this.highlightsSupported = true;
    this.visualAnalysisSupported = true;
    this.transcriptionSupported = true; // Added transcription support flag
    
    // Active listeners tracking for cleanup
    this.activeListeners = new Map();
    this.registeredEvents = new Set();
    
    // Track instances of each event handler to prevent duplicates
    this.handlerRegistry = new Map();
    
    // Timestamp tracking
    this.lastTimestampFetchTime = 0;
    this.activeTimestampFetch = false;
    
    // Transcription tracking
    this.activeTranscriptionRequests = new Map(); // Track active transcription requests
    this.lastTranscriptionRequestTime = 0;
    
    // INCREASED timeout values
    this.DEFAULT_TIMEOUT = 30000;       // Increased from 20s to 30s
    this.TIMESTAMP_TIMEOUT = 20000;     // Increased from 15s to 20s
    this.CONNECTION_TIMEOUT = 45000;    // Increased from 30s to 45s
    this.TRANSCRIPTION_TIMEOUT = 120000; // 2 minutes for transcription
    // API base URL
    this.apiUrl = getApiUrl();
    
    // NEW: Add request tracking to prevent duplicate responses
    this.pendingRequests = new Map();
    this.processedResponses = new Set();
  }
  
  // Check if socket is connected
  isSocketConnected() {
    return this.socket && this.socket.connected && this.isConnected;
  }
  
  // Connect to socket server
  async connect() {
    // If already connected, return socket
    if (this.isConnected && this.socket) {
      console.log('Socket already connected, reusing connection');
      return Promise.resolve(this.socket);
    }
    
    // If connecting in progress, return existing promise
    if (this.isConnecting && this.connectionPromise) {
      console.log('Socket connection in progress, waiting for completion');
      return this.connectionPromise;
    }
    
    // Set connecting flag
    this.isConnecting = true;
    
    // Create a new connection promise
    this.connectionPromise = new Promise((resolve, reject) => {
      try {
        const socketUrl = getSocketUrl();
        console.log(`Connecting to socket server at ${socketUrl}`);
        
        // Create a new socket instance
        this.socket = io(socketUrl, {
          transports: ['websocket', 'polling'], // Try WebSocket first, fall back to polling
          reconnectionAttempts: 10,             // Increased from 8
          reconnectionDelay: 1000,
          timeout: this.CONNECTION_TIMEOUT,      // Use class property instead of hardcoded value
          autoConnect: true,
          forceNew: false,
          withCredentials: true,                 // Important for CORS and authentication
          extraHeaders: {                        // Add explicit CORS headers
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true"
          }
        });
        
        // Reset event tracking on new connection
        this.activeListeners.clear();
        this.registeredEvents.clear();
        this.handlerRegistry.clear();
        
        // Add debug event listeners
        this.socket.on('connect_error', (error) => {
          console.error('Socket connection error:', error.message);
          this.isConnecting = false;
          // Only reject if not already connected
          if (!this.isConnected) {
            reject(error);
          }
        });
        
        this.socket.on('error', (error) => {
          console.error('Socket error:', error);
        });
        
        // Set up connection events
        this.socket.on('connect', () => {
          console.log('Socket connected with ID:', this.socket.id);
          this.isConnected = true;
          this.isConnecting = false;
          
          // Immediately register tab ID if we have one
          if (this.tabId) {
            this.registerTab(this.tabId);
          } else {
            // Generate a random tab ID if none exists
            this.tabId = 'tab_' + Math.random().toString(36).substring(2, 10);
            this.registerTab(this.tabId);
          }
          
          resolve(this.socket);
        });
        
        this.socket.on('disconnect', (reason) => {
          console.log(`Socket disconnected: ${reason}`);
          this.isConnected = false;
          
          // Attempt reconnection if not client-initiated
          if (reason !== 'io client disconnect') {
            console.log('Attempting reconnection...');
            setTimeout(() => {
              if (!this.isConnected) {
                this.reconnect();
              }
            }, 2000);
          }
        });
        
        // Set a connection timeout
        const connectionTimeout = setTimeout(() => {
          if (!this.isConnected) {
            this.isConnecting = false;
            console.error('Socket connection timeout after', this.CONNECTION_TIMEOUT, 'ms');
            reject(new Error('Socket connection timeout'));
          }
        }, this.CONNECTION_TIMEOUT);
        
        // Clear timeout on successful connection
        this.socket.on('connect', () => {
          clearTimeout(connectionTimeout);
        });
        
        // Set up transcription event listeners
        this.setupTranscriptionEventHandlers();
        
      } catch (error) {
        console.error('Socket initialization error:', error);
        this.isConnecting = false;
        reject(error);
      }
    });
    
    return this.connectionPromise;
  }
  
  // Set up transcription event handlers
  setupTranscriptionEventHandlers() {
    if (!this.socket) return;
    
    const transcriptionEvents = [
      'transcription_started',
      'transcription_progress',
      'transcription_completed',
      'transcription_error',
      'transcription_data',
      'transcription_result'
    ];
    
    // Log all transcription events for debugging
    transcriptionEvents.forEach(event => {
      // First remove any existing handlers to prevent duplicates
      this.socket.off(event);
      
      // Then add new handler
      this.socket.on(event, (data) => {
        console.log(`Received ${event} event:`, data);
        
        // If this is a completion or error event, clean up the active request
        if (event.includes('completed') || event.includes('error') || event.includes('result') || event.includes('data')) {
          if (data && data.video_id) {
            this.activeTranscriptionRequests.delete(data.video_id);
          }
        }
      });
      
      // Track this event in our registry
      this.registeredEvents.add(event);
    });
  }
  
  // Reconnect to the socket server
  async reconnect() {
    // Clean up existing socket if needed
    if (this.socket) {
      this.safeCleanup();
    }
    
    // Reset connection state
    this.isConnected = false;
    this.isConnecting = false;
    this.connectionPromise = null;
    
    // Create a new connection
    try {
      await this.connect();
      
      // Re-register tab if we have one
      if (this.tabId) {
        this.registerTab(this.tabId);
      }
      
      return this.socket;
    } catch (error) {
      console.error('Socket reconnection failed:', error);
      throw error;
    }
  }
  
  // Register a tab ID with the server
  registerTab(tabId) {
    if (!this.isSocketConnected()) {
      console.error('Cannot register tab: socket not connected');
      return false;
    }
    
    try {
      // Store the tab ID
      this.tabId = tabId;
      this.registeredTab = tabId;
      
      console.log('Registering tab with ID:', tabId);
      this.socket.emit('register_tab', { tab_id: tabId });
      
      // Also register as client with slight delay
      setTimeout(() => {
        if (this.isSocketConnected()) {
          console.log('Registering client with ID:', tabId);
          this.socket.emit('register_client', { 
            tab_id: tabId, 
            client_id: tabId 
          });
        }
      }, 1000);
      
      return true;
    } catch (error) {
      console.error('Error registering tab:', error);
      return false;
    }
  }
  
  // Generate a unique handler key
  getHandlerKey(event, handler) {
    // Use the handler's toString() as a key
    return `${event}_${handler.toString().substring(0, 100)}`;
  }
  
  // Check if handler is already registered for an event
  isHandlerRegistered(event, handler) {
    const key = this.getHandlerKey(event, handler);
    return this.handlerRegistry.has(key);
  }
  
  // Add an event listener with duplicate prevention
  on(event, handler) {
    if (!this.socket) {
      console.error('Cannot add listener: socket not initialized');
      return false;
    }
    
    try {
      // Check if this exact handler is already registered
      if (this.isHandlerRegistered(event, handler)) {
        console.log(`Handler for event ${event} already registered, skipping duplicate`);
        return true;
      }
      
      // Instead of removing ALL listeners, just add the new one
      this.socket.on(event, handler);
      
      // Track the handler with a unique key
      const handlerKey = this.getHandlerKey(event, handler);
      this.handlerRegistry.set(handlerKey, true);
      
      // Rest of your tracking code...
      
      // Track the event and handler for later cleanup
      if (!this.activeListeners.has(event)) {
        this.activeListeners.set(event, []);
      }
      this.activeListeners.get(event).push(handler);
      this.registeredEvents.add(event);
      
      console.log(`Successfully registered event handler for: ${event}`);
      return true;
    } catch (error) {
      console.error(`Error adding listener for ${event}:`, error);
      return false;
    }
  }
  
  // Remove a specific event listener
  off(event, handler) {
    if (!this.socket) {
      console.error('Cannot remove listener: socket not initialized');
      return false;
    }
    
    try {
      // Remove the specific handler if provided
      if (handler) {
        this.socket.off(event, handler);
        
        // Remove from handler registry
        const handlerKey = this.getHandlerKey(event, handler);
        this.handlerRegistry.delete(handlerKey);
        
        // Update tracking
        if (this.activeListeners.has(event)) {
          const handlers = this.activeListeners.get(event);
          const index = handlers.indexOf(handler);
          if (index !== -1) {
            handlers.splice(index, 1);
          }
          // Clean up empty arrays
          if (handlers.length === 0) {
            this.activeListeners.delete(event);
            this.registeredEvents.delete(event);
          }
        }
      } else {
        // Remove all handlers for this event
        this.socket.off(event);
        
        // Remove from handler registry - all handlers for this event
        Array.from(this.handlerRegistry.keys()).forEach(key => {
          if (key.startsWith(`${event}_`)) {
            this.handlerRegistry.delete(key);
          }
        });
        
        this.activeListeners.delete(event);
        this.registeredEvents.delete(event);
      }
      
      return true;
    } catch (error) {
      console.error(`Error removing listener for ${event}:`, error);
      return false;
    }
  }
  
  // Clean up all event listeners safely and thoroughly
  safeCleanup() {
    if (!this.socket) return;
    
    console.log('AGGRESSIVE CLEANUP: Removing all socket event listeners');
    
    try {
      // Get all registered event names
      const eventNames = this.socket.eventNames?.() || [];
      console.log("Current events to clean:", eventNames);
      
      // Remove all listeners for all events except connection events
      eventNames.forEach(event => {
        if (event !== 'connect' && event !== 'disconnect') {
          console.log(`Removing all listeners for: ${event}`);
          this.socket.removeAllListeners(event);
        }
      });
      
      // Also explicitly remove known critical events
      const criticalEvents = [
        'ai_response', 'transcription', 'transcription_status',
        'response_complete', 'error', 'timestamps_data',
        'visual_analysis_status', 'topics_data', 'highlights_data',
        'connection_established', 'upload_transcription_complete'
      ];
      
      criticalEvents.forEach(event => {
        console.log(`Explicitly removing listeners for critical event: ${event}`);
        this.socket.removeAllListeners(event);
      });
      
      // Reset internal tracking
      this.activeListeners.clear();
      this.registeredEvents.clear();
      
      console.log('Socket event listener cleanup complete');
    } catch (error) {
      console.error('Error during aggressive socket cleanup:', error);
    }
  }
  
  // Emit an event to the server with error handling
  emit(event, data = {}) {
    if (!this.isSocketConnected()) {
      console.error(`Cannot emit event ${event}: socket not connected`);
      return false;
    }
    
    try {
      // Add tab_id to all outgoing events if available
      if (this.tabId && !data.tab_id) {
        data.tab_id = this.tabId;
      }
      
      // Add timestamp to help with debugging
      data.timestamp = Date.now();
      
      this.socket.emit(event, data);
      return true;
    } catch (error) {
      console.error(`Error emitting event ${event}:`, error);
      return false;
    }
  }
  
  // Send an echo message (for testing connection)
  sendEcho(message) {
    return this.emit('echo', message);
  }
  
  // Process a YouTube URL
  processYoutubeUrl(url, tabId = null) {
    // Validate URL format
    if (!url || !url.includes('youtube.com') && !url.includes('youtu.be')) {
      console.error('Invalid YouTube URL format');
      return { success: false, videoId: null };
    }
    
    const data = {
      url: url,
      youtube_url: url,
      tab_id: tabId || this.tabId
    };
    
    // Generate a unique video ID
    const timestamp = Math.floor(Date.now() / 1000);
    const randomId = Math.random().toString(36).substring(2, 8);
    const videoId = `youtube_${timestamp}_${randomId}`;
    
    // Add video ID to data
    data.video_id = videoId;
    
    // Send the request - ONLY use the primary event name
    const success = this.emit('analyze_youtube', data);
    
    // Explicitly request transcription for this video ID
    setTimeout(() => {
      this.requestTranscription(videoId);
    }, 3000); // Wait 3 seconds before requesting transcription
    
    return { success, videoId };
  }
  
  // Request transcription for a video
  requestTranscription(videoId) {
    if (!videoId) {
      console.warn('No video ID provided for transcription request');
      return false;
    }
    
    // Check if we're already processing this video
    if (this.activeTranscriptionRequests.has(videoId)) {
      console.log(`Transcription already in progress for ${videoId}`);
      return true;
    }
    
    if (!this.isSocketConnected()) {
      console.warn('Socket not connected for transcription request');
      return false;
    }
    
    console.log(`Requesting transcription for video ID: ${videoId}`);
    
    // Track this request
    this.activeTranscriptionRequests.set(videoId, {
      startTime: Date.now(),
      status: 'requested'
    });
    this.lastTranscriptionRequestTime = Date.now();
    
    // MODIFIED: Use only the primary event instead of multiple events
    this.emit('request_transcription', {
      video_id: videoId,
      file_id: videoId,
      tab_id: this.tabId,
      include_timestamps: true,
      full_transcription: true
    });
    
    // Also try the REST API approach in parallel
    this.fetchTranscriptionREST(videoId);
    
    return true;
  }
  
  // Get transcription with promise-based approach
  async getVideoTranscription(videoId) {
    if (!videoId) {
      return { transcript: null, timestamps: [] };
    }
    
    // Check if we need to request the transcription first
    if (!this.activeTranscriptionRequests.has(videoId)) {
      this.requestTranscription(videoId);
    }
    
    console.log(`Fetching transcription for video ID: ${videoId}`);
    
    return new Promise((resolve) => {
      // Set a timeout to guarantee the promise resolves
      const timeoutId = setTimeout(() => {
        console.warn('Timeout getting video transcription');
        resolve({ transcript: null, timestamps: [] });
      }, this.TRANSCRIPTION_TIMEOUT);
      
      // Create handler function for multiple response events
      const onTranscription = (data) => {
        // Process the response
        if (data && data.video_id === videoId && 
            (data.transcript || data.transcription || data.text)) {
          console.log(`Received transcription for video ${videoId}`);
          
          // Extract transcript from various possible properties
          const transcript = data.transcript || data.transcription || data.text || null;
          
          // Extract timestamps if available
          const timestamps = data.timestamps || [];
          
          resolve({
            status: 'success',
            transcript: transcript,
            timestamps: timestamps
          });
          
          // Clear timeout
          clearTimeout(timeoutId);
        }
      };
      
      // Register handlers for multiple response event names
      const responseEvents = [
        'transcription_data',
        'transcription_completed',
        'transcription_result',
        'transcript_data',
        'transcript_result'
      ];
      
      responseEvents.forEach(eventName => {
        // Use once to prevent multiple triggers
        this.socket.once(eventName, onTranscription);
        
        // Clean up handler when timeout occurs
        setTimeout(() => {
          if (this.isSocketConnected()) {
            this.socket.off(eventName, onTranscription);
          }
        }, this.TRANSCRIPTION_TIMEOUT);
      });
    });
  }
  
  // Check transcription status
  async checkTranscriptionStatus(videoId) {
    if (!videoId) {
      return { status: 'unknown' };
    }
    
    if (!this.isSocketConnected()) {
      return { status: 'disconnected' };
    }
    
    console.log(`Checking transcription status for video ID: ${videoId}`);
    
    return new Promise((resolve) => {
      // Set a timeout to guarantee the promise resolves
      const timeoutId = setTimeout(() => {
        console.warn('Timeout checking transcription status');
        resolve({ status: 'timeout' });
      }, this.DEFAULT_TIMEOUT);
      
      // Create handler function for multiple response events
      const onStatus = (data) => {
        // Process the response
        if (data && data.video_id === videoId) {
          console.log(`Received transcription status for video ${videoId}:`, data.status);
          resolve({
            status: data.status || 'unknown',
            progress: data.progress || 0,
            message: data.message || ''
          });
          
          // Clear timeout
          clearTimeout(timeoutId);
        }
      };
      
      // Register handlers for multiple response event names
      const responseEvents = [
        'transcription_status',
        'transcription_progress',
        'status_update'
      ];
      
      responseEvents.forEach(eventName => {
        // Use once to prevent multiple triggers
        this.socket.once(eventName, onStatus);
        
        // Clean up handler when timeout occurs
        setTimeout(() => {
          if (this.isSocketConnected()) {
            this.socket.off(eventName, onStatus);
          }
        }, this.DEFAULT_TIMEOUT);
      });
      
      // MODIFIED: Use only one status request event
      this.emit('get_transcription_status', {
        video_id: videoId,
        tab_id: this.tabId
      });
    });
  }
  
  // Fetch transcription via REST API
  async fetchTranscriptionREST(videoId) {
    if (!videoId) {
      console.warn('No video ID provided for transcription REST fetch');
      return null;
    }
    
    console.log(`Attempting REST API call for transcription of video: ${videoId}`);
    
    try {
      // Try multiple endpoint formats
      const endpoints = [
        `/api/v1/transcription/${videoId}`,
        `/api/transcription/${videoId}`,
        `/api/v1/videos/${videoId}/transcription`,
        `/api/videos/${videoId}/transcript`,
        `/api/v1/videos/${videoId}/transcript-data`,
        `/api/transcribe?videoId=${videoId}`
      ];
      
      // We'll try each endpoint with a delay between them
      for (const endpoint of endpoints) {
        try {
          const url = `${this.apiUrl}${endpoint}`;
          console.log(`Trying API endpoint: ${url}`);
          
          const response = await fetch(url, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json'
            },
            credentials: 'include' // Include cookies for authentication
          });
          
          if (response.ok) {
            const data = await response.json();
            console.log(`Successfully fetched transcription from ${endpoint}`);
            return data;
          }
        } catch (endpointError) {
          console.warn(`Error with endpoint ${endpoint}:`, endpointError);
          // Continue to next endpoint
        }
      }
      
      // If all endpoints failed, try a POST request to initiate transcription
      try {
        const url = `${this.apiUrl}/api/transcribe`;
        console.log(`Trying POST request to: ${url}`);
        
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({
            video_id: videoId,
            tab_id: this.tabId
          }),
          credentials: 'include' // Include cookies for authentication
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log(`Successfully initiated transcription via POST`);
          return data;
        }
      } catch (postError) {
        console.warn(`Error with POST request:`, postError);
      }
      
      // If all endpoints failed, throw error
      throw new Error('All transcription endpoints failed');
    } catch (error) {
      console.error('Failed to fetch transcription via REST:', error);
      return null;
    }
  }
  
  // MODIFIED: Track requests to prevent duplicate responses
  generateRequestId(type, data) {
    // Create a unique ID based on the request type and data
    let requestId = `${type}_${data.video_id || 'unknown'}_${Date.now()}`;
    if (data.question) {
      // For questions, include a hash of the question
      requestId += `_${data.question.substring(0, 20).replace(/\s+/g, '_')}`;
    }
    return requestId;
  }
  
  // Ask AI a question about the video - MODIFIED to prevent duplicates
  askAIWithContext(tabId, question, videoId, conversations = []) {
    // Format conversations for the server
    const formattedConversations = conversations.map(c => ({
      role: c.role === 'ai' ? 'assistant' : c.role,
      content: c.content
    }));
    
    const data = {
      tab_id: tabId,
      video_id: videoId,
      question: question,
      history: formattedConversations
    };
    
    // Generate a unique request ID
    const requestId = this.generateRequestId('ai_question', data);
    
    // Register this request ID to track responses
    this.pendingRequests.set(requestId, {
      timestamp: Date.now(),
      data: data,
      responseReceived: false
    });
    
    // MODIFIED: Only emit the primary event
    console.log(`Sending AI question with request ID: ${requestId}`);
    this.emit('ask_ai', data);
    
    // Set up a handler for AI responses to prevent duplicates
    this.setupAIResponseHandler(requestId, videoId, question);
    
    return true;
  }
  
  // NEW: Add a single handler for AI responses to deduplicate
  setupAIResponseHandler(requestId, videoId, question) {
    if (!this.socket) return;
    
    // Response events that may contain AI answers
    const aiResponseEvents = [
      'ai_response',
      'question_response',
      'response_complete',
      'answer'
    ];
    
    // Create a response handler
    const responseHandler = (data) => {
      // Check if this is related to our question
      if (!data || !data.video_id || data.video_id !== videoId) {
        return; // Not our response
      }
      
      // For additional validation, check if question matches
      if (data.question && question && data.question !== question) {
        return; // Not a match for our specific question
      }
      
      // Check if we've already processed a response for this request
      const responseKey = `${videoId}_${question || ''}`;
      if (this.processedResponses.has(responseKey)) {
        console.log(`Ignoring duplicate AI response for: ${responseKey}`);
        return;
      }
      
      // Mark this response as processed
      this.processedResponses.add(responseKey);
      console.log(`Processing AI response for: ${responseKey}`);
      
      // Clean up - remove all handlers for these events
      aiResponseEvents.forEach(event => {
        this.socket.off(event, responseHandler);
      });
      
      // Update request tracking
      const request = this.pendingRequests.get(requestId);
      if (request) {
        request.responseReceived = true;
        request.responseTimestamp = Date.now();
      }
      
      // Allow the original event to propagate naturally
      // (we're not modifying the event flow, just preventing duplicates)
    };
    
    // Add the handler to all possible response events
    aiResponseEvents.forEach(event => {
      this.socket.on(event, responseHandler);
    });
    
    // Clean up these handlers after a timeout
    setTimeout(() => {
      aiResponseEvents.forEach(event => {
        this.socket.off(event, responseHandler);
      });
      
      // Clean up tracking
      this.pendingRequests.delete(requestId);
      
      // After a delay, clean up the processed response tracking to prevent memory leaks
      setTimeout(() => {
        const responseKey = `${videoId}_${question || ''}`;
        this.processedResponses.delete(responseKey);
      }, 60000); // Clean up after 1 minute
      
    }, this.DEFAULT_TIMEOUT);
  }
  
  // Ask a question about visual content - MODIFIED to prevent duplicates
  askVisualQuestion(videoId, question) {
    const data = {
      video_id: videoId,
      question: question,
      tab_id: this.tabId,
      visual: true
    };
    
    // Generate a unique request ID
    const requestId = this.generateRequestId('visual_question', data);
    
    // Register this request ID to track responses
    this.pendingRequests.set(requestId, {
      timestamp: Date.now(),
      data: data,
      responseReceived: false
    });
    
    // MODIFIED: Only emit one event
    console.log(`Sending visual question with request ID: ${requestId}`);
    this.emit('ask_visual', data);
    
    // Set up a handler for AI responses to prevent duplicates
    this.setupAIResponseHandler(requestId, videoId, question);
    
    return true;
  }
  
  // Get timestamps for a video with improved error handling and multiple strategies
  async getVideoTimestamps(videoId) {
    if (!videoId) {
      console.warn('No video ID provided for timestamp fetch');
      return { timestamps: [] };
    }
    
    // Check if there's already an active fetch or if we're on cooldown
    const now = Date.now();
    const TIMESTAMP_FETCH_COOLDOWN = 1000; // Reduced cooldown to 1 second
    
    if (this.activeTimestampFetch || (now - this.lastTimestampFetchTime < TIMESTAMP_FETCH_COOLDOWN)) {
      console.log(`Skipping timestamp fetch - already active or on cooldown`);
      return { timestamps: [] };
    }
    
    // Mark as active
    this.activeTimestampFetch = true;
    this.lastTimestampFetchTime = now;
    
    try {
      if (!this.isSocketConnected()) {
        console.warn('Socket not connected for timestamp fetch');
        this.activeTimestampFetch = false;
        return { timestamps: [] };
      }
      
      console.log(`Fetching timestamps for video ID: ${videoId}`);
      
      return new Promise((resolve) => {
        // Set a timeout to guarantee the promise resolves - increased timeout
        const timeoutId = setTimeout(() => {
          this.activeTimestampFetch = false;
          console.warn('Timeout getting timestamps, returning empty array');
          resolve({ timestamps: [] });
        }, this.TIMESTAMP_TIMEOUT); // Using longer timeout for timestamps
        
        // Create one-time handler function
        const onTimestamps = (data) => {
          // Clean up
          clearTimeout(timeoutId);
          // Remove all handlers for these events to prevent duplication
          responseEvents.forEach(eventName => {
            this.socket.off(eventName, onTimestamps);
          });
          this.activeTimestampFetch = false;
          
          // Check if this data is for our video
          if (data && data.video_id === videoId && Array.isArray(data.timestamps)) {
            console.log(`Received ${data.timestamps.length} timestamps for video ${videoId}`);
            resolve({
              status: 'success',
              timestamps: data.timestamps
            });
          } else {
            // Data didn't match our request
            console.log('Received timestamps data but not for our video, ignoring');
            resolve({ timestamps: [] });
          }
        };
        
        // Register handlers for ALL possible response event names
        const responseEvents = ['timestamps_data', 'video_timestamps', 'timestamp_data', 'transcript_timestamps'];
        
        responseEvents.forEach(eventName => {
          // First remove any existing handlers
          this.socket.off(eventName);
          // Then use once to prevent multiple triggers
          this.socket.once(eventName, onTimestamps);
        });
        
        // MODIFIED: Use only one timestamp request event
        this.socket.emit('get_timestamps', {
          video_id: videoId,
          tab_id: this.tabId,
          format: 'array', // Request array format explicitly
          include_text: true // Request text with timestamps
        });
      });
    } catch (error) {
      console.error('Error fetching timestamps:', error);
      this.activeTimestampFetch = false;
      return { timestamps: [] };
    } finally {
      // Ensure flag is reset in case of errors
      setTimeout(() => {
        this.activeTimestampFetch = false;
      }, 3000);
    }
  }
  
  // Get video scenes (visual analysis results)
  getVideoScenes(videoId) {
    if (!videoId) {
      console.warn('No video ID provided for scene fetch');
      return Promise.resolve({ scenes: [] });
    }
    
    console.log(`Fetching scenes for video ID: ${videoId}`);
    
    return new Promise((resolve) => {
      // Set a timeout to guarantee the promise resolves
      const timeoutId = setTimeout(() => {
        console.warn('Timeout getting scenes, returning empty array');
        resolve({ scenes: [] });
      }, this.DEFAULT_TIMEOUT);
      
      // Create one-time handler function
      const onScenes = (data) => {
        // Clean up
        clearTimeout(timeoutId);
        // Remove all handlers for these events
        responseEvents.forEach(eventName => {
          this.socket.off(eventName, onScenes);
        });
        
        // Check if this data is for our video
        if (data && data.video_id === videoId) {
          console.log(`Received scenes data for video ${videoId}`);
          resolve({
            status: 'success',
            scenes: data.scenes || [],
            data: data
          });
        } else {
          // Data didn't match our request
          console.log('Received scenes data but not for our video, ignoring');
          resolve({ scenes: [] });
        }
      };
      
      // Register handlers for possible response event names
      const responseEvents = [
        'scenes_data', 
        'visual_analysis_data', 
        'video_scenes'
      ];
      
      responseEvents.forEach(eventName => {
        // First remove any existing handlers
        this.socket.off(eventName);
        // Then use once to prevent multiple triggers
        this.socket.once(eventName, onScenes);
      });
      
      // MODIFIED: Use only one scene request event
      this.emit('get_video_scenes', {
        video_id: videoId,
        tab_id: this.tabId
      });
    });
  }
  
  // Request visual analysis for a video
  requestVisualAnalysis(videoId) {
    if (!videoId) {
      console.warn('No video ID provided for visual analysis');
      return Promise.resolve(false);
    }
    
    console.log(`Requesting visual analysis for video ID: ${videoId}`);
    
    const data = {
      video_id: videoId,
      tab_id: this.tabId,
      analyze: true,
      extract_frames: true
    };
    
    // MODIFIED: Use only one request method
    this.emit('request_visual_analysis', data);
    
    // Also try the REST API approach in parallel
    this.requestVisualAnalysisREST(videoId);
    
    return Promise.resolve(true);
  }
  
  // Request visual analysis via REST API
  async requestVisualAnalysisREST(videoId) {
    if (!videoId) {
      console.warn('No video ID provided for visual analysis REST request');
      return false;
    }
    
    console.log(`Attempting REST API call for visual analysis of video: ${videoId}`);
    
    try {
      // Try multiple endpoint formats
      const endpoints = [
        `/api/v1/videos/${videoId}/analyze-visual`,
        `/api/visual-analysis/${videoId}`,
        `/api/v1/visual-analysis/${videoId}`,
        `/api/analyze-visual?videoId=${videoId}`,
        `/api/v1/visual-analysis?video_id=${videoId}`,
        `/api/v1/videos/${videoId}/visual`
      ];
      
      // We'll try each endpoint with a delay between them
      for (const endpoint of endpoints) {
        try {
          const url = `${this.apiUrl}${endpoint}`;
          console.log(`Trying API endpoint: ${url}`);
          
          const response = await fetch(url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
              'Access-Control-Allow-Origin': '*',
              'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
              'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            body: JSON.stringify({
              video_id: videoId,
              tab_id: this.tabId
            }),
            credentials: 'include', // Include cookies for authentication
            mode: 'cors' // Explicitly set CORS mode
          });
          
          if (response.ok) {
            const data = await response.json();
            console.log(`Successfully requested visual analysis from ${endpoint}`);
            return true;
          }
        } catch (endpointError) {
          console.warn(`Error with endpoint ${endpoint}:`, endpointError);
          // Continue to next endpoint
        }
      }
      
      return false;
    } catch (error) {
      console.error('Failed to request visual analysis via REST:', error);
      return false;
    }
  }
  
  // Fetch visual analysis data for a video
  async getVisualAnalysisData(videoId) {
    if (!videoId) {
      console.warn('No video ID provided for visual data fetch');
      return { data: null };
    }
    
    console.log(`Fetching visual analysis data for video ID: ${videoId}`);
    
    try {
      // Try REST API first
      const restData = await this.fetchVisualDataREST(videoId);
      if (restData) {
        return { status: 'success', data: restData };
      }
      
      // If REST fails, try socket
      return new Promise((resolve) => {
        // Set a timeout to guarantee the promise resolves
        const timeoutId = setTimeout(() => {
          console.warn('Timeout getting visual data, resolving with null');
          resolve({ data: null });
        }, this.DEFAULT_TIMEOUT);
        
        // Create one-time handler function
        const onVisualData = (data) => {
          // Clean up
          clearTimeout(timeoutId);
          responseEvents.forEach(eventName => {
            this.socket.off(eventName, onVisualData);
          });
          
          // Check if this data is for our video
          if (data && data.video_id === videoId) {
            console.log(`Received visual data for video ${videoId}`);
            resolve({
              status: 'success',
              data: data
            });
          } else {
            // Data didn't match our request
            console.log('Received visual data but not for our video, ignoring');
            resolve({ data: null });
          }
        };
        
        // Register handlers for possible response event names
        const responseEvents = [
          'visual_data',
          'visual_analysis_data',
          'video_visual_data'
        ];
        
        responseEvents.forEach(eventName => {
          // First remove any existing handlers
          this.socket.off(eventName);
          // Then use once to prevent multiple triggers
          this.socket.once(eventName, onVisualData);
        });
        
        // MODIFIED: Use only one visual data request event
        this.emit('get_visual_data', {
          video_id: videoId,
          tab_id: this.tabId
        });
      });
    } catch (error) {
      console.error('Error getting visual analysis data:', error);
      return { data: null };
    }
  }
  
  // Fetch visual data via REST API
  async fetchVisualDataREST(videoId) {
    if (!videoId) {
      console.warn('No video ID provided for visual data REST fetch');
      return null;
    }
    
    try {
      // Try multiple endpoint formats
      const endpoints = [
        `/api/v1/videos/${videoId}/visual-data`,
        `/api/videos/${videoId}/visual`,
        `/api/analyze-visual?videoId=${videoId}`,
        `/api/v1/visual-analysis?video_id=${videoId}`,
        `/api/v1/videos/${videoId}/visual`,
        `/api/v1/visual-analysis/${videoId}`,
        `/api/visual-analysis/${videoId}`
      ];
      
      // We'll try each endpoint with a delay between them
      for (const endpoint of endpoints) {
        try {
          const url = `${this.apiUrl}${endpoint}`;
          console.log(`Trying visual data endpoint: ${url}`);
          
          const response = await fetch(url, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json'
            },
            credentials: 'include' // Include cookies for authentication
          });
          
          if (response.ok) {
            const data = await response.json();
            console.log(`Successfully fetched visual data from ${endpoint}`);
            return data;
          }
        } catch (endpointError) {
          // Ignore 404 errors for visual requests
          if (endpointError.response && endpointError.response.status === 404) {
            console.info(`Ignoring 404 error for visual request: ${endpoint}`);
          } else {
            console.warn(`Error with endpoint ${endpoint}:`, endpointError);
          }
          // Continue to next endpoint
        }
      }
      
      return null;
    } catch (error) {
      console.error('Failed to fetch visual data via REST:', error);
      return null;
    }
  }
  
  // Get topics for a video
  async getVideoTopics(videoId) {
    if (!videoId) {
      console.warn('No video ID provided for topics fetch');
      return { topics: [] };
    }
    
    if (!this.topicsSupported) {
      console.warn('Topics feature not supported');
      return { topics: [] };
    }
    
    console.log(`Fetching topics for video ID: ${videoId}`);
    
    return new Promise((resolve) => {
      // Set a timeout to guarantee the promise resolves
      const timeoutId = setTimeout(() => {
        console.warn('Timeout getting topics, returning empty array');
        resolve({ topics: [] });
      }, this.DEFAULT_TIMEOUT);
      
      // Create one-time handler function
      const onTopics = (data) => {
        // Clean up
        clearTimeout(timeoutId);
        responseEvents.forEach(eventName => {
          this.socket.off(eventName, onTopics);
        });
        
        // Check if this data is for our video
        if (data && data.video_id === videoId) {
          console.log(`Received topics data for video ${videoId}`);
          resolve({
            status: 'success',
            topics: data.topics || []
          });
        } else {
          // Data didn't match our request
          console.log('Received topics data but not for our video, ignoring');
          resolve({ topics: [] });
        }
      };
      
      // Register handlers for possible response event names
      const responseEvents = ['topics_data', 'video_topics'];
      
      responseEvents.forEach(eventName => {
        // First remove any existing handlers
        this.socket.off(eventName);
        // Then use once to prevent multiple triggers
        this.socket.once(eventName, onTopics);
      });
      
      // MODIFIED: Use only one topic request event
      this.emit('get_topics', {
        video_id: videoId,
        tab_id: this.tabId
      });
    });
  }
  
  // Get highlights for a video
  async getVideoHighlights(videoId) {
    if (!videoId) {
      console.warn('No video ID provided for highlights fetch');
      return { highlights: [] };
    }
    
    if (!this.highlightsSupported) {
      console.warn('Highlights feature not supported');
      return { highlights: [] };
    }
    
    console.log(`Fetching highlights for video ID: ${videoId}`);
    
    return new Promise((resolve) => {
      // Set a timeout to guarantee the promise resolves
      const timeoutId = setTimeout(() => {
        console.warn('Timeout getting highlights, returning empty array');
        resolve({ highlights: [] });
      }, this.DEFAULT_TIMEOUT);
      
      // Create one-time handler function
      const onHighlights = (data) => {
        // Clean up
        clearTimeout(timeoutId);
        responseEvents.forEach(eventName => {
          this.socket.off(eventName, onHighlights);
        });
        
        // Check if this data is for our video
        if (data && data.video_id === videoId) {
          console.log(`Received highlights data for video ${videoId}`);
          resolve({
            status: 'success',
            highlights: data.highlights || []
          });
        } else {
          // Data didn't match our request
          console.log('Received highlights data but not for our video, ignoring');
          resolve({ highlights: [] });
        }
      };
      
      // Register handlers for possible response event names
      const responseEvents = ['highlights_data', 'video_highlights'];
      
      responseEvents.forEach(eventName => {
        // First remove any existing handlers
        this.socket.off(eventName);
        // Then use once to prevent multiple triggers
        this.socket.once(eventName, onHighlights);
      });
      
      // MODIFIED: Use only one highlight request event
      this.emit('get_highlights', {
        video_id: videoId,
        tab_id: this.tabId
      });
    });
  }
  
  // Get video frames
  getVideoFrames(videoId) {
    if (!videoId) {
      console.warn('No video ID provided for frames fetch');
      return Promise.resolve({ frames: [] });
    }
    
    console.log(`Fetching frames for video ID: ${videoId}`);
    
    return new Promise((resolve) => {
      // Set a timeout to guarantee the promise resolves
      const timeoutId = setTimeout(() => {
        console.warn('Timeout getting frames, returning empty array');
        resolve({ frames: [] });
      }, this.DEFAULT_TIMEOUT);
      
      // Create one-time handler function
      const onFrames = (data) => {
        // Clean up
        clearTimeout(timeoutId);
        responseEvents.forEach(eventName => {
          this.socket.off(eventName, onFrames);
        });
        
        // Check if this data is for our video
        if (data && data.video_id === videoId) {
          console.log(`Received frames data for video ${videoId}`);
          resolve({
            status: 'success',
            frames: data.frames || [],
            data: data
          });
        } else {
          // Data didn't match our request
          console.log('Received frames data but not for our video, ignoring');
          resolve({ frames: [] });
        }
      };
      
      // Register handlers for possible response event names
      const responseEvents = [
        'frames_data', 
        'video_frames', 
        'visual_frames'
      ];
      
      responseEvents.forEach(eventName => {
        // First remove any existing handlers
        this.socket.off(eventName);
        // Then use once to prevent multiple triggers
        this.socket.once(eventName, onFrames);
      });
      
      // MODIFIED: Use only one frame request event
      this.emit('get_video_frames', {
        video_id: videoId,
        tab_id: this.tabId
      });
    });
  }
  
  // Get video summary
  async getVideoSummary(videoId) {
    if (!videoId) {
      console.warn('No video ID provided for summary fetch');
      return { summary: null };
    }
    
    console.log(`Fetching summary for video ID: ${videoId}`);
    
    return new Promise((resolve) => {
      // Set a timeout to guarantee the promise resolves
      const timeoutId = setTimeout(() => {
        console.warn('Timeout getting summary, returning null');
        resolve({ summary: null });
      }, this.DEFAULT_TIMEOUT);
      
      // Create one-time handler function
      const onSummary = (data) => {
        // Clean up
        clearTimeout(timeoutId);
        responseEvents.forEach(eventName => {
          this.socket.off(eventName, onSummary);
        });
        
        // Check if this data is for our video
        if (data && data.video_id === videoId) {
          console.log(`Received summary data for video ${videoId}`);
          resolve({
            status: 'success',
            summary: data.summary || null
          });
        } else {
          // Data didn't match our request
          console.log('Received summary data but not for our video, ignoring');
          resolve({ summary: null });
        }
      };
      
      // Register handlers for possible response event names
      const responseEvents = ['summary_data', 'video_summary'];
      
      responseEvents.forEach(eventName => {
        // First remove any existing handlers
        this.socket.off(eventName);
        // Then use once to prevent multiple triggers
        this.socket.once(eventName, onSummary);
      });
      
      // MODIFIED: Use only one summary request event
      this.emit('get_summary', {
        video_id: videoId,
        tab_id: this.tabId
      });
    });
  }
  
  // Cancel ongoing video processing
  cancelVideoProcessing(videoId) {
    if (!videoId) {
      console.warn('No video ID provided for cancellation');
      return false;
    }
    
    console.log(`Cancelling processing for video ID: ${videoId}`);
    
    const data = {
      video_id: videoId,
      tab_id: this.tabId
    };
    
    // MODIFIED: Use only one cancel event
    this.emit('cancel_processing', data);
    
    return true;
  }
}
// Create a singleton instance
const socketService = new SocketService();
export default socketService;