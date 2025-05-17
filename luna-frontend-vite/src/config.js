// src/utils/config.js

// Detect environment
const isDevelopment = import.meta.env.MODE === 'development' || 
                      import.meta.env.VITE_APP_ENV === 'development';

// API Base URL from environment or default
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Socket.IO URL from environment or default to API URL
const socketUrl = import.meta.env.VITE_SOCKET_URL || apiBaseUrl;

const config = {
  // Base API URL - maintain both naming conventions for compatibility
  apiBaseUrl,
  apiUrl: apiBaseUrl,  // Added for compatibility with our fixes
  
  // Socket.IO URL
  socketUrl,
  
  // API endpoint for direct access
  API_URL: apiBaseUrl,
  
  // Default API timeout
  apiTimeout: 30000, // 30 seconds
  
  // Authentication endpoints
  auth: {
    login: '/api/auth/login',
    signup: '/api/auth/signup',
    logout: '/api/auth/logout',
    // Set me endpoint to null to signal it's not available
    me: null,  // This signals that we don't have a user verification endpoint
    verify: '/api/auth/check-auth',
    refresh: '/api/auth/refresh',
    forgotPassword: '/api/auth/forgot-password',
    resetPassword: '/api/auth/reset-password', // + /:token
  },
  
  // Video processing endpoints - direct routes
  video: {
    processUrl: '/process-url',
    uploadVideo: '/upload-video',
    getTranscription: '/get-transcription',
  },
  
  // WebSocket message types
  ws: {
    // Communication events
    askAi: 'ask_ai',
    aiResponse: 'ai_response',
    registerTab: 'register_tab',
    error: 'error',
    echo: 'echo',
    echoResponse: 'echo_response',
    
    // Connection events
    connectionEstablished: 'connection_established',
    welcome: 'welcome',
    
    // Transcription events
    transcribeYoutube: 'transcribe_youtube_video',
    transcriptionStatus: 'transcription_status',
    transcription: 'transcription',
    fetchTranscription: 'fetch_transcription',
    
    // Processing events
    processingStatus: 'processing_status',
    processingUpdate: 'processing_update'
  },
  
  // Socket.IO configuration - preserved from your original config
  socketOptions: {
    path: '/socket.io/',
    transports: ['websocket', 'polling'],  // Try websocket first, then fall back to polling
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    timeout: 20000,
    withCredentials: true,  // For auth cookies
    autoConnect: false  // For manual connection control
  },
  
  // Debug settings
  debug: isDevelopment, // Simple boolean flag for compatibility
  debugLogs: {
    socketLogs: isDevelopment,  // Enable for debugging
    apiLogs: isDevelopment      // Enable for debugging
  },
  
  // Application settings
  app: {
    name: 'Luna AI',
    version: '1.0.0',
    description: 'AI-powered Video Analysis Tool'
  },
  
  // Feature flags
  features: {
    visualAnalysis: true,
    timestamps: true,
    audioTranscription: true,
  },
  
  // Upload limits
  upload: {
    maxVideoSizeMB: 1024,
    maxAudioSizeMB: 1024,
    supportedVideoFormats: ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
    supportedAudioFormats: ['.mp3', '.wav', '.m4a', '.ogg', '.flac'],
  }
};

// Log configuration in development mode
if (isDevelopment) {
  console.log('App Configuration:', config);
}

export default config;