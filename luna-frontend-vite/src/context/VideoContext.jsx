// src/context/VideoContext.jsx
import React, { createContext, useState, useContext, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';
import socketService from '../services/socket';
// Create the context with named export to properly support imports
export const VideoContext = createContext();
// Custom hook for using the VideoContext
export const useVideo = () => {
  const context = useContext(VideoContext);
  if (!context) {
    throw new Error('useVideo must be used within a VideoProvider');
  }
  return context;
};
// Constants
const TIMESTAMP_ERROR_THRESHOLD = 3; // Number of errors before stopping attempts
const TRANSCRIPTION_DELAY = 1000; // Delay after transcription is loaded before generating timestamps
const API_TIMEOUT = 30000; // Increased timeout for API calls (30 seconds)
const SOCKET_RECONNECT_DELAY = 2000; // Delay before socket reconnection attempts
const VISUAL_FEATURE_TIMEOUT = 45000; // 45 seconds timeout for visual features
// Provider component - named function component for better HMR compatibility
export function VideoProvider({ children }) {
  // Add useNavigate hook for programmatic navigation
  const navigate = useNavigate();
  
  // ===== STATE DEFINITIONS FIRST =====
  const [videoData, setVideoData] = useState(null);
  const [transcription, setTranscription] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tabId, setTabId] = useState(null);
  const [currentVideoId, setCurrentVideoId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [isAIThinking, setIsAIThinking] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('initializing');
  const [transcriptionStatus, setTranscriptionStatus] = useState('idle');
  const [processingYoutubeUrl, setProcessingYoutubeUrl] = useState(false);
  const [transcriptionLoading, setTranscriptionLoading] = useState(false);
  const [processingStage, setProcessingStage] = useState('idle');
  
  // Visual analysis states
  const [visualAnalysisAvailable, setVisualAnalysisAvailable] = useState(false);
  const [visualAnalysisInProgress, setVisualAnalysisInProgress] = useState(false);
  
  // New enhanced feature states
  const [videoTopics, setVideoTopics] = useState([]);
  const [videoHighlights, setVideoHighlights] = useState([]);
  const [visualSummary, setVisualSummary] = useState(null);
  const [currentView, setCurrentView] = useState('chat');
  const [selectedTimestamp, setSelectedTimestamp] = useState(null);
  const [detectedScenes, setDetectedScenes] = useState([]);
  const [keyFrames, setKeyFrames] = useState([]);
  
  // Visual content understanding states
  const [videoObjects, setVideoObjects] = useState([]);
  const [visualContext, setVisualContext] = useState(null);
  const [timestampContent, setTimestampContent] = useState({});
  const [visibleTimestamps, setVisibleTimestamps] = useState([]);
  const [sidebarTimestamps, setSidebarTimestamps] = useState([]);
  
  // Timestamp state for VideoPlayer integration
  const [timestamps, setTimestamps] = useState([]);
  const [activeTimestampIndex, setActiveTimestampIndex] = useState(-1);
  const [currentTime, setCurrentTime] = useState(0);
  
  // Timestamp fetching state
  const [timestampFetchAttempts, setTimestampFetchAttempts] = useState(0);
  const [timestampFetchFailed, setTimestampFetchFailed] = useState(false);
  const [hasGeneratedTimestamps, setHasGeneratedTimestamps] = useState(false);
  // ===== REFS DEFINITION =====
  const isMounted = useRef(true);
  const thinkingTimeoutRef = useRef(null);
  const visualDataLoadingRef = useRef(false);
  const timestampHandlerRef = useRef(null);
  const fallbackTimerRef = useRef(null);
  const heartbeatIntervalRef = useRef(null);
  const activeTranscriptionRef = useRef(null);
  const pingIntervalRef = useRef(null);
  const socketInitializedRef = useRef(false);
  const socketConnectedRef = useRef(false);
  const connectionRetryCountRef = useRef(0);
  const apiFallbackAttemptsRef = useRef(0);
  const activeApiRequestRef = useRef(null);
  const timestampGenerationAttemptsRef = useRef(0);
  const eventRegistrationRef = useRef({
    timestamps_data: false,
    timestamps_available: false,
  });
  const delayedTimestampGenRef = useRef(null);
  const authTokenRef = useRef(null); // Added ref for authentication token
  const transcriptionReceivedRef = useRef(false); // Track if we've received transcription data
  const registeredEventsRef = useRef(new Set()); // NEW: Track registered events to prevent duplicates
  // Reference to track function implementations to break circular dependencies
  const functionsRef = useRef({
    fetchSidebarTimestamps: null,
    generateTimestampsFromTranscript: null,
    setupSocketListeners: null,
    fetchTranscriptionViaREST: null,
    fetchVisualData: null,
    navigateToAnalysisPage: null
  });
  // ===== TRACK COMPONENT MOUNT STATE =====
  useEffect(() => {
    isMounted.current = true;
    console.log("VideoContext mounted");
    return () => {
      console.log("VideoContext unmounting");
      isMounted.current = false;
    };
  }, []);
  // ===== SETUP TAB ID =====
  useEffect(() => {
    try {
      // First try to get auth token
      const token = localStorage.getItem('authToken');
      if (token) {
        authTokenRef.current = token;
        console.log("Retrieved auth token from localStorage");
      }
      
      const storedTabId = localStorage.getItem('tabId');
      if (storedTabId) {
        console.log("Retrieved stored tabId:", storedTabId);
        setTabId(storedTabId);
      } else {
        const newTabId = Date.now().toString();
        console.log("Generated new tabId:", newTabId);
        localStorage.setItem('tabId', newTabId);
        setTabId(newTabId);
      }
    } catch (e) {
      console.error("Error accessing localStorage:", e);
      // Fallback to memory-only tabId
      setTabId(Date.now().toString());
    }
  }, []);
  // ===== UTILITY FUNCTIONS =====
  // Helper function to extract ID from API response for uploaded files - IMPROVED VERSION
  const extractIdFromResponse = useCallback((responseData) => {
    // Check for uploaded file ID in upload response
    if (responseData && responseData.video_id) {
      // Direct access to the server-assigned ID
      return responseData.video_id;
    }
    
    // Try to find ID in form of upload_{timestamp}_{random}
    if (typeof responseData === 'string') {
      const uploadIdRegex = /upload_\d+_[a-z0-9]+/;
      if (uploadIdRegex.test(responseData)) {
        return responseData.match(uploadIdRegex)[0];
      }
    }
    
    // Check for ID in JSON string
    if (typeof responseData === 'string') {
      try {
        const parsedData = JSON.parse(responseData);
        if (parsedData && parsedData.video_id) {
          return parsedData.video_id;
        }
      } catch (e) {
        // Not a valid JSON string, ignore
      }
    }
    
    // NEW: Additional pattern matching for different ID formats
    if (typeof responseData === 'object') {
      // Look for any property that might contain a file ID
      for (const key in responseData) {
        if (typeof responseData[key] === 'string') {
          // Look for upload_{timestamp}_{hash} pattern
          const uploadMatch = responseData[key].match(/upload_\d+_[a-z0-9]+/);
          if (uploadMatch) return uploadMatch[0];
          
          // Look for file_{timestamp} pattern
          const fileMatch = responseData[key].match(/file_\d+/);
          if (fileMatch) return fileMatch[0];
        }
      }
    }
    
    // NEW: Convert between upload_TIMESTAMP_HASH and file_TIMESTAMP formats
    if (currentVideoId) {
      if (currentVideoId.startsWith('upload_')) {
        const parts = currentVideoId.split('_');
        if (parts.length >= 2) {
          return `file_${parts[1]}`;
        }
      } else if (currentVideoId.startsWith('file_')) {
        const timestamp = currentVideoId.replace('file_', '');
        // We can't recreate the hash part, so just return what we have
        return currentVideoId;
      }
    }
    
    return null;
  }, [currentVideoId]);
  
  // Helper function to get auth headers for API calls
  const getAuthHeaders = useCallback(() => {
    const headers = {};
    // First check current ref for most up-to-date token
    if (authTokenRef.current) {
      headers['Authorization'] = `Bearer ${authTokenRef.current}`;
    } else {
      // Fall back to localStorage
      const token = localStorage.getItem('authToken');
      if (token) {
        authTokenRef.current = token; // Update ref
        headers['Authorization'] = `Bearer ${token}`;
      }
    }
    return headers;
  }, []);
  // Helper function to handle API response errors
  const handleApiError = useCallback(async (error, endpoint) => {
    console.error(`API error for ${endpoint}:`, error);
    
    // Handle auth errors (401)
    if (error.response && error.response.status === 401) {
      console.log('Authentication error, attempting token refresh');
      
      try {
        // Try multiple refresh endpoints
        const refreshEndpoints = [
          '/api/auth/refresh',
          '/api/v1/auth/refresh',
          '/api/refresh-token'
        ];
        
        for (const refreshEndpoint of refreshEndpoints) {
          try {
            const refreshResponse = await api.post(refreshEndpoint);
            if (refreshResponse.data && refreshResponse.data.token) {
              // Update token in localStorage and ref
              localStorage.setItem('authToken', refreshResponse.data.token);
              authTokenRef.current = refreshResponse.data.token;
              console.log('Successfully refreshed auth token');
              return true; // Success
            }
          } catch (refreshError) {
            console.log(`Refresh endpoint ${refreshEndpoint} failed:`, refreshError.message);
          }
        }
        
        // If we get here, all refresh attempts failed
        console.error('All token refresh attempts failed');
        return false;
      } catch (refreshError) {
        console.error('Error refreshing token:', refreshError);
        return false;
      }
    }
    
    return false; // Default to failed
  }, []);
  
  // Check server health - improved to handle 404 errors
  const checkServerHealth = useCallback(async () => {
    try {
      // Try multiple health endpoints
      const endpoints = [
        '/api/health',
        '/api/v1/health',
        '/api/ping'
      ];
      
      for (const endpoint of endpoints) {
        try {
          const response = await api.get(endpoint, { timeout: 5000 });
          if (response.status === 200) {
            return true;
          }
        } catch (endpointError) {
          // 404 may be normal if the endpoint doesn't exist
          if (endpointError.response && endpointError.response.status === 404) {
            console.log(`Health endpoint ${endpoint} not found (404), trying next`);
          } else {
            console.warn(`Health endpoint ${endpoint} check failed:`, endpointError.message);
          }
        }
      }
      
      // If all explicit health checks failed, try a simple API endpoint
      // to see if the server is responding at all
      try {
        await api.get('/api', { timeout: 5000 });
        return true; // Server responds to basic endpoint
      } catch (e) {
        console.warn('Basic API endpoint check failed');
      }
      
      return false; // All health checks failed
    } catch (error) {
      console.warn('Server health check failed:', error.message);
      return false;
    }
  }, []);
  // Helper function to check if two video IDs are related - IMPROVED VERSION
  const areVideoIdsRelated = useCallback((id1, id2) => {
    if (!id1 || !id2) return false;
    
    // Direct match
    if (id1 === id2) return true;
    
    // Both are YouTube IDs
    if (id1.startsWith('youtube_') && id2.startsWith('youtube_')) {
      // Extract timestamp parts
      const parts1 = id1.split('_');
      const parts2 = id2.split('_');
      
      if (parts1.length >= 2 && parts2.length >= 2) {
        // Check if base timestamp parts are close (within 5 seconds)
        const time1 = parseInt(parts1[1]);
        const time2 = parseInt(parts2[1]);
        
        if (!isNaN(time1) && !isNaN(time2)) {
          return Math.abs(time1 - time2) < 5;
        }
      }
    }
    
    // IMPROVED: Better handling for upload/file ID formats
    // Case 1: Both are upload IDs 
    if (id1.startsWith('upload_') && id2.startsWith('upload_')) {
      // Extract timestamps from both IDs
      const timestamp1 = id1.split('_')[1];
      const timestamp2 = id2.split('_')[1];
      
      // If timestamps are the same or very close, they're related
      if (timestamp1 && timestamp2 && Math.abs(parseInt(timestamp1) - parseInt(timestamp2)) < 5) {
        return true;
      }
      
      // Otherwise check if they share the same hash suffix
      const hash1 = id1.split('_')[2];
      const hash2 = id2.split('_')[2];
      if (hash1 && hash2 && hash1 === hash2) {
        return true;
      }
      
      return false; // Different uploads
    }
    
    // Case 2: Both are file IDs
    if (id1.startsWith('file_') && id2.startsWith('file_')) {
      // Extract timestamps from both IDs
      const timestamp1 = id1.replace('file_', '');
      const timestamp2 = id2.replace('file_', '');
      
      // If timestamps are the same or very close, they're related
      if (timestamp1 && timestamp2) {
        // Allow 10 second difference for file IDs
        return Math.abs(parseInt(timestamp1) - parseInt(timestamp2)) < 10;
      }
      return false;
    }
    
    // Case 3: Both are audio IDs
    if (id1.startsWith('audio_') && id2.startsWith('audio_')) {
      // Extract timestamps from both IDs
      const timestamp1 = id1.replace('audio_', '');
      const timestamp2 = id2.replace('audio_', '');
      
      // If timestamps are the same or very close, they're related
      if (timestamp1 && timestamp2) {
        return Math.abs(parseInt(timestamp1) - parseInt(timestamp2)) < 10;
      }
      return false;
    }
    
    // IMPROVED: More robust handling of file_TIMESTAMP vs upload_TIMESTAMP_HASH formats
    if ((id1.startsWith('file_') && id2.startsWith('upload_')) || 
        (id1.startsWith('upload_') && id2.startsWith('file_'))) {
      
      let fileId, uploadId;
      if (id1.startsWith('file_')) {
        fileId = id1;
        uploadId = id2;
      } else {
        fileId = id2;
        uploadId = id1;
      }
      
      // Extract timestamps from both IDs
      const fileTimestamp = fileId.replace('file_', '');
      const uploadTimestamp = uploadId.split('_')[1];
      
      // If timestamps are the same or very close, they're related
      if (fileTimestamp && uploadTimestamp) {
        // Allow a difference of up to 5 seconds to account for slight variations
        const diff = Math.abs(parseInt(fileTimestamp) - parseInt(uploadTimestamp));
        return diff < 5000; // More flexible matching
      }
    }
    
    // NEW: Check if one is the direct "upload_" counterpart to a "file_" ID
    if (id1.startsWith('file_') && id2.startsWith('upload_')) {
      const fileTimestamp = id1.replace('file_', '');
      // Check if the timestamps in the IDs are related
      if (id2.includes(`_${fileTimestamp.substring(0, 8)}`)) {
        return true;
      }
    } 
    else if (id1.startsWith('upload_') && id2.startsWith('file_')) {
      const fileTimestamp = id2.replace('file_', '');
      // Check if the timestamps in the IDs are related
      if (id1.includes(`_${fileTimestamp.substring(0, 8)}`)) {
        return true;
      }
    }
    
    return false;
  }, []);
  // Convert backend timestamp format to frontend format
  const convertTimestampFormat = useCallback((backendTimestamps) => {
    if (!Array.isArray(backendTimestamps)) return [];
    return backendTimestamps.map(ts => ({
      time: ts.start_time || ts.time || 0,
      time_formatted: ts.display_time || formatTimeDisplay(ts.start_time || ts.time || 0),
      text: ts.text || '',
      end_time: ts.end_time || (ts.start_time ? ts.start_time + 10 : ts.time ? ts.time + 10 : 10)
    }));
  }, []);
  // Format time in seconds to MM:SS display
  const formatTimeDisplay = useCallback((seconds) => {
    if (isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }, []);
  // ===== PRIMARY ACTIONS AND CALLBACKS =====
  // Add message to conversation
  const addMessage = useCallback((role, content) => {
    setConversations(prev => [...prev, { role, content }]);
  }, []);
  
  // Clear conversation
  const clearConversation = useCallback(() => {
    setConversations([]);
  }, []);
  // Navigate to a specific timestamp
  const navigateToTimestamp = useCallback((timestamp) => {
    console.log("Navigate to timestamp:", timestamp);
    setSelectedTimestamp(timestamp);
  }, []);
  // Navigation to analysis page after successful URL processing
  const navigateToAnalysisPage = useCallback(() => {
    console.log("Navigating to analysis page");
    navigate('/analysis', { replace: false });
  }, [navigate]);
  // Store the navigation function for later use
  functionsRef.current.navigateToAnalysisPage = navigateToAnalysisPage;
  // Seek to a specific timestamp
  const seekToTimestamp = useCallback((index) => {
    if (index >= 0 && index < timestamps.length) {
      setActiveTimestampIndex(index);
      const timestamp = timestamps[index];
      return timestamp.start_time || timestamp.time || 0;
    }
    return null;
  }, [timestamps]);
  // Update current time from video player
  const updateCurrentTime = useCallback((time) => {
    setCurrentTime(time);
    // Find the current active timestamp based on playback position
    if (timestamps.length > 0) {
      let foundIndex = -1;
      for (let i = 0; i < timestamps.length; i++) {
        const timestamp = timestamps[i];
        const start = timestamp.start_time || timestamp.time || 0;
        const end = timestamp.end_time || (start + 10); // Default 10 seconds if no end time
        if (time >= start && time <= end) {
          foundIndex = i;
          break;
        }
      }
      if (foundIndex !== activeTimestampIndex) {
        setActiveTimestampIndex(foundIndex);
      }
    }
  }, [timestamps, activeTimestampIndex]);
  // Switch view function
  const switchView = useCallback((view) => {
    console.log("Switch view to:", view);
    setCurrentView(view);
  }, []);
  // Fetch transcription via REST API - IMPROVED VERSION
  const fetchTranscriptionViaREST = useCallback(async (videoId) => {
    if (!videoId) return false;
    
    // Prevent multiple simultaneous requests
    if (activeApiRequestRef.current) {
      console.log('Already fetching transcription via REST API, skipping duplicate request');
      return false;
    }
    
    console.log(`🔄 Fetching transcription via REST API fallback for video ID: ${videoId}`);
    setProcessingStage('rest-api-fallback');
    
    try {
      // Set active request flag
      activeApiRequestRef.current = true;
      
      // Get auth headers
      const headers = getAuthHeaders();
      
      // First try the tab ID-based endpoint
      try {
        console.log(`Trying direct video ID endpoint: /api/v1/videos/${videoId}/transcription`);
        const tabIdResponse = await api.get(`/api/v1/videos/${videoId}/transcription`, {
          headers, 
          timeout: API_TIMEOUT 
        });
        
        if (tabIdResponse.data) {
          let transcriptionText = '';
          
          if (tabIdResponse.data.transcription) {
            transcriptionText = tabIdResponse.data.transcription;
          } else if (tabIdResponse.data.transcript) {
            transcriptionText = tabIdResponse.data.transcript;
          } else if (tabIdResponse.data.text) {
            transcriptionText = tabIdResponse.data.text;
          } else if (typeof tabIdResponse.data === 'string') {
            transcriptionText = tabIdResponse.data;
          }
          
          if (transcriptionText && transcriptionText.length > 0) {
            console.log(`✅ Tab ID endpoint successful, received ${transcriptionText.length} characters`);
            
            // Also update the video ID if provided
            if (tabIdResponse.data.video_id && tabIdResponse.data.video_id !== currentVideoId) {
              console.log(`Updating video ID from ${currentVideoId} to ${tabIdResponse.data.video_id}`);
              setCurrentVideoId(tabIdResponse.data.video_id);
            }
            
            if (isMounted.current) {
              // Mark that we've received transcription data
              transcriptionReceivedRef.current = true;
              
              setTranscription(transcriptionText);
              setTranscriptionStatus('loaded');
              setLoading(false);
              setTranscriptionLoading(false);
              setProcessingYoutubeUrl(false);
              setProcessingStage('transcription-complete');
              
              // Reset timestamp generation attempt counter
              timestampGenerationAttemptsRef.current = 0;
              
              // Wait a short while before generating timestamps to ensure state updates properly
              setTimeout(() => {
                if (isMounted.current) {
                  // Generate timestamps from the transcription
                  if (functionsRef.current.generateTimestampsFromTranscript) {
                    functionsRef.current.generateTimestampsFromTranscript();
                  }
                  
                  // Also fetch timestamps after successful transcription
                  if (functionsRef.current.fetchSidebarTimestamps) {
                    functionsRef.current.fetchSidebarTimestamps(videoId);
                  }
                }
              }, TRANSCRIPTION_DELAY);
            }
            
            // Clear active request flag
            activeApiRequestRef.current = false;
            return true;
          }
        }
      } catch (tabIdError) {
        // console.log(`Tab ID endpoint failed:`, tabIdError.message);
        console.log(`Direct video ID endpoint failed:`, tabIdError.message);
        // Continue to try other endpoints
      }
      
      // IMPROVED: Try more variations of endpoints including converting between formats
      // Try to derive alternative ID formats for better compatibility
      const alternativeIds = [videoId]; // Start with the original ID
      
      // If videoId is file_TIMESTAMP format, try to derive upload_TIMESTAMP format
      if (videoId.startsWith('file_')) {
        const timestamp = videoId.replace('file_', '');
        if (timestamp) {
          const uploadFormatId = `upload_${timestamp}`;
          alternativeIds.push(uploadFormatId);
          
          // Also try a truncated version of the timestamp (browsers sometimes round differently)
          const truncatedTimestamp = timestamp.substring(0, timestamp.length - 3);
          if (truncatedTimestamp && truncatedTimestamp.length > 8) {
            alternativeIds.push(`upload_${truncatedTimestamp}`);
          }
        }
      }
      // If videoId is upload_TIMESTAMP_HASH format, try to derive file_TIMESTAMP format
      else if (videoId.startsWith('upload_')) {
        const parts = videoId.split('_');
        if (parts.length >= 2) {
          const timestamp = parts[1];
          if (timestamp) {
            const fileFormatId = `file_${timestamp}`;
            alternativeIds.push(fileFormatId);
          }
        }
      }
      
      // NEW: Try specific endpoints for uploaded files with both formatIds
      // Generate endpoint patterns for every possible ID
      const endpointPatterns = [];
      alternativeIds.forEach(id => {
        // Standard endpoints
        endpointPatterns.push(
          { url: `/api/v1/videos/${id}/transcription`, method: 'get' },
          { url: `/api/videos/${id}/transcription`, method: 'get' },
          { url: `/api/transcription/${id}`, method: 'get' },
          { url: `/api/transcript/${id}`, method: 'get' }, 
          { url: `/api/v1/transcript/${id}`, method: 'get' },
          // Upload-specific endpoints 
          { url: `/api/v1/uploads/${id}/transcription`, method: 'get' },
          { url: `/api/uploads/${id}/transcription`, method: 'get' },
          { url: `/api/v1/files/${id}/transcription`, method: 'get' },
          { url: `/api/files/${id}/transcription`, method: 'get' },
          { url: `/api/v1/audio/${id}/transcription`, method: 'get' },
          { url: `/api/audio/${id}/transcription`, method: 'get' }
        );
      });
      
      // Also add some special patterns that might be supported by the server
      const baseEndpoints = [
        { url: '/api/v1/transcription', method: 'post', data: { video_id: videoId, tab_id: tabId } },
        { url: '/api/transcription', method: 'post', data: { video_id: videoId, tab_id: tabId } },
        { url: '/api/v1/transcript', method: 'post', data: { video_id: videoId, tab_id: tabId } },
        { url: '/api/transcript', method: 'post', data: { video_id: videoId, tab_id: tabId } }
      ];
      
      // Add these to our endpoint patterns
      endpointPatterns.push(...baseEndpoints);
      
      // Try each endpoint pattern
      for (const endpoint of endpointPatterns) {
        try {
          console.log(`Trying endpoint: ${endpoint.url}`);
          let response;
          
          if (endpoint.method === 'get') {
            response = await api.get(endpoint.url, { 
              headers, 
              timeout: API_TIMEOUT 
            });
          } else {
            response = await api.post(endpoint.url, endpoint.data || {}, { 
              headers, 
              timeout: API_TIMEOUT 
            });
          }
          
          // Log full response for debugging
          console.log(`Response from ${endpoint.url}:`, response.data);
          
          // Check for transcription data in multiple possible formats
          const transcriptionText = response.data.transcription || 
                                   response.data.transcript || 
                                   response.data.text ||
                                   (typeof response.data === 'string' ? response.data : null);
          
          if (transcriptionText) {
            console.log(`✅ REST API transcription fallback successful, received ${transcriptionText.length} characters`);
            if (isMounted.current) {
              // Mark that we've received transcription data
              transcriptionReceivedRef.current = true;
              
              setTranscription(transcriptionText);
              setTranscriptionStatus('loaded');
              setLoading(false);
              setTranscriptionLoading(false);
              setProcessingYoutubeUrl(false);
              setProcessingStage('transcription-complete');
              
              // Reset timestamp generation attempt counter
              timestampGenerationAttemptsRef.current = 0;
              
              // Wait a short while before generating timestamps to ensure state updates properly
              setTimeout(() => {
                if (isMounted.current) {
                  // Generate timestamps from the transcription
                  if (functionsRef.current.generateTimestampsFromTranscript) {
                    functionsRef.current.generateTimestampsFromTranscript();
                  }
                  
                  // Also fetch timestamps after successful transcription
                  if (functionsRef.current.fetchSidebarTimestamps) {
                    functionsRef.current.fetchSidebarTimestamps(videoId);
                  }
                }
              }, TRANSCRIPTION_DELAY);
            }
            
            // Clear active request flag
            activeApiRequestRef.current = false;
            return true;
          }
        } catch (endpointError) {
          console.log(`Endpoint ${endpoint.url} failed:`, endpointError.message);
          
          // Try to handle auth errors
          if (endpointError.response && endpointError.response.status === 401) {
            const refreshed = await handleApiError(endpointError, endpoint.url);
            if (refreshed) {
              // Try again with refreshed token
              try {
                let retryResponse;
                const updatedHeaders = getAuthHeaders();
                
                if (endpoint.method === 'get') {
                  retryResponse = await api.get(endpoint.url, { 
                    headers: updatedHeaders, 
                    timeout: API_TIMEOUT 
                  });
                } else {
                  retryResponse = await api.post(endpoint.url, endpoint.data || {}, { 
                    headers: updatedHeaders, 
                    timeout: API_TIMEOUT 
                  });
                }
                
                if (retryResponse.data && 
                   (retryResponse.data.transcription || retryResponse.data.transcript || retryResponse.data.text)) {
                  
                  const transcriptionText = retryResponse.data.transcription || 
                                           retryResponse.data.transcript || 
                                           retryResponse.data.text;
                  
                  console.log('✅ REST API transcription fallback successful after token refresh');
                  if (isMounted.current) {
                    // Mark that we've received transcription data
                    transcriptionReceivedRef.current = true;
                    
                    setTranscription(transcriptionText);
                    setTranscriptionStatus('loaded');
                    setLoading(false);
                    setTranscriptionLoading(false);
                    setProcessingYoutubeUrl(false);
                    setProcessingStage('transcription-complete');
                    
                    // Reset timestamp generation attempt counter
                    timestampGenerationAttemptsRef.current = 0;
                    
                    // Wait a short while before generating timestamps to ensure state updates properly
                    setTimeout(() => {
                      if (isMounted.current) {
                        // Generate timestamps from the transcription
                        if (functionsRef.current.generateTimestampsFromTranscript) {
                          functionsRef.current.generateTimestampsFromTranscript();
                        }
                        
                        // Also fetch timestamps after successful transcription
                        if (functionsRef.current.fetchSidebarTimestamps) {
                          functionsRef.current.fetchSidebarTimestamps(videoId);
                        }
                      }
                    }, TRANSCRIPTION_DELAY);
                  }
                  
                  // Clear active request flag
                  activeApiRequestRef.current = false;
                  return true;
                }
              } catch (retryError) {
                console.log(`Retry after token refresh failed for ${endpoint.url}:`, retryError.message);
              }
            }
          }
        }
      }
      
      // NEW: Special handling for upload_ID_HASH formats vs file_ID formats
      // Try to extract from saved transcription files using best possible alternatives
      // This approach assumes the server uses a similar pattern for storage as in the logs
      if (videoId.startsWith('file_')) {
        // Derive potential upload ID format without hash
        const fileTimestamp = videoId.replace('file_', '');
        
        // Try looking directly in the transcription file formats
        try {
          // Try two formats for the transcription file
          for (const format of ['json', 'txt']) {
            try {
              const response = await api.get(`/data/transcriptions/transcription_upload_${fileTimestamp}.${format}`);
              if (response.data) {
                let transcriptionText = '';
                
                if (typeof response.data === 'object' && response.data.transcription) {
                  transcriptionText = response.data.transcription;
                } else if (typeof response.data === 'string') {
                  transcriptionText = response.data;
                }
                
                if (transcriptionText && transcriptionText.length > 0) {
                  console.log(`✅ Direct transcription file access successful, ${transcriptionText.length} chars`);
                  if (isMounted.current) {
                    transcriptionReceivedRef.current = true;
                    setTranscription(transcriptionText);
                    setTranscriptionStatus('loaded');
                    setLoading(false);
                    setTranscriptionLoading(false);
                    setProcessingYoutubeUrl(false);
                    setProcessingStage('transcription-complete');
                    
                    // Reset timestamp generation attempt counter
                    timestampGenerationAttemptsRef.current = 0;
                    
                    setTimeout(() => {
                      if (isMounted.current) {
                        if (functionsRef.current.generateTimestampsFromTranscript) {
                          functionsRef.current.generateTimestampsFromTranscript();
                        }
                        if (functionsRef.current.fetchSidebarTimestamps) {
                          functionsRef.current.fetchSidebarTimestamps(videoId);
                        }
                      }
                    }, TRANSCRIPTION_DELAY);
                  }
                  
                  activeApiRequestRef.current = false;
                  return true;
                }
              }
            } catch (err) {
              console.log(`Direct file access failed for format ${format}:`, err.message);
            }
          }
        } catch (directErr) {
          console.log('All direct file access attempts failed');
        }
      }
      
      console.log('❌ All REST API fallback endpoints returned no transcription data');
      
      // Increment fallback attempts for potential retry strategies
      apiFallbackAttemptsRef.current++;
      
      // If we've tried a few times, update the error message
      if (apiFallbackAttemptsRef.current >= 3 && isMounted.current) {
        setError('Unable to transcribe video after multiple attempts. The server might be overloaded or the video may be too long.');
        setTranscriptionStatus('error');
        setLoading(false);
        setTranscriptionLoading(false);
      }
      
      // Clear active request flag
      activeApiRequestRef.current = false;
      return false;
    } catch (err) {
      console.error('❌ REST API transcription fallback failed:', err);
      // Clear active request flag
      activeApiRequestRef.current = false;
      
      // Don't update error state here, as we've already shown an error from the socket
      return false;
    }
  }, [tabId, getAuthHeaders, handleApiError, currentVideoId]);
  // Store the fetchTranscriptionViaREST implementation for later use
  functionsRef.current.fetchTranscriptionViaREST = fetchTranscriptionViaREST;
  // Generate timestamps from transcript text
  const generateTimestampsFromTranscript = useCallback(() => {
    // Clear any existing delayed generation timeout
    if (delayedTimestampGenRef.current) {
      clearTimeout(delayedTimestampGenRef.current);
      delayedTimestampGenRef.current = null;
    }
    
    console.log('Attempting to generate timestamps from transcript...');
    
    // Increment generation attempt counter
    timestampGenerationAttemptsRef.current += 1;
    
    if (!transcription) {
      console.warn('No transcription available to generate timestamps from');
      
      // If we've made multiple attempts and still don't have transcription,
      // don't keep trying indefinitely
      if (timestampGenerationAttemptsRef.current > 5) {
        console.error('Failed to generate timestamps after multiple attempts. No transcription available.');
        return;
      }
      
      // Schedule another attempt if we're still in the loading state
      if (transcriptionLoading) {
        console.log('Transcription still loading, will try generating timestamps again shortly');
        delayedTimestampGenRef.current = setTimeout(() => {
          if (isMounted.current) {
            generateTimestampsFromTranscript();
          }
        }, 2000); // Try again in 2 seconds
      }
      
      return;
    }
    
    try {
      console.log('Generating timestamps from transcript text, length:', transcription.length);
      
      // If transcription is just whitespace or extremely short, don't try to parse it
      if (transcription.trim().length < 10) {
        console.warn('Transcription is too short to generate meaningful timestamps');
        return;
      }
      
      // Try to get video duration from videoData, fall back to estimating based on transcript length
      // A rough estimate is about 5-7 characters per second of video
      const charsPerSecond = 6;
      const estimatedDuration = videoData?.duration || Math.max(60, Math.ceil(transcription.length / charsPerSecond));
      console.log(`Using estimated duration: ${estimatedDuration} seconds`);
      
      // Create timestamps from the transcript text - use proper sentence splitting
      // This regex handles periods, question marks, exclamation points followed by spaces or end of string
      const sentences = transcription
        .split(/[.!?](?:\s|$)/)
        .filter(s => s.trim().length > 0)
        .map(s => s.trim());
      
      console.log(`Split transcript into ${sentences.length} sentences`);
      
      if (sentences.length === 0) {
        console.warn('No sentences found in transcript');
        return;
      }
      
      const totalLength = transcription.length;
      const generatedTimestamps = sentences.map((sentence, index) => {
        // Find position of this sentence in the full text
        const startPosition = transcription.indexOf(sentence);
        if (startPosition === -1) {
          console.warn(`Couldn't find sentence in transcript: ${sentence.substring(0, 30)}...`);
          // Use approximate position based on index if we can't find it
          const approxPosition = (index / sentences.length) * totalLength;
          const startTime = Math.max(0, Math.floor((approxPosition / totalLength) * estimatedDuration));
          const endTime = Math.min(estimatedDuration, Math.floor(((index + 1) / sentences.length) * estimatedDuration));
          
          // Format times
          const startMin = Math.floor(startTime / 60);
          const startSec = Math.floor(startTime % 60);
          const startFormatted = `${startMin}:${startSec.toString().padStart(2, '0')}`;
          
          const endMin = Math.floor(endTime / 60);
          const endSec = Math.floor(endTime % 60);
          const endFormatted = `${endMin}:${endSec.toString().padStart(2, '0')}`;
          
          return {
            start_time: startTime,
            end_time: endTime,
            display_time: `${startFormatted} - ${endFormatted}`,
            text: sentence
          };
        }
        
        // Calculate times based on position in text
        const startTime = Math.max(0, Math.floor((startPosition / totalLength) * estimatedDuration));
        
        // Calculate end time (start time of next sentence or end of video)
        let endTime;
        if (index < sentences.length - 1) {
          const nextSentence = sentences[index + 1];
          const nextSentencePos = transcription.indexOf(nextSentence);
          
          if (nextSentencePos !== -1) {
            endTime = Math.min(estimatedDuration, Math.floor((nextSentencePos / totalLength) * estimatedDuration));
          } else {
            // If we can't find the next sentence, estimate it
            endTime = Math.min(estimatedDuration, Math.floor(startTime + (sentence.length / charsPerSecond)));
          }
        } else {
          endTime = estimatedDuration;
        }
        
        // Format time strings
        const startMin = Math.floor(startTime / 60);
        const startSec = Math.floor(startTime % 60);
        const startFormatted = `${startMin}:${startSec.toString().padStart(2, '0')}`;
        
        const endMin = Math.floor(endTime / 60);
        const endSec = Math.floor(endTime % 60);
        const endFormatted = `${endMin}:${endSec.toString().padStart(2, '0')}`;
        
        return {
          start_time: startTime,
          end_time: endTime,
          display_time: `${startFormatted} - ${endFormatted}`,
          text: sentence
        };
      });
      
      if (generatedTimestamps.length > 0) {
        console.log(`Successfully generated ${generatedTimestamps.length} timestamps from transcript`);
        setSidebarTimestamps(convertTimestampFormat(generatedTimestamps));
        setTimestamps(generatedTimestamps);
        setHasGeneratedTimestamps(true);
      } else {
        console.warn('Failed to generate any timestamps from transcript');
      }
    } catch (error) {
      console.error('Error generating timestamps from transcript:', error);
      
      // If this was the first attempt, try again with a delay
      if (timestampGenerationAttemptsRef.current <= 2) {
        console.log('Will retry timestamp generation shortly');
        delayedTimestampGenRef.current = setTimeout(() => {
          if (isMounted.current) {
            generateTimestampsFromTranscript();
          }
        }, 2000); // Try again in 2 seconds
      }
    }
  }, [transcription, videoData, convertTimestampFormat, transcriptionLoading]);
  // Store the generateTimestampsFromTranscript implementation for later use
  functionsRef.current.generateTimestampsFromTranscript = generateTimestampsFromTranscript;
  // Function to monitor and recover from stalled transcription with more aggressive fallback
  const setupTranscriptionTimeout = useCallback((videoId) => {
    console.log('Setting up transcription monitoring for:', videoId);
    // Clear any existing timeout
    if (fallbackTimerRef.current) {
      clearTimeout(fallbackTimerRef.current);
    }
    // Set an initial quick check timeout
    fallbackTimerRef.current = setTimeout(() => {
      console.log('⏱️ Initial transcription check');
      
      // If we're still loading, start monitoring
      if (transcriptionLoading && isMounted.current) {
        console.log('Transcription still loading, setting up monitoring sequence');
        
        // Clear the current timeout
        clearTimeout(fallbackTimerRef.current);
        
        // First recovery attempt - shorter timeout
        fallbackTimerRef.current = setTimeout(() => {
          if (transcriptionLoading && isMounted.current) {
            console.log('⏱️ First recovery attempt for stalled transcription');
            
            // Try to ping the server for status
            if (socketService.isSocketConnected() && socketConnectedRef.current) {
              try {
                socketService.emit('ping', { tab_id: tabId, video_id: videoId });
                socketService.emit('check_transcription_status', { tab_id: tabId, video_id: videoId });
              } catch (e) {
                console.error('Error sending ping:', e);
              }
            }
            
            // Set up second recovery attempt
            clearTimeout(fallbackTimerRef.current);
            fallbackTimerRef.current = setTimeout(() => {
              if (transcriptionLoading && isMounted.current) {
                console.log('⏱️ Second recovery attempt - trying socket reconnection');
                
                // Try socket reconnection
                try {
                  socketService.reconnect();
                  setTimeout(() => {
                    if (socketService.isSocketConnected() && isMounted.current) {
                      socketService.emit('check_transcription_status', { 
                        tab_id: tabId, 
                        video_id: videoId 
                      });
                    }
                  }, 1000);
                } catch (e) {
                  console.error('Socket reconnection failed:', e);
                }
                
                // Set up third recovery attempt - try REST API
                clearTimeout(fallbackTimerRef.current);
                fallbackTimerRef.current = setTimeout(() => {
                  if (transcriptionLoading && isMounted.current) {
                    console.log('⏱️ Third recovery attempt - trying REST API fallback');
                    if (functionsRef.current.fetchTranscriptionViaREST) {
                      functionsRef.current.fetchTranscriptionViaREST(videoId);
                    }
                    
                    // Final attempt with more information for user
                    clearTimeout(fallbackTimerRef.current);
                    fallbackTimerRef.current = setTimeout(() => {
                      if (transcriptionLoading && isMounted.current) {
                        console.log('⏱️ Final recovery attempt - still not responding');
                        setError('Transcription is taking longer than expected. We\'re still trying, but this might indicate an issue with the video or server load.');
                        
                        // One last REST API attempt
                        if (functionsRef.current.fetchTranscriptionViaREST) {
                          functionsRef.current.fetchTranscriptionViaREST(videoId);
                        }
                      }
                    }, 30000); // Final check after 30 more seconds
                  }
                }, 15000); // Third attempt after 15 seconds
              }
            }, 15000); // Second attempt after 15 seconds
          }
        }, 20000); // First recovery after 20 seconds 
      }
    }, 10000); // Initial check after 10 seconds
    return () => {
      // Cleanup function
      if (fallbackTimerRef.current) {
        clearTimeout(fallbackTimerRef.current);
        fallbackTimerRef.current = null;
      }
    };
  }, [tabId, transcriptionLoading]);
  
  // Request visual analysis
  const requestVisualAnalysis = useCallback(async () => {
    console.log("Request visual analysis for:", currentVideoId);
    if (!currentVideoId) {
      console.warn("No video ID available for visual analysis");
      return;
    }
    
    setVisualAnalysisInProgress(true);
    try {
      // Get auth headers
      const headers = getAuthHeaders();
      
      // Try both socket and REST API methods for requesting visual analysis
      let requestSent = false;
      
      // Try socket first if connected
      if (socketService.isSocketConnected() && socketConnectedRef.current) {
        try {
          await socketService.requestVisualAnalysis(currentVideoId);
          requestSent = true;
          console.log("Visual analysis request sent via socket");
        } catch (socketError) {
          console.error('Socket error requesting visual analysis:', socketError);
        }
      }
      
      // Fallback to REST API if socket failed
      if (!requestSent) {
        try {
          // Try multiple endpoints
          const endpoints = [
            `/api/v1/videos/${currentVideoId}/analyze-visual`,
            `/api/videos/${currentVideoId}/visual-analysis`,
            `/api/analyze-visual?videoId=${currentVideoId}`,
            `/api/v1/visual-analysis?video_id=${currentVideoId}`
          ];
          
          for (const endpoint of endpoints) {
            try {
              console.log(`Trying visual analysis endpoint: ${endpoint}`);
              await api.post(endpoint, {}, { headers, timeout: VISUAL_FEATURE_TIMEOUT });
              requestSent = true;
              console.log(`Successfully requested visual analysis via ${endpoint}`);
              break;
            } catch (endpointError) {
              console.log(`Endpoint ${endpoint} failed:`, endpointError.message);
              
              // Try to handle auth errors
              if (endpointError.response && endpointError.response.status === 401) {
                const refreshed = await handleApiError(endpointError, endpoint);
                if (refreshed) {
                  // Try again with refreshed token
                  try {
                    await api.post(endpoint, {}, { 
                      headers: getAuthHeaders(), 
                      timeout: VISUAL_FEATURE_TIMEOUT 
                    });
                    requestSent = true;
                    console.log(`Successfully requested visual analysis after token refresh via ${endpoint}`);
                    break;
                  } catch (retryError) {
                    console.log(`Retry after token refresh failed for ${endpoint}:`, retryError.message);
                  }
                }
              }
            }
          }
        } catch (apiError) {
          console.error('API error requesting visual analysis:', apiError);
        }
      }
      
      // If no request was sent successfully, try one more approach - direct URL request
      if (!requestSent) {
        console.log("Trying alternative visual analysis request approaches");
        
        try {
          // Try a direct GET request to the analyze endpoint
          const response = await api.get(`/api/v1/videos/${currentVideoId}/visual`, { 
            headers, 
            timeout: VISUAL_FEATURE_TIMEOUT 
          });
          
          if (response.status === 200 || response.status === 202) {
            console.log("Successfully requested visual analysis via direct GET");
            requestSent = true;
          }
        } catch (directError) {
          console.log("Direct GET request for visual analysis failed:", directError.message);
        }
      }
      
      // If no request was sent successfully, still keep analysis in progress
      // but log an error
      if (!requestSent) {
        console.error('Failed to request visual analysis through any method');
        // Set a timeout to check for visual data directly
        setTimeout(() => {
          if (functionsRef.current.fetchVisualData && isMounted.current) {
            functionsRef.current.fetchVisualData(currentVideoId);
          }
        }, 5000);
      }
    } catch (error) {
      console.error('Error in requestVisualAnalysis:', error);
    }
    
    // Do not reset the visual analysis in progress flag here - wait for the
    // server to notify us when analysis is complete
  }, [currentVideoId, getAuthHeaders, handleApiError]);
  // Get video summary
  const getVideoSummary = useCallback(async () => {
    console.log("Get video summary for:", currentVideoId);
    if (!currentVideoId) return;
    
    try {
      // Get auth headers
      const headers = getAuthHeaders();
      
      // Try both socket and REST API methods
      let summaryRequested = false;
      
      // Try socket first if connected
      if (socketService.isSocketConnected() && socketConnectedRef.current) {
        try {
          const response = await socketService.getVideoSummary(currentVideoId);
          if (response && response.summary) {
            setVisualSummary(response.summary);
            summaryRequested = true;
          }
        } catch (socketError) {
          console.error('Socket error getting video summary:', socketError);
        }
      }
      
      // Fallback to REST API if socket failed
      if (!summaryRequested) {
        try {
          // Try multiple endpoints
          const endpoints = [
            `/api/v1/videos/${currentVideoId}/summary`,
            `/api/videos/${currentVideoId}/summary`,
            `/api/summary?videoId=${currentVideoId}`,
            `/api/v1/summary?video_id=${currentVideoId}`
          ];
          
          for (const endpoint of endpoints) {
            try {
              console.log(`Trying summary endpoint: ${endpoint}`);
              const response = await api.get(endpoint, { 
                headers, 
                timeout: VISUAL_FEATURE_TIMEOUT
              });
              
              if (response.data && response.data.summary) {
                setVisualSummary(response.data.summary);
                summaryRequested = true;
                break;
              }
            } catch (endpointError) {
              console.log(`Endpoint ${endpoint} failed:`, endpointError.message);
              
              // Try to handle auth errors
              if (endpointError.response && endpointError.response.status === 401) {
                const refreshed = await handleApiError(endpointError, endpoint);
                if (refreshed) {
                  // Try again with refreshed token
                  try {
                    const retryResponse = await api.get(endpoint, { 
                      headers: getAuthHeaders(), 
                      timeout: VISUAL_FEATURE_TIMEOUT 
                    });
                    
                    if (retryResponse.data && retryResponse.data.summary) {
                      setVisualSummary(retryResponse.data.summary);
                      summaryRequested = true;
                      break;
                    }
                  } catch (retryError) {
                    console.log(`Retry after token refresh failed for ${endpoint}:`, retryError.message);
                  }
                }
              }
            }
          }
        } catch (apiError) {
          console.error('API error getting video summary:', apiError);
        }
      }
      
      if (!summaryRequested) {
        console.error('Failed to request video summary through any method');
      }
    } catch (error) {
      console.error('Error in getVideoSummary:', error);
    }
  }, [currentVideoId, getAuthHeaders, handleApiError]);
  // Get video topic analysis
  const getVideoTopicAnalysis = useCallback(async () => {
    console.log("Get video topic analysis for:", currentVideoId);
    if (!currentVideoId) return;
    
    try {
      // Get auth headers
      const headers = getAuthHeaders();
      
      // Try both socket and REST API methods
      let topicsRequested = false;
      
      // Try socket first if connected
      if (socketService.isSocketConnected() && socketConnectedRef.current) {
        try {
          const response = await socketService.getVideoTopics(currentVideoId);
          if (response && response.topics && response.topics.length > 0) {
            setVideoTopics(response.topics);
            topicsRequested = true;
          }
        } catch (socketError) {
          console.error('Socket error getting video topics:', socketError);
        }
      }
      
      // Fallback to REST API if socket failed
      if (!topicsRequested) {
        try {
          // Try multiple endpoints
          const endpoints = [
            `/api/v1/videos/${currentVideoId}/topics`,
            `/api/videos/${currentVideoId}/topics`,
            `/api/topics?videoId=${currentVideoId}`,
            `/api/v1/topics?video_id=${currentVideoId}`
          ];
          
          for (const endpoint of endpoints) {
            try {
              console.log(`Trying topics endpoint: ${endpoint}`);
              const response = await api.get(endpoint, { 
                headers, 
                timeout: VISUAL_FEATURE_TIMEOUT
              });
              
              if (response.data && response.data.topics && response.data.topics.length > 0) {
                setVideoTopics(response.data.topics);
                topicsRequested = true;
                break;
              }
            } catch (endpointError) {
              console.log(`Endpoint ${endpoint} failed:`, endpointError.message);
              
              // Try to handle auth errors
              if (endpointError.response && endpointError.response.status === 401) {
                const refreshed = await handleApiError(endpointError, endpoint);
                if (refreshed) {
                  // Try again with refreshed token
                  try {
                    const retryResponse = await api.get(endpoint, { 
                      headers: getAuthHeaders(), 
                      timeout: VISUAL_FEATURE_TIMEOUT 
                    });
                    
                    if (retryResponse.data && retryResponse.data.topics && retryResponse.data.topics.length > 0) {
                      setVideoTopics(retryResponse.data.topics);
                      topicsRequested = true;
                      break;
                    }
                  } catch (retryError) {
                    console.log(`Retry after token refresh failed for ${endpoint}:`, retryError.message);
                  }
                }
              }
            }
          }
        } catch (apiError) {
          console.error('API error getting video topics:', apiError);
        }
      }
      
      if (!topicsRequested) {
        console.log('Failed to request video topics through any method, returning empty array');
        // Return empty array rather than throwing error
        setVideoTopics([]);
      }
    } catch (error) {
      console.error('Error in getVideoTopicAnalysis:', error);
      // Set empty array on error
      setVideoTopics([]);
    }
  }, [currentVideoId, getAuthHeaders, handleApiError]);
  // Upload video file - IMPROVED VERSION
  const uploadVideoFile = useCallback(async (file) => {
    console.log("Upload video file:", file);
    try {
      // Get auth headers
      const headers = getAuthHeaders();
      
      // Create form data for the file upload
      const formData = new FormData();
      formData.append('file', file);
      formData.append('tab_id', tabId);
      
      // Generate a unique file ID now
      const timestamp = Date.now();
      const fileId = `file_${timestamp}`;
      
      // Add the fileId to the form data
      formData.append('file_id', fileId);
      formData.append('video_id', fileId);
      
      // Try multiple endpoints with progress tracking
      const endpoints = [
        '/api/v1/videos/upload',
        '/api/videos/upload',
        '/api/upload-video'
      ];
      
      for (const endpoint of endpoints) {
        try {
          console.log(`Trying video upload endpoint: ${endpoint}`);
          
          // Make the upload request with progress tracking
          const response = await api.post(endpoint, formData, {
            headers: {
              ...headers,
              'Content-Type': 'multipart/form-data',
            },
            onUploadProgress: (progressEvent) => {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              console.log(`Upload progress: ${percentCompleted}%`);
              // You could update a progress state here if needed
            },
            timeout: 60000, // Longer timeout for uploads
          });
          
          if (response.data && (response.data.success || response.status === 200)) {
            // Extract video ID with priority to server-assigned ID
            let videoId = response.data.video_id || fileId;
            
            // Check if there's an uploaded file ID in the response
            const extractedId = extractIdFromResponse(response.data);
            if (extractedId) {
              videoId = extractedId;
              console.log(`Found upload ID in response: ${videoId}`);
            }
            
            setCurrentVideoId(videoId);
            setVideoData({ 
              fileUpload: true, 
              fileName: file.name, 
              videoId: videoId 
            });
            
            // Setup processing and monitoring
            setTranscriptionStatus('uploading');
            setTranscriptionLoading(true);
            setProcessingStage('uploading');
            
            // Set up monitoring for stalled transcription
            setupTranscriptionTimeout(videoId);
            
            // Navigate to analysis page
            if (functionsRef.current.navigateToAnalysisPage) {
              functionsRef.current.navigateToAnalysisPage();
            }
            
            // Check transcription status after upload
            setTimeout(() => {
              if (isMounted.current) {
                // Ping for transcription status
                if (socketService.isSocketConnected() && socketConnectedRef.current) {
                  try {
                    // Use both original fileId and extracted ID for better compatibility
                    socketService.emit('check_transcription_status', { 
                      tab_id: tabId, 
                      video_id: videoId 
                    });
                    
                    // Also try with the fileId as a fallback
                    if (videoId !== fileId) {
                      socketService.emit('check_transcription_status', { 
                        tab_id: tabId, 
                        video_id: fileId,
                        file_id: fileId  // NEW: Add file_id parameter explicitly
                      });
                    }
                  } catch (e) {
                    console.error('Error checking transcription status:', e);
                  }
                }
                
                // Try REST API fallback
                if (functionsRef.current.fetchTranscriptionViaREST) {
                  // Try first with the extracted or server-provided ID
                  functionsRef.current.fetchTranscriptionViaREST(videoId);
                  
                  // If they differ, also try with the fileId as fallback
                  if (videoId !== fileId) {
                    setTimeout(() => {
                      if (isMounted.current && !transcriptionReceivedRef.current) {
                        functionsRef.current.fetchTranscriptionViaREST(fileId);
                      }
                    }, 3000);
                  }
                }
              }
            }, 5000); // Check after 5 seconds
            
            return { 
              success: true, 
              videoId: videoId,
              fileId: fileId // NEW: Return fileId for reference
            };
          }
        } catch (endpointError) {
          console.log(`Endpoint ${endpoint} failed:`, endpointError.message);
          
          // Try to handle auth errors
          if (endpointError.response && endpointError.response.status === 401) {
            const refreshed = await handleApiError(endpointError, endpoint);
            if (refreshed) {
              // Try again with refreshed token
              try {
                const updatedHeaders = {
                  ...getAuthHeaders(),
                  'Content-Type': 'multipart/form-data',
                };
                
                const retryResponse = await api.post(endpoint, formData, {
                  headers: updatedHeaders,
                  onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round(
                      (progressEvent.loaded * 100) / progressEvent.total
                    );
                    console.log(`Upload progress after token refresh: ${percentCompleted}%`);
                  },
                  timeout: 60000,
                });
                
                if (retryResponse.data && (retryResponse.data.success || retryResponse.status === 200)) {
                  // Extract video ID with priority to server-assigned ID
                  let videoId = retryResponse.data.video_id || fileId;
                  
                  // Check if there's an uploaded file ID in the response
                  const extractedId = extractIdFromResponse(retryResponse.data);
                  if (extractedId) {
                    videoId = extractedId;
                    console.log(`Found upload ID in retry response: ${videoId}`);
                  }
                  
                  setCurrentVideoId(videoId);
                  setVideoData({ 
                    fileUpload: true, 
                    fileName: file.name, 
                    videoId: videoId 
                  });
                  
                  // Setup processing and monitoring
                  setTranscriptionStatus('uploading');
                  setTranscriptionLoading(true);
                  setProcessingStage('uploading');
                  
                  // Set up monitoring for stalled transcription
                  setupTranscriptionTimeout(videoId);
                  
                  // Navigate to analysis page
                  if (functionsRef.current.navigateToAnalysisPage) {
                    functionsRef.current.navigateToAnalysisPage();
                  }
                  
                  // Check transcription status after upload
                  setTimeout(() => {
                    if (isMounted.current) {
                      // Ping for transcription status
                      if (socketService.isSocketConnected() && socketConnectedRef.current) {
                        try {
                          // Use both original fileId and extracted ID for better compatibility
                          socketService.emit('check_transcription_status', { 
                            tab_id: tabId, 
                            video_id: videoId 
                          });
                          
                          // Also try with the fileId as a fallback
                          if (videoId !== fileId) {
                            socketService.emit('check_transcription_status', { 
                              tab_id: tabId, 
                              video_id: fileId,
                              file_id: fileId  // NEW: Add file_id parameter explicitly
                            });
                          }
                        } catch (e) {
                          console.error('Error checking transcription status:', e);
                        }
                      }
                      
                      // Try REST API fallback
                      if (functionsRef.current.fetchTranscriptionViaREST) {
                        // Try first with the extracted or server-provided ID
                        functionsRef.current.fetchTranscriptionViaREST(videoId);
                        
                        // If they differ, also try with the fileId as fallback
                        if (videoId !== fileId) {
                          setTimeout(() => {
                            if (isMounted.current && !transcriptionReceivedRef.current) {
                              functionsRef.current.fetchTranscriptionViaREST(fileId);
                            }
                          }, 3000);
                        }
                      }
                    }
                  }, 5000); // Check after 5 seconds
                  
                  return { 
                    success: true, 
                    videoId: videoId,
                    fileId: fileId // NEW: Return fileId for reference
                  };
                }
              } catch (retryError) {
                console.log(`Retry after token refresh failed for ${endpoint}:`, retryError.message);
              }
            }
          }
        }
      }
      
      throw new Error('All upload endpoints failed');
    } catch (error) {
      console.error('Error uploading video file:', error);
      setError('Failed to upload video: ' + error.message);
      return { success: false, error: error.message };
    }
  }, [tabId, getAuthHeaders, handleApiError, extractIdFromResponse, setupTranscriptionTimeout]);
  // Upload audio file
  // Upload audio file
const uploadAudioFile = useCallback(async (file) => {
  console.log("Upload audio file:", file);
  try {
    // Instead of reimplementing the upload logic, use the fixed api.uploadFile function
    const response = await api.uploadFile(file, 'audio', tabId);
    
    if (response.success) {
      const videoId = response.videoId || response.audioId;
      
      // Update state with audio file info
      setCurrentVideoId(videoId);
      setVideoData({ 
        fileUpload: true, 
        fileName: file.name, 
        videoId: videoId,
        isAudio: true
      });
      
      // Setup processing and monitoring
      setTranscriptionStatus('uploading');
      setTranscriptionLoading(true);
      setProcessingStage('uploading');
      
      // Set up monitoring for stalled transcription
      setupTranscriptionTimeout(videoId);
      
      // Navigate to analysis page
      if (functionsRef.current.navigateToAnalysisPage) {
        functionsRef.current.navigateToAnalysisPage();
      }
      
      // Check transcription status after upload
      setTimeout(() => {
        if (isMounted.current) {
          // Ping for transcription status
          if (socketService.isSocketConnected() && socketConnectedRef.current) {
            try {
              socketService.emit('check_transcription_status', { 
                tab_id: tabId, 
                video_id: videoId 
              });
            } catch (e) {
              console.error('Error checking transcription status:', e);
            }
          }
          
          // Try REST API fallback
          if (functionsRef.current.fetchTranscriptionViaREST) {
            functionsRef.current.fetchTranscriptionViaREST(videoId);
          }
        }
      }, 5000); // Check after 5 seconds
      
      return { 
        success: true, 
        videoId: videoId
      };
    } else {
      throw new Error(response.error || 'Upload failed');
    }
  } catch (error) {
    console.error('Error uploading audio file:', error);
    setError('Failed to upload audio: ' + error.message);
    return { success: false, error: error.message };
  }
}, [tabId, getAuthHeaders, setupTranscriptionTimeout]);

  // Function to detect if a question is related to timestamps
  const detectTimestampQuestion = useCallback((question) => {
    if (!question) return false;
    
    const lowercaseQuestion = question.toLowerCase();
    
    // Check for direct timestamp mentions
    const timestampPatterns = [
      /at\s+(\d+):(\d+)/i, // "at 1:30"
      /at\s+(\d+)\s+minute/i, // "at 2 minutes"
      /at\s+(\d+)\s+second/i, // "at 45 seconds"
      /time\s+(\d+):(\d+)/i, // "time 1:30"
      /timestamp\s+(\d+):(\d+)/i, // "timestamp 1:30"
      /\b(\d+):(\d+)\b/i // "1:30" standalone
    ];
    
    // Check for "when" questions
    const whenPatterns = [
      /\bwhen\b.*\bshown\b/i,
      /\bwhen\b.*\bappears?\b/i,
      /\bwhen\b.*\bhappens?\b/i,
      /\bwhen\b.*\bsee\b/i,
      /\bwhen\b.*\bdisplayed\b/i,
      /\bwhen\b.*\bvisible\b/i,
      /\bat what (?:time|point|moment)\b/i
    ];
    
    // Check timestamp patterns
    for (const pattern of timestampPatterns) {
      if (pattern.test(lowercaseQuestion)) {
        return true;
      }
    }
    
    // Check when patterns
    for (const pattern of whenPatterns) {
      if (pattern.test(lowercaseQuestion)) {
        return true;
      }
    }
    
    return false;
  }, []);
  // Extract timestamp from question
  const extractTimestampFromQuestion = useCallback((question) => {
    if (!question) return null;
    
    const lowercaseQuestion = question.toLowerCase();
    
    // Check for MM:SS format
    const mmssPattern = /\b(\d+):(\d+)\b/i;
    const mmssMatch = lowercaseQuestion.match(mmssPattern);
    
    if (mmssMatch) {
      const minutes = parseInt(mmssMatch[1], 10);
      const seconds = parseInt(mmssMatch[2], 10);
      
      return {
        minutes,
        secondsComponent: seconds,
        formatted: `${minutes}:${seconds.toString().padStart(2, '0')}`,
        totalSeconds: minutes * 60 + seconds
      };
    }
    
    // Check for "X minutes Y seconds" format
    const minutesSecondsPattern = /\b(\d+)\s+minute(?:s)?\s+(?:and\s+)?(\d+)\s+second(?:s)?\b/i;
    const minutesSecondsMatch = lowercaseQuestion.match(minutesSecondsPattern);
    
    if (minutesSecondsMatch) {
      const minutes = parseInt(minutesSecondsMatch[1], 10);
      const seconds = parseInt(minutesSecondsMatch[2], 10);
      
      return {
        minutes,
        secondsComponent: seconds,
        formatted: `${minutes}:${seconds.toString().padStart(2, '0')}`,
        totalSeconds: minutes * 60 + seconds
      };
    }
    
    // Check for "X minutes" format
    const minutesPattern = /\b(\d+)\s+minute(?:s)?\b/i;
    const minutesMatch = lowercaseQuestion.match(minutesPattern);
    
    if (minutesMatch) {
      const minutes = parseInt(minutesMatch[1], 10);
      
      return {
        minutes,
        secondsComponent: 0,
        formatted: `${minutes}:00`,
        totalSeconds: minutes * 60
      };
    }
    
    // Check for "X seconds" format
    const secondsPattern = /\b(\d+)\s+second(?:s)?\b/i;
    const secondsMatch = lowercaseQuestion.match(secondsPattern);
    
    if (secondsMatch) {
      const seconds = parseInt(secondsMatch[1], 10);
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      
      return {
        minutes,
        secondsComponent: remainingSeconds,
        formatted: `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`,
        totalSeconds: seconds
      };
    }
    
    return null;
  }, []);
  // Function to detect if a question is about visual content
  const detectVisualQuestion = useCallback((question) => {
    if (!question) return false;
    
    const lowercaseQuestion = question.toLowerCase();
    // Check for visual-related keywords
    const visualKeywords = [
      'see', 'show', 'shown', 'appears', 'appear', 'look',
      'visible', 'display', 'screen', 'image', 'picture',
      'visual', 'scene', 'background', 'object', 'person',
      'what is in', 'who is in', 'can you see', 'visible in',
      'on screen', 'in frame', 'on camera', 'in the video',
      'painting', 'text', 'showing'
    ];
    for (const keyword of visualKeywords) {
      if (lowercaseQuestion.includes(keyword)) {
        return true;
      }
    }
    return false;
  }, []);
  // ===== DEFINE CORE FUNCTIONS WITH CIRCULAR DEPENDENCIES =====
  // Fetch sidebar timestamps with comprehensive fallback
  const fetchSidebarTimestamps = useCallback(async (videoId = null) => {
    // Use provided videoId or current one
    const targetVideoId = videoId || currentVideoId;
    if (!targetVideoId) {
      console.warn('No video ID to fetch timestamps for');
      return;
    }
    
    console.log(`Fetching sidebar timestamps for video ID: ${targetVideoId}`);
    
    // First check if we already have timestamps to avoid duplicating work
    if (timestamps.length > 0 && sidebarTimestamps.length > 0) {
      console.log('Already have timestamps, skipping fetch');
      return;
    }
    
    // Check if we've already reached max attempts 
    if (timestampFetchAttempts >= TIMESTAMP_ERROR_THRESHOLD) {
      console.warn(`Skipping timestamp fetch after ${timestampFetchAttempts} failed attempts`);
      if (!timestampFetchFailed) {
        setTimestampFetchFailed(true);
      }
      
      // Always fall back to generating timestamps from transcript if available
      if (transcription && (!timestamps || timestamps.length === 0)) {
        console.log('Falling back to local timestamp generation after max attempts');
        if (functionsRef.current.generateTimestampsFromTranscript) {
          functionsRef.current.generateTimestampsFromTranscript();
        }
      }
      
      return;
    }
    
    try {
      // Get auth headers
      const headers = getAuthHeaders();
      
      // First, use our own generation function if we have transcript data
      // This provides a local fallback immediately so we're not dependent on API
      if (transcription && (!timestamps || timestamps.length === 0) && !hasGeneratedTimestamps) {
        console.log('Generating timestamps from transcript text first');
        if (functionsRef.current.generateTimestampsFromTranscript) {
          functionsRef.current.generateTimestampsFromTranscript();
        }
        // Still try other methods to potentially get better timestamps
      }
      
      // Try using Socket.IO if connected
      if (socketService.isSocketConnected() && socketConnectedRef.current) {
        try {
          console.log('Trying to get timestamps via socket');
          // Get timestamps using our socket service function
          const result = await socketService.getVideoTimestamps(targetVideoId);
          if (result && result.timestamps && result.timestamps.length > 0) {
            console.log(`Socket returned ${result.timestamps.length} timestamps`);
            const formattedTimestamps = convertTimestampFormat(result.timestamps);
            setSidebarTimestamps(formattedTimestamps);
            setTimestamps(result.timestamps);
            setTimestampFetchAttempts(0); // Reset attempts counter on success
            setHasGeneratedTimestamps(true);
            return;
          } else {
            console.log('Socket returned no timestamps');
          }
        } catch (socketError) {
          console.error('Socket timestamp fetch error:', socketError);
          // Increment the attempts counter
          setTimestampFetchAttempts(prev => prev + 1);
        }
      }
      
      // Skip REST API calls that might cause auth issues if we already have timestamps
      if (timestamps && timestamps.length > 0) {
        console.log('Using already generated timestamps instead of making API calls');
        return;
      }
      
      // Try multiple API endpoints to handle different backend configurations
      const endpoints = [
        // Direct path with video ID
        `/api/v1/videos/${targetVideoId}/timestamps`,
        `/api/videos/${targetVideoId}/timestamps`,
        `/api/timestamps/${targetVideoId}`,
        // Try for upload ID directly too
        `/api/v1/uploads/${targetVideoId}/timestamps`,
        `/api/uploads/${targetVideoId}/timestamps`,
        // Query param approaches
        `/api/timestamps?videoId=${targetVideoId}`,
        `/api/v1/timestamps?video_id=${targetVideoId}`,
        // Alternative endpoint patterns
        `/api/v1/timestamps/video/${targetVideoId}`,
        `/api/timestamps/video/${targetVideoId}`
      ];
      
      let fetchSuccess = false;
      
      // Try each endpoint until one works
      for (const endpoint of endpoints) {
        try {
          console.log(`Trying timestamps endpoint: ${endpoint}`);
          
          const response = await api.get(endpoint, { 
            headers, 
            timeout: API_TIMEOUT
          });
          
          if (response.data && (response.data.timestamps || response.data.data)) {
            const receivedTimestamps = response.data.timestamps || response.data.data;
            if (Array.isArray(receivedTimestamps) && receivedTimestamps.length > 0) {
              console.log(`API endpoint ${endpoint} returned ${receivedTimestamps.length} timestamps`);
              setSidebarTimestamps(convertTimestampFormat(receivedTimestamps));
              setTimestamps(receivedTimestamps);
              setTimestampFetchAttempts(0); // Reset on success
              setHasGeneratedTimestamps(true);
              fetchSuccess = true;
              break;
            }
          }
        } catch (endpointError) {
          console.warn(`Endpoint ${endpoint} failed:`, endpointError.message);
          
          // Try to handle auth errors
          if (endpointError.response && endpointError.response.status === 401) {
            const refreshed = await handleApiError(endpointError, endpoint);
            if (refreshed) {
              // Try again with refreshed token
              try {
                const retryResponse = await api.get(endpoint, { 
                  headers: getAuthHeaders(), 
                  timeout: API_TIMEOUT 
                });
                
                if (retryResponse.data && (retryResponse.data.timestamps || retryResponse.data.data)) {
                  const receivedTimestamps = retryResponse.data.timestamps || retryResponse.data.data;
                  if (Array.isArray(receivedTimestamps) && receivedTimestamps.length > 0) {
                    console.log(`API endpoint ${endpoint} returned ${receivedTimestamps.length} timestamps after token refresh`);
                    setSidebarTimestamps(convertTimestampFormat(receivedTimestamps));
                    setTimestamps(receivedTimestamps);
                    setTimestampFetchAttempts(0); // Reset on success
                    setHasGeneratedTimestamps(true);
                    fetchSuccess = true;
                    break;
                  }
                }
              } catch (retryError) {
                console.log(`Retry after token refresh failed for ${endpoint}:`, retryError.message);
              }
            }
          }
        }
      }
      
      // If all API calls failed but we still don't have timestamps
      if (!fetchSuccess && (!hasGeneratedTimestamps || timestamps.length === 0)) {
        setTimestampFetchAttempts(prev => prev + 1);
        
        // Just ensure we generate timestamps from transcript if we have one
        if (transcription) {
          console.log('All timestamp endpoints failed, generating from transcript');
          if (functionsRef.current.generateTimestampsFromTranscript) {
            functionsRef.current.generateTimestampsFromTranscript();
          }
        }
      }
    } catch (err) {
      console.error('Error in fetchSidebarTimestamps:', err);
      setTimestampFetchAttempts(prev => prev + 1);
      
      // Always fall back to generating timestamps from transcript if available
      if (transcription && (!timestamps || timestamps.length === 0)) {
        console.log('Falling back to local timestamp generation after error');
        if (functionsRef.current.generateTimestampsFromTranscript) {
          functionsRef.current.generateTimestampsFromTranscript();
        }
      }
    }
  }, [
    currentVideoId,
    transcription,
    timestamps,
    sidebarTimestamps,
    convertTimestampFormat,
    timestampFetchAttempts,
    hasGeneratedTimestamps,
    getAuthHeaders,
    handleApiError
  ]);
  // Store the fetchSidebarTimestamps implementation for later use
  functionsRef.current.fetchSidebarTimestamps = fetchSidebarTimestamps;
  // Fetch visual data for a video with comprehensive fallback
  const fetchVisualData = useCallback(async (videoId = null) => {
    // Use provided videoId or current one
    const targetVideoId = videoId || currentVideoId;
    // Prevent multiple simultaneous fetches
    if (visualDataLoadingRef.current) {
      console.log('Visual data already loading, ignoring duplicate request');
      return;
    }
    if (!targetVideoId) {
      console.warn('No video ID to fetch visual data for');
      return;
    }
    try {
      // Set loading flag
      visualDataLoadingRef.current = true;
      console.log(`Fetching visual data for video ID: ${targetVideoId}`);
      
      // Get auth headers
      const headers = getAuthHeaders();
      
      // Try using Socket.IO if connected
      if (socketService.isSocketConnected() && socketConnectedRef.current) {
        let hasData = false;
        try {
          // Get scenes
          const scenesResponse = await Promise.race([
            socketService.getVideoScenes(targetVideoId),
            new Promise((_, reject) => setTimeout(() => reject(new Error('Scenes request timeout')), VISUAL_FEATURE_TIMEOUT))
          ]).catch(err => {
            console.log(`Scene fetch timeout: ${err.message}`);
            return { scenes: [] };
          });
          
          if (scenesResponse && scenesResponse.scenes && scenesResponse.scenes.length > 0) {
            setDetectedScenes(scenesResponse.scenes);
            hasData = true;
          }
          
          // Get topics if supported
          if (socketService.topicsSupported) {
            const topicsResponse = await Promise.race([
              socketService.getVideoTopics(targetVideoId),
              new Promise((_, reject) => setTimeout(() => reject(new Error('Topics request timeout')), VISUAL_FEATURE_TIMEOUT))
            ]).catch(err => {
              console.log(`Topics fetch timeout: ${err.message}`);
              return { topics: [] };
            });
            
            if (topicsResponse && topicsResponse.topics && topicsResponse.topics.length > 0) {
              setVideoTopics(topicsResponse.topics);
              hasData = true;
            } else {
              // Set empty array instead of leaving undefined
              setVideoTopics([]);
            }
          }
          
          // Get highlights if supported
          if (socketService.highlightsSupported) {
            const highlightsResponse = await Promise.race([
              socketService.getVideoHighlights(targetVideoId),
              new Promise((_, reject) => setTimeout(() => reject(new Error('Highlights request timeout')), VISUAL_FEATURE_TIMEOUT))
            ]).catch(err => {
              console.log(`Highlights fetch timeout: ${err.message}`);
              return { highlights: [] };
            });
            
            if (highlightsResponse && highlightsResponse.highlights && highlightsResponse.highlights.length > 0) {
              setVideoHighlights(highlightsResponse.highlights);
              hasData = true;
            } else {
              // Set empty array to prevent undefined
              setVideoHighlights([]);
            }
          }
          
          // Get key frames for visual understanding
          if (socketService.visualAnalysisSupported) {
            const framesResponse = await Promise.race([
              socketService.getVideoFrames(targetVideoId),
              new Promise((_, reject) => setTimeout(() => reject(new Error('Frames request timeout')), VISUAL_FEATURE_TIMEOUT))
            ]).catch(err => {
              console.log(`Frames fetch timeout: ${err.message}`);
              return { frames: [] };
            });
            
            if (framesResponse && framesResponse.frames && framesResponse.frames.length > 0) {
              setKeyFrames(framesResponse.frames);
              hasData = true;
            }
          }
        } catch (socketError) {
          console.error('Socket error fetching visual data:', socketError);
        }
        
        // Also fetch sidebar timestamps
        if (functionsRef.current.fetchSidebarTimestamps) {
          functionsRef.current.fetchSidebarTimestamps(targetVideoId);
        }
        
        // Only set visualAnalysisAvailable if we actually have data
        if (hasData && isMounted.current) {
          setVisualAnalysisAvailable(true);
          visualDataLoadingRef.current = false;
          return;
        }
      }
      
      // Fallback to API with multiple endpoint attempts
      const endpoints = [
        // Standard patterns
        `/api/v1/videos/${targetVideoId}/visual-data`,
        `/api/videos/${targetVideoId}/visual`,
        // Query parameter patterns
        `/api/analyze-visual?videoId=${targetVideoId}`,
        `/api/v1/visual-analysis?video_id=${targetVideoId}`,
        // Additional patterns
        `/api/v1/videos/${targetVideoId}/visual`,
        `/api/v1/visual-analysis/${targetVideoId}`,
        `/api/visual-analysis/${targetVideoId}`
      ];
      
      for (const endpoint of endpoints) {
        try {
          console.log(`Trying visual data endpoint: ${endpoint}`);
          const response = await api.get(endpoint, { 
            headers,
            timeout: VISUAL_FEATURE_TIMEOUT 
          });
          
          if (response.data && isMounted.current) {
            let hasData = false;
            // Extract relevant data
            if (response.data.scenes && response.data.scenes.length > 0) {
              setDetectedScenes(response.data.scenes);
              hasData = true;
            }
            if (response.data.topics && response.data.topics.length > 0) {
              setVideoTopics(response.data.topics);
              hasData = true;
            } else {
              // Set empty array to prevent undefined
              setVideoTopics([]);
            }
            if (response.data.highlights && response.data.highlights.length > 0) {
              setVideoHighlights(response.data.highlights);
              hasData = true;
            } else {
              // Set empty array to prevent undefined
              setVideoHighlights([]);
            }
            if (response.data.visual_summary) {
              setVisualSummary(response.data.visual_summary);
              hasData = true;
            }
            if (response.data.frames && response.data.frames.length > 0) {
              setKeyFrames(response.data.frames);
              hasData = true;
            }
            // Also fetch sidebar timestamps
            if (functionsRef.current.fetchSidebarTimestamps) {
              functionsRef.current.fetchSidebarTimestamps(targetVideoId);
            }
            // Only set visualAnalysisAvailable if we actually have data
            setVisualAnalysisAvailable(hasData);
            
            if (hasData) {
              visualDataLoadingRef.current = false;
              return;
            }
          }
        } catch (endpointError) {
          console.log(`Endpoint ${endpoint} failed:`, endpointError.message);
          
          // Try to handle auth errors
          if (endpointError.response && endpointError.response.status === 401) {
            const refreshed = await handleApiError(endpointError, endpoint);
            if (refreshed) {
              // Try again with refreshed token
              try {
                const retryResponse = await api.get(endpoint, { 
                  headers: getAuthHeaders(), 
                  timeout: VISUAL_FEATURE_TIMEOUT 
                });
                
                if (retryResponse.data && isMounted.current) {
                  let hasData = false;
                  // Extract relevant data
                  if (retryResponse.data.scenes && retryResponse.data.scenes.length > 0) {
                    setDetectedScenes(retryResponse.data.scenes);
                    hasData = true;
                  }
                  if (retryResponse.data.topics && retryResponse.data.topics.length > 0) {
                    setVideoTopics(retryResponse.data.topics);
                    hasData = true;
                  } else {
                    // Set empty array to prevent undefined
                    setVideoTopics([]);
                  }
                  if (retryResponse.data.highlights && retryResponse.data.highlights.length > 0) {
                    setVideoHighlights(retryResponse.data.highlights);
                    hasData = true;
                  } else {
                    // Set empty array to prevent undefined
                    setVideoHighlights([]);
                  }
                  if (retryResponse.data.visual_summary) {
                    setVisualSummary(retryResponse.data.visual_summary);
                    hasData = true;
                  }
                  if (retryResponse.data.frames && retryResponse.data.frames.length > 0) {
                    setKeyFrames(retryResponse.data.frames);
                    hasData = true;
                  }
                  
                  // Only set visualAnalysisAvailable if we actually have data
                  setVisualAnalysisAvailable(hasData);
                  
                  if (hasData) {
                    visualDataLoadingRef.current = false;
                    return;
                  }
                }
              } catch (retryError) {
                console.log(`Retry after token refresh failed for ${endpoint}:`, retryError.message);
              }
            }
          }
        }
      }
      
      // If we get here, we didn't get any data yet - request visual analysis
      if (!visualAnalysisInProgress && isMounted.current) {
        console.log('No visual data available, requesting visual analysis');
        setVisualAnalysisInProgress(true);
        
        // Try socket request first
        if (socketService.isSocketConnected() && socketConnectedRef.current) {
          try {
            await socketService.requestVisualAnalysis(targetVideoId);
          } catch (socketError) {
            console.error('Socket error requesting visual analysis:', socketError);
            
            // Fallback to REST API
            try {
              await api.post(`/api/v1/videos/${targetVideoId}/analyze-visual`, {}, { headers });
            } catch (apiError) {
              console.error('API error requesting visual analysis:', apiError);
              if (isMounted.current) {
                // Don't set to false here - let's assume it's still in progress
                // and the backend might still be processing
                
                // Instead, set up a timer to check again later
                setTimeout(() => {
                  if (isMounted.current && visualAnalysisInProgress) {
                    console.log("Visual analysis may still be in progress - setting default values");
                    // If still in progress after timeout, set default empty values for better UX
                    setVideoTopics([]);
                    setVideoHighlights([]);
                    // Keep visualAnalysisInProgress true - the user can retry if needed
                  }
                }, 20000); // 20 seconds timeout
              }
            }
          }
        } else {
          // Direct to REST API
          try {
            await api.post(`/api/v1/videos/${targetVideoId}/analyze-visual`, {}, { headers });
          } catch (apiError) {
            console.error('API error requesting visual analysis:', apiError);
            if (isMounted.current) {
              // Same approach as above - don't set to false immediately
              setTimeout(() => {
                if (isMounted.current && visualAnalysisInProgress) {
                  console.log("Visual analysis may still be in progress - setting default values");
                  setVideoTopics([]);
                  setVideoHighlights([]);
                }
              }, 20000); // 20 seconds timeout
            }
          }
        }
      }
    } catch (err) {
      console.error('Error fetching visual data:', err);
      if (isMounted.current) {
        // Always set empty arrays rather than undefined values
        setVideoTopics([]);
        setVideoHighlights([]);
        setVisualAnalysisInProgress(false);
        setError('Failed to load visual data. Some features may be limited.');
      }
    } finally {
      // Clear loading flag
      visualDataLoadingRef.current = false;
    }
  }, [currentVideoId, getAuthHeaders, handleApiError]);
  // Store the fetchVisualData implementation for later use
  functionsRef.current.fetchVisualData = fetchVisualData;
  
  // Set up socket event listeners - FIX for duplicate responses
  const setupSocketListeners = useCallback(() => {
    console.log("Setting up socket event listeners");
    
    // Make sure socket is connected before setting up listeners
    if (!socketService.isSocketConnected()) {
      console.warn("Socket not connected yet, cannot set up listeners");
      return false;
    }
    
    // More aggressive cleanup - explicitly remove key listeners before adding new ones
    try {
      // First try the safe cleanup
      socketService.safeCleanup();
      
      // Then explicitly remove critical event handlers to prevent duplicates
      if (socketService.socket) {
        socketService.socket.off('ai_response');
        socketService.socket.off('transcription');
        socketService.socket.off('response_complete');
        socketService.socket.off('error');
        socketService.socket.off('transcription_status');
        socketService.socket.off('upload_transcription_complete');
        socketService.socket.off('timestamps_data');
        socketService.socket.off('timestamps_available');
      }
    } catch (error) {
      console.error("Error cleaning up previous listeners:", error);
    }
    
    // Create registry to track which events we've already set up in this call
    const registeredEvents = {};
    
    // Helper function to register an event only once
    const registerEventOnce = (event, handler) => {
      if (registeredEvents[event]) {
        console.log(`Event ${event} already registered in this setup call, skipping`);
        return;
      }
      
      socketService.on(event, handler);
      registeredEvents[event] = true;
      console.log(`Registered event: ${event}`);
      
      // Also track globally which events have been registered
      registeredEventsRef.current.add(event);
    };
    
    console.log("Registering core events with duplicate prevention");
    
    // Handle transcription data with more robust ID handling - IMPROVED VERSION
    registerEventOnce('transcription', (data) => {
      console.log('Received transcription data:', data);
      // Log the raw data for debugging
      console.log('Raw transcription data received:', JSON.stringify(data));
      
      if (!isMounted.current) return;
      
      // IMPORTANT: Check for video_id in data and update our ID if needed
      if (data.video_id && data.video_id !== currentVideoId) {
        console.log(`Updating video ID from ${currentVideoId} to server ID ${data.video_id}`);
        setCurrentVideoId(data.video_id);
      }
      
      // NEW: Check for file_id in data which is critical for uploads
      if (data.file_id && !data.video_id) {
        console.log(`Found file_id in transcription data: ${data.file_id}`);
        // Use file_id as video_id if no video_id present
        if (data.file_id !== currentVideoId) {
          console.log(`Updating video ID to match file_id: ${data.file_id}`);
          setCurrentVideoId(data.file_id);
        }
      }
      
      // Extract upload ID if present
      const extractedId = extractIdFromResponse(data);
      if (extractedId && extractedId !== currentVideoId) {
        console.log(`Found upload ID in transcription: ${extractedId}`);
        setCurrentVideoId(extractedId);
      }
      
      // Check if this transcription is for the current video
      const receivedVideoId = data.video_id || data.file_id; // NEW: Also check file_id
      if (receivedVideoId && currentVideoId) {
        // Special handling for YouTube IDs that might have different suffixes but same prefix
        const isYoutubeID = receivedVideoId.startsWith('youtube_') && currentVideoId.startsWith('youtube_');
        
        // New: Special handling for uploaded files
        const isUploadedFile = receivedVideoId.startsWith('upload_') && currentVideoId.startsWith('upload_');
        const isFileId = receivedVideoId.startsWith('file_') && currentVideoId.startsWith('file_');
        const isAudioId = receivedVideoId.startsWith('audio_') && currentVideoId.startsWith('audio_');
        
        if (isYoutubeID) {
          // Extract the timestamp part (first component after youtube_)
          const receivedTimePart = receivedVideoId.split('_')[1];
          const currentTimePart = currentVideoId.split('_')[1];
          
          // If time parts match or are very close (within 5 seconds), consider it a match
          if (receivedTimePart && currentTimePart && 
              Math.abs(parseInt(receivedTimePart) - parseInt(currentTimePart)) < 5000) {
            console.log(`Accepting similar video ID. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
            // Update current video ID to match what the server is using
            setCurrentVideoId(receivedVideoId);
          } else if (receivedVideoId !== currentVideoId) {
            console.log(`Ignoring transcription for different video. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
            return;
          }
        } 
        // New condition for uploaded files - MORE FLEXIBLE
        else if (isUploadedFile || isFileId || isAudioId) {
          // For uploaded files, be more lenient with ID matching - check timestamp parts
          if (isUploadedFile && currentVideoId.startsWith('file_')) {
            // Extract timestamps from both IDs
            const uploadTimestamp = receivedVideoId.split('_')[1];
            const fileTimestamp = currentVideoId.replace('file_', '');
            
            // If timestamps are very close, consider it a match
            if (uploadTimestamp && fileTimestamp && 
                Math.abs(parseInt(uploadTimestamp) - parseInt(fileTimestamp)) < 10000) {
              console.log(`Accepting upload/file ID time match. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
              // Update to server's ID
              setCurrentVideoId(receivedVideoId);
            } else {
              // Even if timestamps don't match closely, we'll accept it for uploads
              console.log(`Accepting upload ID loosely. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
              setCurrentVideoId(receivedVideoId);
            }
          } else {
            console.log(`Accepting upload/file ID. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
            // Update current video ID to match what the server is using
            if (receivedVideoId !== currentVideoId) {
              setCurrentVideoId(receivedVideoId);
            }
          }
        }
        // Check if one is file_ and the other is upload_ - IMPROVED
        else if ((receivedVideoId.startsWith('file_') && currentVideoId.startsWith('upload_')) || 
                (receivedVideoId.startsWith('upload_') && currentVideoId.startsWith('file_'))) {
          // More detailed check for file/upload ID relationships
          let fileId, uploadId;
          if (receivedVideoId.startsWith('file_')) {
            fileId = receivedVideoId;
            uploadId = currentVideoId;
          } else {
            fileId = currentVideoId;
            uploadId = receivedVideoId;
          }
          
          // Extract timestamps
          const fileTimestamp = fileId.replace('file_', '');
          const uploadParts = uploadId.split('_');
          if (uploadParts.length >= 2) {
            const uploadTimestamp = uploadParts[1];
            
            // Check if timestamps are related (within 10 seconds)
            if (fileTimestamp && uploadTimestamp && 
                Math.abs(parseInt(fileTimestamp) - parseInt(uploadTimestamp)) < 10000) {
              console.log(`Accepting related IDs with timestamp match. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
            } else {
              console.log(`Accepting related IDs without timestamp match. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
            }
          } else {
            console.log(`Accepting related IDs without validation. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
          }
          
          // Accept it either way and update to use the server's ID
          setCurrentVideoId(receivedVideoId);
        }
        else if (receivedVideoId !== currentVideoId) {
          // For non-YouTube IDs, still use strict comparison
          console.log(`Ignoring transcription for different video. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
          return;
        }
      }
      
      // Critical fix: Check for transcript in various formats the server might send
      let transcriptText = '';
      
      if (data.status === 'success' && data.transcript) {
        transcriptText = data.transcript;
      } else if (data.transcription) {
        transcriptText = data.transcription;
      } else if (data.text) {
        transcriptText = data.text;
      } else if (typeof data === 'string') {
        transcriptText = data;
      } else if (data.status === 'success' && typeof data.data === 'string') {
        transcriptText = data.data;
      }
      
      // Log the extracted text for debugging
      console.log('Extracted transcription text:', transcriptText ? transcriptText.substring(0, 100) + '...' : 'No text',
                 'Length:', transcriptText ? transcriptText.length : 0);
      
      // Only proceed if we actually have text content
      if (transcriptText && transcriptText.length > 0) {
        console.log(`Setting transcription, length: ${transcriptText.length} characters`);
        
        // Mark that we've received transcription data
        transcriptionReceivedRef.current = true;
        
        setTranscription(transcriptText);
        setTranscriptionStatus('loaded');
        setLoading(false);
        setTranscriptionLoading(false); // Critical to set this to false
        setProcessingYoutubeUrl(false);
        setProcessingStage('transcription-complete');
        
        // Clear any fallback timer
        if (fallbackTimerRef.current) {
          clearTimeout(fallbackTimerRef.current);
          fallbackTimerRef.current = null;
        }
        
        // Reset timestamp generation attempt counter
        timestampGenerationAttemptsRef.current = 0;
        
        // Reset API fallback attempts counter
        apiFallbackAttemptsRef.current = 0;
        
        // Wait a moment for the state to update before fetching visual data and timestamps
        setTimeout(() => {
          if (isMounted.current) {
            // Fetch data after a short delay to ensure state is updated
            if (functionsRef.current.fetchVisualData) {
              functionsRef.current.fetchVisualData(receivedVideoId || currentVideoId);
            }
            if (functionsRef.current.generateTimestampsFromTranscript) {
              functionsRef.current.generateTimestampsFromTranscript();
            }
            if (functionsRef.current.fetchSidebarTimestamps) {
              functionsRef.current.fetchSidebarTimestamps(receivedVideoId || currentVideoId);
            }
          }
        }, TRANSCRIPTION_DELAY);
        
        // Log after state update
        console.log('State updated with transcription text');
      } else if (data.status === 'error') {
        // Handle error transcription
        console.error('Transcription error received:', data.transcript || 'No error message');
        setError(data.transcript || 'Error receiving transcription');
        setTranscriptionStatus('error');
        setLoading(false);
        setTranscriptionLoading(false);
        setProcessingYoutubeUrl(false);
        setProcessingStage('transcription-error');
        
        // Try fallback if this is the first error
        if (apiFallbackAttemptsRef.current === 0) {
          console.log('Attempting REST API fallback after socket error');
          setTimeout(() => {
            if (isMounted.current && functionsRef.current.fetchTranscriptionViaREST) {
              functionsRef.current.fetchTranscriptionViaREST(currentVideoId);
            }
          }, 1000);
        }
      } else {
        console.warn('Received transcription event but no usable transcript data found in:', data);
        
        // New: if we receive an empty transcription but status is success, try REST API fallback
        if (data.status === 'success') {
          console.log('Received successful transcription event but no data, trying REST API fallback');
          setTimeout(() => {
            if (isMounted.current && functionsRef.current.fetchTranscriptionViaREST) {
              functionsRef.current.fetchTranscriptionViaREST(currentVideoId);
            }
          }, 1000);
        }
      }
    });
    
    // Handle transcription status updates with improved upload file handling
    registerEventOnce('transcription_status', (data) => {
      console.log('Transcription status update:', data);
      
      if (!isMounted.current) return;
      
      // NEW: Check for file_id in data which is critical for uploads
      if (data.file_id && !data.video_id) {
        console.log(`Found file_id in status data: ${data.file_id}`);
        // Use file_id as video_id if no video_id present
        if (data.file_id !== currentVideoId) {
          console.log(`Updating video ID to match file_id: ${data.file_id}`);
          setCurrentVideoId(data.file_id);
        }
      }
      
      // Check if this status update is for the current video
      const receivedVideoId = data.video_id || data.file_id; // NEW: Also check file_id
      if (receivedVideoId && currentVideoId) {
        // Same flexible ID check as in transcription handler
        const isYoutubeID = receivedVideoId.startsWith('youtube_') && currentVideoId.startsWith('youtube_');
        const isUploadedFile = receivedVideoId.startsWith('upload_') && currentVideoId.startsWith('upload_');
        const isFileId = receivedVideoId.startsWith('file_') && currentVideoId.startsWith('file_');
        const isAudioId = receivedVideoId.startsWith('audio_') && currentVideoId.startsWith('audio_');
        
        if (isYoutubeID) {
          const receivedTimePart = receivedVideoId.split('_')[1];
          const currentTimePart = currentVideoId.split('_')[1];
          
          if (receivedTimePart && currentTimePart && 
              Math.abs(parseInt(receivedTimePart) - parseInt(currentTimePart)) < 5000) {
            console.log(`Accepting status for similar video ID. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
            // Update current video ID to match server
            if (receivedVideoId !== currentVideoId) {
              setCurrentVideoId(receivedVideoId);
            }
          } else if (receivedVideoId !== currentVideoId) {
            console.log(`Ignoring status for different video. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
            return;
          }
        } 
        // New condition for uploaded files - IMPROVED
        else if (isUploadedFile || isFileId || isAudioId) {
          // For uploaded files, be more lenient with ID matching - check timestamp parts
          if (isUploadedFile && currentVideoId.startsWith('file_')) {
            // Extract timestamps from both IDs
            const uploadTimestamp = receivedVideoId.split('_')[1];
            const fileTimestamp = currentVideoId.replace('file_', '');
            
            // If timestamps are very close, consider it a match
            if (uploadTimestamp && fileTimestamp && 
                Math.abs(parseInt(uploadTimestamp) - parseInt(fileTimestamp)) < 10000) {
              console.log(`Accepting status for upload/file ID time match. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
              // Update to server's ID
              setCurrentVideoId(receivedVideoId);
            } else {
              // Even if timestamps don't match closely, we'll accept it for uploads
              console.log(`Accepting status for upload ID loosely. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
              setCurrentVideoId(receivedVideoId);
            }
          } else {
            console.log(`Accepting status for uploaded file. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
            if (receivedVideoId !== currentVideoId) {
              setCurrentVideoId(receivedVideoId);
            }
          }
        }
        // Check if one is file_ and the other is upload_ - IMPROVED
        else if ((receivedVideoId.startsWith('file_') && currentVideoId.startsWith('upload_')) || 
                (receivedVideoId.startsWith('upload_') && currentVideoId.startsWith('file_'))) {
          // More detailed check for file/upload ID relationships
          let fileId, uploadId;
          if (receivedVideoId.startsWith('file_')) {
            fileId = receivedVideoId;
            uploadId = currentVideoId;
          } else {
            fileId = currentVideoId;
            uploadId = receivedVideoId;
          }
          
          // Extract timestamps
          const fileTimestamp = fileId.replace('file_', '');
          const uploadParts = uploadId.split('_');
          if (uploadParts.length >= 2) {
            const uploadTimestamp = uploadParts[1];
            
            // Check if timestamps are related (within 10 seconds)
            if (fileTimestamp && uploadTimestamp && 
                Math.abs(parseInt(fileTimestamp) - parseInt(uploadTimestamp)) < 10000) {
              console.log(`Accepting status for related IDs with timestamp match. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
            } else {
              console.log(`Accepting status for related IDs without timestamp match. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
            }
          } else {
            console.log(`Accepting status for related IDs without validation. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
          }
          
          // Accept it either way and update to use the server's ID
          setCurrentVideoId(receivedVideoId);
        }
        else if (receivedVideoId !== currentVideoId) {
          console.log(`Ignoring status for different video. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
          return;
        }
      }
      
      // Update transcription status
      setTranscriptionStatus(data.status);
      
      // Update processing stage based on status
      switch (data.status) {
        case 'queued':
          setProcessingStage('transcription-queued');
          break;
        case 'downloading':
          setProcessingStage('downloading-video');
          break;
        case 'transcribing':
          setProcessingStage('transcribing-audio');
          break;
        case 'completed':
          setProcessingStage('transcription-complete');
          break;
        case 'error':
          setProcessingStage('transcription-error');
          break;
        default:
          setProcessingStage('processing');
      }
      
      // Handle specific status changes
      if (data.status === 'error') {
        // Check for timeout specific errors
        if (data.message && (data.message.includes('Timeout') || data.message.includes('timeout'))) {
          setError('Timeout transcribing YouTube video. Please try a shorter video or check your network connection.');
          // Also update the transcription with a user-friendly message
          setTranscription('Transcription timed out. This can happen with longer videos or slow network connections. Try a shorter clip or try again later.');
        } else {
          setError(data.message || 'An error occurred');
        }
        setLoading(false);
        setProcessingYoutubeUrl(false);
        setTranscriptionLoading(false);
        // Attempt fallback REST API after error
        if (currentVideoId && data.status === 'error') {
          console.log('Attempting fallback REST API after socket error');
          setTimeout(() => {
            if (isMounted.current && functionsRef.current.fetchTranscriptionViaREST) {
              functionsRef.current.fetchTranscriptionViaREST(currentVideoId);
            }
          }, 1000);
        }
      } else if (data.status === 'completed') {
        // Wait for the actual transcription data
        setTranscriptionLoading(false);
        
        // Reset timestamp generation attempt counter
        timestampGenerationAttemptsRef.current = 0;
        
        // If we already have transcription but haven't generated timestamps yet,
        // this is a good time to try generating timestamps
        if (transcription && (!timestamps || timestamps.length === 0) && !hasGeneratedTimestamps) {
          console.log('Transcription status completed, generating timestamps from existing transcription');
          // Schedule the generation with a delay to allow state updates
          setTimeout(() => {
            if (isMounted.current && functionsRef.current.generateTimestampsFromTranscript) {
              functionsRef.current.generateTimestampsFromTranscript();
            }
          }, TRANSCRIPTION_DELAY);
        }
        
        // If we haven't received transcription data yet but the status is completed,
        // try to fetch it via REST API
        if (!transcriptionReceivedRef.current) {
          console.log('Transcription status completed but no data received yet, trying REST API');
          setTimeout(() => {
            if (isMounted.current && functionsRef.current.fetchTranscriptionViaREST && currentVideoId) {
              functionsRef.current.fetchTranscriptionViaREST(currentVideoId);
            }
          }, 500); // Short delay before trying REST
        }
        
        // For uploads, be more proactive with completed status
        if (currentVideoId && 
           (currentVideoId.startsWith('file_') || 
            currentVideoId.startsWith('upload_') || 
            currentVideoId.startsWith('audio_'))) {
          
          console.log('Upload file transcription completed, fetching via REST API');
          // Short delay before trying REST
          setTimeout(() => {
            if (isMounted.current && functionsRef.current.fetchTranscriptionViaREST) {
              functionsRef.current.fetchTranscriptionViaREST(currentVideoId);
            }
          }, 1000);
        }
      } else if (data.status === 'downloading' || data.status === 'transcribing' || data.status === 'queued') {
        // These are progress states, keep loading true
        setTranscriptionLoading(true);
        setError(null); // Clear any previous errors
      }
    });
    
    // Add a specific socket event for uploaded files - GREATLY IMPROVED
    registerEventOnce('upload_transcription_complete', (data) => {
      console.log('Upload transcription complete notification:', data);
      
      if (!isMounted.current) return;
      
      // Update our ID if needed
      if (data.video_id && data.video_id !== currentVideoId) {
        console.log(`Updating video ID from ${currentVideoId} to server ID ${data.video_id}`);
        setCurrentVideoId(data.video_id);
      }
      
      // NEW: Check for file_id in data which is critical for uploads
      if (data.file_id && !data.video_id) {
        console.log(`Found file_id in notification: ${data.file_id}`);
        // Use file_id as video_id if no video_id present
        if (data.file_id !== currentVideoId) {
          console.log(`Updating video ID to match file_id: ${data.file_id}`);
          setCurrentVideoId(data.file_id);
        }
      }
      
      // Extract upload ID if present
      const extractedId = extractIdFromResponse(data);
      if (extractedId && extractedId !== currentVideoId) {
        console.log(`Found upload ID in notification: ${extractedId}`);
        setCurrentVideoId(extractedId);
      }
      
      // Try multiple potential IDs for better compatibility
      const potentialIds = [];
      
      // First, add any explicitly provided IDs
      if (data.video_id) potentialIds.push(data.video_id);
      if (data.file_id) potentialIds.push(data.file_id);
      if (extractedId) potentialIds.push(extractedId);
      
      // Always add the current video ID as a fallback
      if (currentVideoId && !potentialIds.includes(currentVideoId)) {
        potentialIds.push(currentVideoId);
      }
      
      // Also try to derive related IDs
      if (currentVideoId.startsWith('file_')) {
        const fileTimestamp = currentVideoId.replace('file_', '');
        if (fileTimestamp) {
          // Try derived upload format
          const derivedUploadId = `upload_${fileTimestamp}`;
          if (!potentialIds.includes(derivedUploadId)) {
            potentialIds.push(derivedUploadId);
          }
        }
      } else if (currentVideoId.startsWith('upload_')) {
        const parts = currentVideoId.split('_');
        if (parts.length >= 2) {
          const uploadTimestamp = parts[1];
          if (uploadTimestamp) {
            // Try derived file format
            const derivedFileId = `file_${uploadTimestamp}`;
            if (!potentialIds.includes(derivedFileId)) {
              potentialIds.push(derivedFileId);
            }
          }
        }
      }
      
      // Try to fetch the transcription for each potential ID
      const tryFetchingForIds = async () => {
        if (!isMounted.current) return;
        
        for (const id of potentialIds) {
          if (!id) continue;
          
          console.log(`Trying to fetch transcription for ID: ${id}`);
          if (functionsRef.current.fetchTranscriptionViaREST) {
            const success = await functionsRef.current.fetchTranscriptionViaREST(id);
            if (success) {
              console.log(`Successfully fetched transcription for ID: ${id}`);
              return; // Stop trying if we succeed
            }
          }
          // Add a small delay between attempts
          await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        console.log('None of the potential IDs worked for fetching transcription');
      };
      
      // Start the fetch process
      setTimeout(() => {
        tryFetchingForIds();
      }, 500);
    });
    
    // Handle AI responses
    registerEventOnce('ai_response', (data) => {
      console.log('Received AI response:', data);
      
      if (!isMounted.current) return;
      
      // Check if this response is for the current video with flexible ID check
      const receivedVideoId = data.video_id;
      if (receivedVideoId && currentVideoId) {
        const isRelated = areVideoIdsRelated(receivedVideoId, currentVideoId);
        
        if (!isRelated && receivedVideoId !== currentVideoId) {
          console.log(`Ignoring AI response for different video. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
          return;
        }
      }
      
      if (data.answer) {
        addMessage('ai', data.answer);
        // Check for timestamps in the response
        if (data.timestamps && data.timestamps.length > 0) {
          // Always store timestamps in the sidebar
          setSidebarTimestamps(prev => {
            // Filter out duplicates and merge with existing timestamps
            const existingTimes = new Set(prev.map(ts => ts.time));
            const newTimestamps = data.timestamps.filter(ts => !existingTimes.has(ts.time));
            return [...prev, ...newTimestamps];
          });
          // Store the latest timestamp for auto-navigation
          setSelectedTimestamp(data.timestamps[0]);
          // Store all visible timestamps
          setVisibleTimestamps(data.timestamps);
          
          // If we didn't have timestamps before, use these
          if (!timestamps || timestamps.length === 0) {
            setTimestamps(data.timestamps);
            setHasGeneratedTimestamps(true);
          }
        }
        // Immediately set thinking to false
        setIsAIThinking(false);
        // Clear any existing thinking timeout
        if (thinkingTimeoutRef.current) {
          clearTimeout(thinkingTimeoutRef.current);
          thinkingTimeoutRef.current = null;
        }
      }
    });
    
    // Handle response complete
    registerEventOnce('response_complete', (data) => {
      console.log('Response complete signal received:', data);
      
      if (!isMounted.current) return;
      
      // Force thinking state to false
      setIsAIThinking(false);
      // Clear any existing thinking timeout
      if (thinkingTimeoutRef.current) {
        clearTimeout(thinkingTimeoutRef.current);
        thinkingTimeoutRef.current = null;
      }
    });
    
    // Handle socket errors
    registerEventOnce('error', (data) => {
      console.error('Socket error:', data);
      
      if (!isMounted.current) return;
      
      // Check if this error is for the current video with flexible ID check
      const receivedVideoId = data.video_id;
      if (receivedVideoId && currentVideoId) {
        const isRelated = areVideoIdsRelated(receivedVideoId, currentVideoId);
        
        if (!isRelated && receivedVideoId !== currentVideoId) {
          console.log(`Ignoring error for different video. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
          return;
        }
      }
      
      // Special handling for timeout errors
      if (data.message && (data.message.includes('Timeout') || data.message.includes('timeout'))) {
        setError('Timeout transcribing YouTube video. Please try a shorter video or check your network connection.');
        // Also update the transcription with a user-friendly message
        setTranscription('Transcription timed out. This can happen with longer videos or slow network connections. Try a shorter clip or try again later.');
        // Attempt REST API fallback after timeout
        if (currentVideoId) {
          console.log('Attempting REST API fallback after timeout');
          setTimeout(() => {
            if (isMounted.current && functionsRef.current.fetchTranscriptionViaREST) {
              functionsRef.current.fetchTranscriptionViaREST(currentVideoId);
            }
          }, 1000);
        }
      } else {
        setError(data.message || 'An error occurred with the connection');
      }
      
      setIsAIThinking(false);
      setLoading(false);
      setProcessingYoutubeUrl(false);
      setTranscriptionLoading(false);
      setProcessingStage('error');
      
      // Clear any existing thinking timeout
      if (thinkingTimeoutRef.current) {
        clearTimeout(thinkingTimeoutRef.current);
        thinkingTimeoutRef.current = null;
      }
      
      // Also update transcription status
      setTranscriptionStatus('error');
    });
    
    // Handle connection established
    registerEventOnce('connection_established', (data) => {
      console.log('Connection established:', data);
      
      if (!isMounted.current) return;
      
      setConnectionStatus('connected');
      socketConnectedRef.current = true;
      
      // Reset retry counter on successful connection
      connectionRetryCountRef.current = 0;
    });
    
    // Handle connection errors
    registerEventOnce('connect_error', (error) => {
      console.error('Socket connection error:', error);
      
      if (!isMounted.current) return;
      
      setConnectionStatus('error');
      socketConnectedRef.current = false;
      
      // Increment retry counter
      connectionRetryCountRef.current++;
      
      // Auto-retry with exponential backoff
      const delay = Math.min(30000, 1000 * Math.pow(2, connectionRetryCountRef.current));
      console.log(`Will attempt reconnection in ${delay}ms (attempt #${connectionRetryCountRef.current})`);
      
      setTimeout(() => {
        if (isMounted.current) {
          console.log('Attempting socket reconnection...');
          socketService.reconnect();
        }
      }, delay);
    });
    
    // Handle disconnect
    registerEventOnce('disconnect', (reason) => {
      console.log(`Socket disconnected: ${reason}`);
      
      if (!isMounted.current) return;
      
      setConnectionStatus('disconnected');
      socketConnectedRef.current = false;
      
      // Try immediate reconnection if it wasn't an intentional disconnect
      if (reason !== 'io client disconnect' && isMounted.current) {
        console.log('Attempting immediate reconnection...');
        socketService.reconnect();
      }
    });
    
    // Now register the timestamps events separately
    console.log("Registering timestamps events after other events are set up");
    
    // Timestamps data handler
    registerEventOnce('timestamps_data', (data) => {
      console.log('Received timestamps data:', data);
      
      if (!isMounted.current) return;
      
      // Check if this data is for the current video with flexible ID check
      const receivedVideoId = data.video_id;
      if (receivedVideoId && currentVideoId) {
        const isRelated = areVideoIdsRelated(receivedVideoId, currentVideoId);
        
        if (!isRelated && receivedVideoId !== currentVideoId) {
          console.log(`Ignoring timestamps for different video. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
          return;
        }
      }
      
      // Process the timestamps array
      if (data.timestamps && Array.isArray(data.timestamps)) {
        // Store raw timestamps for VideoPlayer integration
        setTimestamps(data.timestamps);
        // Convert backend timestamp format to frontend format for sidebar
        const formattedTimestamps = convertTimestampFormat(data.timestamps);
        setSidebarTimestamps(formattedTimestamps);
        setHasGeneratedTimestamps(true);
      }
    });
    
    // Timestamps available handler
    registerEventOnce('timestamps_available', (data) => {
      console.log('Timestamps available notification:', data);
      
      if (!isMounted.current) return;
      
      // Check if this notification is for the current video with flexible ID check
      const receivedVideoId = data.video_id;
      if (receivedVideoId && currentVideoId) {
        const isRelated = areVideoIdsRelated(receivedVideoId, currentVideoId);
        
        if (!isRelated && receivedVideoId !== currentVideoId) {
          console.log(`Ignoring timestamps notification for different video. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
          return;
        }
      }
      
      // Fetch the timestamps
      if (functionsRef.current.fetchSidebarTimestamps) {
        functionsRef.current.fetchSidebarTimestamps(receivedVideoId || currentVideoId);
      }
    });
    
    // Visual analysis events handlers (similar patterns to transcription)
    registerEventOnce('visual_analysis_status', (data) => {
      console.log('Visual analysis status update:', data);
      
      if (!isMounted.current) return;
      
      // Check if this notification is for the current video
      const receivedVideoId = data.video_id;
      if (receivedVideoId && currentVideoId && receivedVideoId !== currentVideoId) {
        console.log(`Ignoring visual analysis status for different video. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
        return;
      }
      
      // Handle the completed status to update our state
      if (data.status === 'completed') {
        setVisualAnalysisInProgress(false);
        setVisualAnalysisAvailable(true);
        
        // Try to fetch the visual data
        if (functionsRef.current.fetchVisualData) {
          functionsRef.current.fetchVisualData(receivedVideoId || currentVideoId);
        }
      } else if (data.status === 'error') {
        setVisualAnalysisInProgress(false);
        console.error('Visual analysis error:', data.message || 'Unknown error');
      }
    });
    
    // Topics data handler
    registerEventOnce('topics_data', (data) => {
      console.log('Received topics data:', data);
      
      if (!isMounted.current) return;
      
      // Check if this data is for the current video
      const receivedVideoId = data.video_id;
      if (receivedVideoId && currentVideoId && receivedVideoId !== currentVideoId) {
        console.log(`Ignoring topics data for different video. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
        return;
      }
      
      if (data.topics && Array.isArray(data.topics)) {
        setVideoTopics(data.topics);
      } else {
        // If no topics or invalid format, ensure we have an empty array
        setVideoTopics([]);
      }
    });
    
    // Highlights data handler
    registerEventOnce('highlights_data', (data) => {
      console.log('Received highlights data:', data);
      
      if (!isMounted.current) return;
      
      // Check if this data is for the current video
      const receivedVideoId = data.video_id;
      if (receivedVideoId && currentVideoId && receivedVideoId !== currentVideoId) {
        console.log(`Ignoring highlights data for different video. Current: ${currentVideoId}, Received: ${receivedVideoId}`);
        return;
      }
      
      if (data.highlights && Array.isArray(data.highlights)) {
        setVideoHighlights(data.highlights);
      } else {
        // If no highlights or invalid format, ensure we have an empty array
        setVideoHighlights([]);
      }
    });
    
    // Set up heartbeat to keep connection alive
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }
    
    heartbeatIntervalRef.current = setInterval(() => {
      if (socketService.isSocketConnected() && isMounted.current) {
        console.log('Sending heartbeat');
        socketService.sendEcho({ message: "Heartbeat", timestamp: Date.now() });
      } else if (isMounted.current) {
        console.log('Socket disconnected, attempting to reconnect...');
        socketConnectedRef.current = false;
        socketService.connect()
          .then(() => {
            console.log('Socket reconnected successfully');
            setConnectionStatus('connected');
            socketConnectedRef.current = true;
            socketService.registerTab(tabId);
            // Re-setup socket listeners after reconnection
            setupSocketListeners();
          })
          .catch(err => {
            console.error('Socket reconnection failed:', err);
            setConnectionStatus(`error: ${err.message}`);
            socketConnectedRef.current = false;
          });
      }
    }, 30000); // every 30 seconds
    
    // Add a manual ping function for actively checking status during transcription
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }
    
    pingIntervalRef.current = setInterval(() => {
      if (socketService.isSocketConnected() && socketConnectedRef.current && transcriptionLoading && isMounted.current) {
        console.log('📍 Pinging server for status while transcription is loading');
        socketService.emit('ping', { tab_id: tabId, video_id: currentVideoId });
      }
    }, 15000); // Every 15 seconds
    
    return true;
  }, [
    tabId, 
    currentVideoId,
    transcription,
    transcriptionLoading,
    timestamps,
    hasGeneratedTimestamps,
    areVideoIdsRelated,
    convertTimestampFormat,
    addMessage,
    extractIdFromResponse
  ]);
  // Store the setupSocketListeners implementation for later use
  functionsRef.current.setupSocketListeners = setupSocketListeners;
  // Ask about visible content
  const askAboutVisibleContent = useCallback(async (question) => {
    if (!currentVideoId) {
      throw new Error('No video selected');
    }
    
    if (!socketService.isSocketConnected() || !socketConnectedRef.current) {
      // Try reconnecting first
      try {
        console.log('Attempting to reconnect socket before visual question...');
        await socketService.connect();
        await new Promise(resolve => setTimeout(resolve, SOCKET_RECONNECT_DELAY));
        
        if (!socketService.isSocketConnected()) {
          throw new Error('Socket connection failed');
        }
        
        socketConnectedRef.current = true;
        socketService.registerTab(tabId);
        if (functionsRef.current.setupSocketListeners) {
          functionsRef.current.setupSocketListeners();
        }
      } catch (reconnectError) {
        console.error('Socket reconnection failed:', reconnectError);
        throw new Error('Video or connection not available');
      }
    }
    
    try {
      // Add user message to conversation
      addMessage('user', question);
      
      // Set thinking state
      setIsAIThinking(true);
      
      // Use specialized visual question method
      await socketService.askVisualQuestion(currentVideoId, question);
      
      // Set timeout to reset thinking state if no response
      thinkingTimeoutRef.current = setTimeout(() => {
        console.log(`Forcing thinking state to false after 30s timeout for visual question`);
        if (isMounted.current) {
          setIsAIThinking(false);
          addMessage('ai', 'Sorry, I didn\'t receive a response for your visual question. Please try again.');
        }
      }, 30000);
      
      return true;
    } catch (err) {
      console.error('Error asking about visible content:', err);
      if (isMounted.current) {
        setIsAIThinking(false);
        
        // Add fallback message
        addMessage('ai', 'I\'m having trouble analyzing the visual content right now. Please try again later.');
      }
      
      throw err;
    }
  }, [currentVideoId, tabId, addMessage]);
  // Send message to AI
  const sendMessageToAI = useCallback(async (message) => {
    if (!socketService.isSocketConnected() || !socketConnectedRef.current) {
      console.error('Socket connection not available');
      
      // Try reconnecting first
      try {
        console.log('Attempting to reconnect socket before sending message...');
        await socketService.connect();
        await new Promise(resolve => setTimeout(resolve, SOCKET_RECONNECT_DELAY));
        
        if (!socketService.isSocketConnected()) {
          throw new Error('Socket connection failed');
        }
        
        socketConnectedRef.current = true;
        socketService.registerTab(tabId);
        if (functionsRef.current.setupSocketListeners) {
          functionsRef.current.setupSocketListeners();
        }
      } catch (reconnectError) {
        console.error('Socket reconnection failed:', reconnectError);
        if (isMounted.current) {
          setError('Connection to AI server is not available. Please refresh the page and try again.');
        }
        throw new Error('Socket connection not available after reconnection attempt');
      }
    }
    
    try {
      // Add user message to conversation
      addMessage('user', message);
      
      // Set thinking state
      setIsAIThinking(true);
      
      // Clear any existing thinking timeout
      if (thinkingTimeoutRef.current) {
        clearTimeout(thinkingTimeoutRef.current);
      }
      
      // Analyze question to detect timestamp queries
      const isTimestampQuestion = detectTimestampQuestion(message);
      
      if (isTimestampQuestion) {
        // Extract timestamp from question if present
        const timestamp = extractTimestampFromQuestion(message);
        
        if (timestamp) {
          console.log(`Detected timestamp query for time: ${timestamp.formatted}`);
          
          // Handle timestamp question by navigating to the timestamp
          navigateToTimestamp({
            time: timestamp.totalSeconds,
            time_formatted: timestamp.formatted
          });
        }
      }
      
      // Send via socket with current videoId and conversations for context
      await socketService.askAIWithContext(tabId, message, currentVideoId, conversations);
      
      // Add timeout to reset thinking state if no response
      thinkingTimeoutRef.current = setTimeout(() => {
        console.log(`Forcing thinking state to false after 30s timeout`);
        if (isMounted.current) {
          setIsAIThinking(false);
          addMessage('ai', 'Sorry, I didn\'t receive a response in time. Please try again.');
        }
      }, 30000);
      
      return true;
    } catch (err) {
      console.error('Error sending message to AI:', err);
      
      if (!isMounted.current) return false;
      
      setError('Failed to get AI response');
      setIsAIThinking(false);
      
      // Clear any existing thinking timeout
      if (thinkingTimeoutRef.current) {
        clearTimeout(thinkingTimeoutRef.current);
        thinkingTimeoutRef.current = null;
      }
      
      // Add fallback error message to the conversation
      addMessage('ai', 'Sorry, I encountered an error while processing your request. Please try again later.');
      
      throw err;
    }
  }, [
    tabId,
    currentVideoId,
    conversations,
    addMessage,
    detectTimestampQuestion,
    extractTimestampFromQuestion,
    navigateToTimestamp
  ]);
  // Reset all video-related state
  const resetVideoState = useCallback(() => {
    console.log('Resetting video state');
    // Reset all state related to the current video
    setVideoData(null);
    setTranscription('');
    setError(null);
    setCurrentVideoId(null);
    setTranscriptionStatus('idle');
    setProcessingYoutubeUrl(false);
    setTranscriptionLoading(false);
    setVisualAnalysisAvailable(false);
    setVisualAnalysisInProgress(false);
    setProcessingStage('idle');
    // Reset enhanced feature states
    setVideoTopics([]);
    setVideoHighlights([]);
    setVisualSummary(null);
    setSelectedTimestamp(null);
    setDetectedScenes([]);
    setKeyFrames([]);
    setCurrentView('chat');
    // Reset visual understanding states
    setVideoObjects([]);
    setVisualContext(null);
    setTimestampContent({});
    setVisibleTimestamps([]);
    setSidebarTimestamps([]);
    setTimestamps([]);
    setActiveTimestampIndex(-1);
    setCurrentTime(0);
    // Reset timestamp fetching state
    setTimestampFetchAttempts(0);
    setTimestampFetchFailed(false);
    setHasGeneratedTimestamps(false);
    // Reset timestamp generation attempt counter
    timestampGenerationAttemptsRef.current = 0;
    // Reset transcription received flag
    transcriptionReceivedRef.current = false;
    // Also clear conversations if needed
    setConversations([]);
    // Clear any active timeouts and intervals
    if (thinkingTimeoutRef.current) {
      clearTimeout(thinkingTimeoutRef.current);
      thinkingTimeoutRef.current = null;
    }
    if (timestampHandlerRef.current) {
      clearTimeout(timestampHandlerRef.current);
      timestampHandlerRef.current = null;
    }
    if (fallbackTimerRef.current) {
      clearTimeout(fallbackTimerRef.current);
      fallbackTimerRef.current = null;
    }
    if (delayedTimestampGenRef.current) {
      clearTimeout(delayedTimestampGenRef.current);
      delayedTimestampGenRef.current = null;
    }
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
    // Cancel active transcription if exists
    if (activeTranscriptionRef.current && typeof activeTranscriptionRef.current.cancel === 'function') {
      try {
        activeTranscriptionRef.current.cancel();
      } catch (e) {
        console.error('Error cancelling active transcription:', e);
      }
      activeTranscriptionRef.current = null;
    }
    // Reset active API request
    activeApiRequestRef.current = null;
    
    // Reset fallback attempts counter
    apiFallbackAttemptsRef.current = 0;
    // Force thinking state to false
    setIsAIThinking(false);
    // Reset any socket-related state
    if (socketService.isSocketConnected() && socketConnectedRef.current) {
      // Re-register the tab to reset server state
      socketService.registerTab(tabId);
    }
    // Reset visual data loading ref
    visualDataLoadingRef.current = false;
    console.log("Video state has been reset");
  }, [tabId]);
  // Process YouTube URL with improved reliability and fallback mechanisms
  const processYoutubeUrl = useCallback(async (url) => {
    try {
      // Helper function for more reliable YouTube URL validation
      const isValidYoutubeUrl = (url) => {
        try {
          // Check for basic YouTube domains
          if (!url.includes('youtube.com') && !url.includes('youtu.be')) {
            return false;
          }
          // Handle standard youtube.com URLs
          if (url.includes('youtube.com/watch')) {
            const urlParams = new URLSearchParams(url.split('?')[1]);
            return urlParams.has('v'); // Must have a video ID parameter
          }
          
          // Handle youtu.be short URLs
          if (url.includes('youtu.be/')) {
            const path = url.split('youtu.be/')[1];
            return path.length > 0; // Must have something after youtu.be/
          }
          
          // Handle YouTube shorts
          if (url.includes('youtube.com/shorts/')) {
            const path = url.split('youtube.com/shorts/')[1];
            return path.length > 0; // Must have something after shorts/
          }
          
          return false;
        } catch (error) {
          console.error("Error validating YouTube URL:", error);
          return false;
        }
      };
      // Reset API fallback attempts counter
      apiFallbackAttemptsRef.current = 0;
      
      // Reset timestamp generation attempts
      timestampGenerationAttemptsRef.current = 0;
      
      // Reset transcription received flag
      transcriptionReceivedRef.current = false;
      
      // Check server health before attempting, but proceed anyway
      const isHealthy = await checkServerHealth();
      if (!isHealthy) {
        console.log("Server health check indicated server might be unavailable, proceeding anyway");
      }
      
      // Reset state before starting
      setLoading(true);
      setError(null);
      setTranscription(''); // Clear previous transcription
      setTranscriptionStatus('idle');
      setProcessingYoutubeUrl(true);
      setTranscriptionLoading(true);
      setVisualAnalysisAvailable(false);
      setVisualAnalysisInProgress(false);
      setProcessingStage('starting');
      setHasGeneratedTimestamps(false);
      
      // Reset enhanced feature states
      setVideoTopics([]);
      setVideoHighlights([]);
      setVisualSummary(null);
      setSelectedTimestamp(null);
      setDetectedScenes([]);
      setKeyFrames([]);
      setSidebarTimestamps([]);
      setTimestamps([]);
      setActiveTimestampIndex(-1);
      
      // Reset visual understanding states
      setVideoObjects([]);
      setVisualContext(null);
      setTimestampContent({});
      setVisibleTimestamps([]);
      
      // Make sure URL is complete and valid
      if (!url.startsWith('http')) {
        url = 'https://' + url;
      }
      
      // Validate the YouTube URL
      if (!isValidYoutubeUrl(url)) {
        throw new Error('Invalid YouTube URL format. Please use a standard YouTube video or shorts URL.');
      }
      
      // Generate a unique video ID for this request
      // Use timestamp without milliseconds to make IDs more consistent
      const timestamp = Math.floor(Date.now() / 1000);
      const randomId = Math.random().toString(36).substring(2, 8);
      const videoId = `youtube_${timestamp}_${randomId}`;
      console.log(`Generated new video ID for YouTube URL: ${videoId}`);
      
      // Set as current video ID immediately
      setCurrentVideoId(videoId);
      // Set up video data immediately so user can navigate to the analysis page
      setVideoData({ youtubeUrl: url, videoId: videoId });
      
      // Get auth headers
      const headers = getAuthHeaders();
      
      // Set up monitoring for stalled transcription
      setupTranscriptionTimeout(videoId);
      console.log(`Processing YouTube URL via socket for video ID: ${videoId}`);
      setProcessingStage('socket-connecting');
      
      // Ensure socket is connected
      if (!socketService.isSocketConnected() || !socketConnectedRef.current) {
        try {
          await socketService.connect();
          console.log("Socket connected successfully");
          socketConnectedRef.current = true;
          
          // Wait for socket to be fully ready
          await new Promise(resolve => setTimeout(resolve, SOCKET_RECONNECT_DELAY));
          
          socketService.registerTab(tabId);
          if (functionsRef.current.setupSocketListeners) {
            functionsRef.current.setupSocketListeners();
          }
        } catch (connectError) {
          console.error("Failed to connect to socket:", connectError);
          socketConnectedRef.current = false;
          // Continue with direct API approach
        }
      }
      
      // Navigate to analysis page immediately after setting up initial state
      // This is critical - it needs to happen before the long-running processes
      if (functionsRef.current.navigateToAnalysisPage) {
        console.log('Navigating to analysis page now');
        functionsRef.current.navigateToAnalysisPage();
      }
      
      // Use the socket to send the YouTube analysis request
      if (socketService.isSocketConnected() && socketConnectedRef.current) {
        try {
          console.log('Sending YouTube analysis request with tabId:', tabId);
          setProcessingStage('socket-analysis');
          
          // Emit the socket request with all required parameters
          socketService.emit('analyze_youtube', {
            url: url,
            youtube_url: url,
            tabId: tabId,
            tab_id: tabId,
            video_id: videoId
          });
          
          // Also try the alternative event name for compatibility
          setTimeout(() => {
            if (socketService.isSocketConnected()) {
              socketService.emit('process_youtube', {
                url: url,
                youtube_url: url,
                tabId: tabId,
                tab_id: tabId,
                video_id: videoId
              });
            }
          }, 500);
          
          console.log(`Processing YouTube URL via socket: ${url}`);
          
          // Return immediately with success to allow navigation
          return { 
            success: true, 
            youtubeUrl: url, 
            videoId: videoId,
            message: "Processing started"
          };
        } catch (socketError) {
          console.error('Error analyzing YouTube via socket:', socketError);
          
          // Try REST API fallback immediately after socket error
          console.log('Attempting REST API fallback after socket error');
          setProcessingStage('socket-error-fallback');
          
          try {
            // Try multiple endpoint patterns
            const endpoints = [
              { url: '/api/v1/videos/youtube', method: 'post', data: { youtube_url: url, tab_id: tabId, video_id: videoId }},
              { url: '/api/videos/youtube', method: 'post', data: { youtube_url: url, tab_id: tabId, video_id: videoId }},
              { url: '/api/v1/analyze/youtube', method: 'post', data: { url: url, tab_id: tabId, video_id: videoId }},
              { url: `/api/v1/videos/youtube?url=${encodeURIComponent(url)}&tab_id=${tabId}&video_id=${videoId}`, method: 'get' }
            ];
            
            for (const endpoint of endpoints) {
              try {
                console.log(`Trying REST API endpoint: ${endpoint.url}`);
                let response;
                
                if (endpoint.method === 'post') {
                  response = await api.post(endpoint.url, endpoint.data, { headers, timeout: API_TIMEOUT });
                } else {
                  response = await api.get(endpoint.url, { headers, timeout: API_TIMEOUT });
                }
                
                if (response.data && (response.data.success || response.status === 200)) {
                  // If server returned a different video ID, update our currentVideoId
                  if (response.data.video_id && response.data.video_id !== videoId && isMounted.current) {
                    console.log(`Endpoint ${endpoint.url} gave different video ID. Updating from ${videoId} to ${response.data.video_id}`);
                    setCurrentVideoId(response.data.video_id);
                    setVideoData(prev => ({ ...prev, videoId: response.data.video_id }));
                  }
                  
                  // Return success
                  return { 
                    success: true, 
                    youtubeUrl: url, 
                    videoId: response.data.video_id || videoId,
                    endpoint: endpoint.url
                  };
                }
              } catch (endpointError) {
                console.log(`Endpoint ${endpoint.url} failed:`, endpointError.message);
                
                // Try to handle auth errors
                if (endpointError.response && endpointError.response.status === 401) {
                  const refreshed = await handleApiError(endpointError, endpoint.url);
                  if (refreshed) {
                    // Try again with refreshed token
                    try {
                      let retryResponse;
                      const updatedHeaders = getAuthHeaders();
                      
                      if (endpoint.method === 'post') {
                        retryResponse = await api.post(endpoint.url, endpoint.data, { 
                          headers: updatedHeaders, 
                          timeout: API_TIMEOUT 
                        });
                      } else {
                        retryResponse = await api.get(endpoint.url, { 
                          headers: updatedHeaders, 
                          timeout: API_TIMEOUT 
                        });
                      }
                      
                      if (retryResponse.data && (retryResponse.data.success || retryResponse.status === 200)) {
                        // If server returned a different video ID, update our currentVideoId
                        if (retryResponse.data.video_id && retryResponse.data.video_id !== videoId && isMounted.current) {
                          console.log(`Endpoint ${endpoint.url} gave different video ID after token refresh. Updating from ${videoId} to ${retryResponse.data.video_id}`);
                          setCurrentVideoId(retryResponse.data.video_id);
                          setVideoData(prev => ({ ...prev, videoId: retryResponse.data.video_id }));
                        }
                        
                        // Return success
                        return { 
                          success: true, 
                          youtubeUrl: url, 
                          videoId: retryResponse.data.video_id || videoId,
                          endpoint: endpoint.url
                        };
                      }
                    } catch (retryError) {
                      console.log(`Retry after token refresh failed for ${endpoint.url}:`, retryError.message);
                    }
                  }
                }
              }
            }
            
            // All endpoints failed, but continue anyway
            console.log('All REST API fallback attempts failed, but continuing with navigation');
          } catch (fallbackError) {
            console.error('Fallback orchestration failed:', fallbackError);
          }
        }
      } else {
        // Fallback to REST API directly if socket not connected
        console.log('Socket not connected, using REST API directly');
        setProcessingStage('rest-direct');
        
        try {
          // Try multiple endpoint patterns
          const endpoints = [
            { url: '/api/v1/videos/youtube', method: 'post', data: { youtube_url: url, tab_id: tabId, video_id: videoId }},
            { url: '/api/videos/youtube', method: 'post', data: { youtube_url: url, tab_id: tabId, video_id: videoId }},
            { url: '/api/v1/analyze/youtube', method: 'post', data: { url: url, tab_id: tabId, video_id: videoId }},
            { url: `/api/v1/videos/youtube?url=${encodeURIComponent(url)}&tab_id=${tabId}&video_id=${videoId}`, method: 'get' }
          ];
          
          for (const endpoint of endpoints) {
            try {
              console.log(`Trying REST API endpoint: ${endpoint.url}`);
              let response;
              
              if (endpoint.method === 'post') {
                response = await api.post(endpoint.url, endpoint.data, { headers, timeout: API_TIMEOUT });
              } else {
                response = await api.get(endpoint.url, { headers, timeout: API_TIMEOUT });
              }
              
              if (response.data && (response.data.success || response.status === 200)) {
                // If server returned a different video ID, update our currentVideoId
                if (response.data.video_id && response.data.video_id !== videoId && isMounted.current) {
                  console.log(`Endpoint ${endpoint.url} gave different video ID. Updating from ${videoId} to ${response.data.video_id}`);
                  setCurrentVideoId(response.data.video_id);
                  setVideoData(prev => ({ ...prev, videoId: response.data.video_id }));
                }
                
                return { 
                  success: true, 
                  youtubeUrl: url, 
                  videoId: response.data.video_id || videoId,
                  endpoint: endpoint.url
                };
              }
            } catch (endpointError) {
              console.log(`Endpoint ${endpoint.url} failed:`, endpointError.message);
              
              // Try to handle auth errors
              if (endpointError.response && endpointError.response.status === 401) {
                const refreshed = await handleApiError(endpointError, endpoint.url);
                if (refreshed) {
                  // Try again with refreshed token
                  try {
                    let retryResponse;
                    const updatedHeaders = getAuthHeaders();
                    
                    if (endpoint.method === 'post') {
                      retryResponse = await api.post(endpoint.url, endpoint.data, { 
                        headers: updatedHeaders, 
                        timeout: API_TIMEOUT 
                      });
                    } else {
                      retryResponse = await api.get(endpoint.url, { 
                        headers: updatedHeaders, 
                        timeout: API_TIMEOUT 
                      });
                    }
                    
                    if (retryResponse.data && (retryResponse.data.success || retryResponse.status === 200)) {
                      // If server returned a different video ID, update our currentVideoId
                      if (retryResponse.data.video_id && retryResponse.data.video_id !== videoId && isMounted.current) {
                        console.log(`Endpoint ${endpoint.url} gave different video ID after token refresh. Updating from ${videoId} to ${retryResponse.data.video_id}`);
                        setCurrentVideoId(retryResponse.data.video_id);
                        setVideoData(prev => ({ ...prev, videoId: retryResponse.data.video_id }));
                      }
                      
                      return { 
                        success: true, 
                        youtubeUrl: url, 
                        videoId: retryResponse.data.video_id || videoId,
                        endpoint: endpoint.url
                      };
                    }
                  } catch (retryError) {
                    console.log(`Retry after token refresh failed for ${endpoint.url}:`, retryError.message);
                  }
                }
              }
            }
          }
        } catch (restError) {
          console.error('REST API approach failed:', restError);
        }
      }
      
      // Return success even if we're not sure, to allow navigation
      // The actual processing will continue in the background
      return { 
        success: true, 
        youtubeUrl: url, 
        videoId: videoId,
        pending: true
      };
    } catch (err) {
      console.error('Error processing YouTube URL:', err);
      
      if (!isMounted.current) return { success: false, error: 'Component unmounted' };
      
      // Special handling for timeout errors
      if (err.message && err.message.includes('Timeout')) {
        setError('Timeout transcribing YouTube video. Please try a shorter video or check your network connection.');
        // Update transcription with a user-friendly message
        setTranscription('Transcription timed out. This can happen with longer videos or slow network connections. Try a shorter clip or try again later.');
      } else {
        setError(err.message || 'Failed to process YouTube URL');
      }
      
      setProcessingYoutubeUrl(false);
      setTranscriptionLoading(false);
      setTranscriptionStatus('error');
      setProcessingStage('error');
      
      return { success: false, error: err.message };
    }
  }, [
    tabId, 
    checkServerHealth, 
    setupTranscriptionTimeout,
    getAuthHeaders,
    handleApiError
  ]);
  // ===== SOCKET INITIALIZATION =====
  // Initialize socket connection when tabId is available
  useEffect(() => {
    if (!tabId || !isMounted.current) return;
    
    // Only try to initialize socket once
    if (socketInitializedRef.current) return;
    
    console.log(`Initializing socket connection with tabId: ${tabId}`);
    setConnectionStatus('connecting');
    
    const initializeSocket = async () => {
      try {
        // Connect to socket with longer timeout
        console.log("Attempting to connect to socket...");
        await socketService.connect();
        console.log('Socket connected successfully');
        setConnectionStatus('connected');
        socketConnectedRef.current = true;
        
        // IMPORTANT: Wait longer before doing anything else
        console.log("Waiting for socket to fully initialize before registering tab or events...");
        await new Promise(resolve => setTimeout(resolve, SOCKET_RECONNECT_DELAY));
        
        // Register the tab ID with the server
        console.log("Registering tab ID with server:", tabId);
        socketService.registerTab(tabId);
        
        // Wait again before setting up listeners
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Set up listeners once socket is fully connected
        console.log("Setting up socket event listeners...");
        try {
          // Now setup all listeners properly
          const listenersSetup = functionsRef.current.setupSocketListeners();
          if (listenersSetup) {
            socketInitializedRef.current = true;
            connectionRetryCountRef.current = 0; // Reset retry counter
            console.log("Socket fully initialized with all listeners");
            
            // Send an echo to verify connection is working
            socketService.sendEcho({ message: "Connection test after initialization", timestamp: Date.now() });
          } else {
            console.error('Failed to set up socket listeners on first attempt');
            
            // Try again after a longer delay
            await new Promise(resolve => setTimeout(resolve, SOCKET_RECONNECT_DELAY));
            console.log("Second attempt to set up socket listeners...");
            
            const secondAttempt = functionsRef.current.setupSocketListeners();
            if (secondAttempt) {
              socketInitializedRef.current = true;
              connectionRetryCountRef.current = 0;
              console.log("Socket listeners initialized on second attempt");
            } else {
              console.error('Failed to set up socket listeners on second attempt');
            }
          }
        } catch (listenerError) {
          console.error("Error setting up socket listeners:", listenerError);
          // Continue anyway and let fallback mechanisms handle event registration
        }
      } catch (err) {
        console.error('Socket connection error:', err);
        setConnectionStatus(`error: ${err.message}`);
        setError(`Failed to connect to the server: ${err.message}`);
        socketConnectedRef.current = false;
        
        // Increment retry counter
        connectionRetryCountRef.current++;
        
        // Set a retry for socket connection with exponential backoff
        const delay = Math.min(30000, 1000 * Math.pow(2, connectionRetryCountRef.current));
        console.log(`Will retry socket connection in ${delay}ms (attempt #${connectionRetryCountRef.current})`);
        
        setTimeout(() => {
          if (!socketInitializedRef.current && isMounted.current) {
            console.log('Retrying socket connection...');
            initializeSocket();
          }
        }, delay);
      }
    };
    
    // Start the initialization process
    initializeSocket();
    
    // Clean up function
    return () => {
      console.log('Cleaning up socket connections and timers');
      
      // Use the safeCleanup method from socketService for better reliability
      if (socketService) {
        try {
          socketService.safeCleanup();
        } catch (error) {
          console.error('Error during socket cleanup:', error);
        }
      }
      
      // Clear any active timeouts and intervals
      if (thinkingTimeoutRef.current) {
        clearTimeout(thinkingTimeoutRef.current);
        thinkingTimeoutRef.current = null;
      }
      if (timestampHandlerRef.current) {
        clearTimeout(timestampHandlerRef.current);
        timestampHandlerRef.current = null;
      }
      if (fallbackTimerRef.current) {
        clearTimeout(fallbackTimerRef.current);
        fallbackTimerRef.current = null;
      }
      if (delayedTimestampGenRef.current) {
        clearTimeout(delayedTimestampGenRef.current);
        delayedTimestampGenRef.current = null;
      }
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
        heartbeatIntervalRef.current = null;
      }
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }
      
      // Cancel active transcription if exists
      if (activeTranscriptionRef.current && typeof activeTranscriptionRef.current.cancel === 'function') {
        try {
          activeTranscriptionRef.current.cancel();
        } catch (e) {
          console.error('Error cancelling active transcription:', e);
        }
        activeTranscriptionRef.current = null;
      }
      
      // Reset API active request flag
      activeApiRequestRef.current = null;
      
      socketConnectedRef.current = false;
    };
  }, [tabId]);
  // Effect to generate timestamps when transcription is loaded
  useEffect(() => {
    if (!hasGeneratedTimestamps && transcription && transcription.length > 0 && 
        (!timestamps || timestamps.length === 0)) {
      // If we have transcription but no timestamps, generate them after a short delay
      console.log('Transcription loaded but no timestamps, scheduling generation');
      
      // Clear any existing timeout
      if (delayedTimestampGenRef.current) {
        clearTimeout(delayedTimestampGenRef.current);
      }
      
      // Schedule timestamp generation after a short delay
      delayedTimestampGenRef.current = setTimeout(() => {
        if (isMounted.current) {
          console.log('Automatically generating timestamps from transcript');
          if (functionsRef.current.generateTimestampsFromTranscript) {
            functionsRef.current.generateTimestampsFromTranscript();
          }
        }
      }, TRANSCRIPTION_DELAY);
      
      // Clean up timeout on unmount
      return () => {
        if (delayedTimestampGenRef.current) {
          clearTimeout(delayedTimestampGenRef.current);
          delayedTimestampGenRef.current = null;
        }
      };
    }
  }, [transcription, timestamps, hasGeneratedTimestamps]);
  // ===== CONTEXT PROVIDER =====
  // Return the context provider with all values
  return (
    <VideoContext.Provider value={{
      videoData,
      transcription,
      loading,
      error,
      conversations,
      addMessage,
      sendMessageToAI,
      processYoutubeUrl,
      uploadVideoFile,
      uploadAudioFile,
      fetchTranscription: fetchTranscriptionViaREST,
      transcriptionStatus,
      processingYoutubeUrl,
      transcriptionLoading,
      visualAnalysisAvailable,
      visualAnalysisInProgress,
      videoTopics,
      videoHighlights,
      visualSummary,
      currentView,
      selectedTimestamp,
      detectedScenes,
      keyFrames,
      currentVideoId,
      requestVisualAnalysis,
      navigateToTimestamp,
      getVideoSummary,
      getVideoTopicAnalysis,
      switchView,
      isAIThinking,
      setIsAIThinking,
      resetVideoState,
      // Timestamps
      timestamps,
      activeTimestampIndex,
      setActiveTimestampIndex,
      updateCurrentTime,
      seekToTimestamp,
      currentTime,
      // Enhanced features
      detectTimestampQuestion,
      extractTimestampFromQuestion,
      askAboutVisibleContent,
      detectVisualQuestion,
      visibleTimestamps,
      clearConversation,
      sidebarTimestamps,
      processingStage,
      // Timestamp fetching state
      timestampFetchAttempts,
      timestampFetchFailed,
      generateTimestampsFromTranscript,
      hasGeneratedTimestamps
    }}>
      {children}
    </VideoContext.Provider>
  );
}
export default VideoContext;