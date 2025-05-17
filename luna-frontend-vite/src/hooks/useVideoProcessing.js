import { useState, useEffect, useCallback, useRef } from 'react';
import { useSocket } from './useSocket';
import api from '../services/api';

const useVideoProcessing = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [videoData, setVideoData] = useState(null);
  const [transcription, setTranscription] = useState(null);
  const [visualData, setVisualData] = useState(null);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [transcriptionStatus, setTranscriptionStatus] = useState('idle'); // Added to track transcription status
  
  const { socket, connected } = useSocket(); // Added connected state from useSocket
  const processingVideoIdRef = useRef(null); // Added to keep track of current processing ID

  // Reset all states
  const resetState = useCallback(() => {
    setLoading(false);
    setError(null);
    setProgress(0);
    setVideoData(null);
    setTranscription(null);
    setVisualData(null);
    setAnalysisComplete(false);
    setTranscriptionStatus('idle');
    processingVideoIdRef.current = null;
  }, []);

  // Process YouTube URL
  const processYoutubeURL = useCallback(async (url) => {
    resetState();
    setLoading(true);
    setTranscriptionStatus('waiting');
    
    try {
      // Enhanced YouTube URL parsing
      let videoId = '';
      
      // Handle various YouTube URL formats
      if (url.includes('youtube.com') || url.includes('youtu.be')) {
        try {
          const urlObj = new URL(url);
          
          if (url.includes('youtube.com/watch')) {
            // This is the key fix - extract just the 'v' parameter from the query string
            videoId = urlObj.searchParams.get('v');
          } else if (url.includes('youtube.com/shorts/')) {
            // Handle shorts format
            videoId = urlObj.pathname.split('/shorts/')[1]?.split('/')[0]?.split('?')[0];
          } else if (url.includes('youtu.be/')) {
            // Handle shortened URL format
            videoId = urlObj.pathname.slice(1).split('/')[0].split('?')[0];
          } else if (url.includes('youtube.com/embed/')) {
            // Handle embed format
            videoId = urlObj.pathname.split('/embed/')[1]?.split('/')[0]?.split('?')[0];
          } else if (url.includes('youtube.com/v/')) {
            // Handle old-style v parameter format
            videoId = urlObj.pathname.split('/v/')[1]?.split('/')[0]?.split('?')[0];
          }
          
          console.log(`Extracted videoId: "${videoId}" from URL: "${url}"`);
          
        } catch (parseError) {
          console.error('Error parsing YouTube URL:', parseError);
          
          // Try a regex fallback for YouTube IDs - this is more robust for complex URLs
          const regexMatch = url.match(/(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/);
          if (regexMatch && regexMatch[1]) {
            videoId = regexMatch[1];
            console.log(`Extracted videoId via regex: "${videoId}" from URL: "${url}"`);
          }
        }
      } else if (/^[a-zA-Z0-9_-]{11}$/.test(url)) {
        // Direct video ID input (11 characters)
        videoId = url;
        console.log(`Using direct videoId: "${videoId}"`);
      }
      
      if (!videoId || videoId.length !== 11) {
        throw new Error('Invalid YouTube URL or ID. Please provide a valid YouTube URL (e.g., https://www.youtube.com/watch?v=...)');
      }

      // Format the videoId properly for API calls
      const formattedVideoId = `youtube_${videoId}`;
      processingVideoIdRef.current = formattedVideoId;
      
      // Check socket connection
      if (!socket || !connected) {
        throw new Error('Socket connection not available. Please refresh the page and try again.');
      }
      
      // Emit socket event to start processing
      console.log(`Requesting processing for YouTube video: ${formattedVideoId}`);
      socket.emit('process_youtube', { 
        videoId: formattedVideoId,
        requestTimestamp: new Date().toISOString() // Add timestamp for debugging
      });
      
      // Add a fallback API call to ensure the request is properly received
      try {
        await api.post('/api/v1/videos/process', { 
          videoId: formattedVideoId,
          source: 'youtube'
        });
      } catch (apiError) {
        console.warn('Fallback API call failed, continuing with socket method:', apiError);
        // Continue with socket method even if this fails
      }
      
      setTranscriptionStatus('processing');
      return formattedVideoId;
    } catch (err) {
      console.error('Error processing YouTube URL:', err);
      setError(err.message || 'Failed to process YouTube URL');
      setLoading(false);
      setTranscriptionStatus('error');
      return null;
    }
  }, [socket, connected, resetState]);

  // Upload a video file
  const uploadVideo = useCallback(async (file) => {
    resetState();
    setLoading(true);
    setTranscriptionStatus('waiting');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      // First upload the file
      const response = await api.post('/api/v1/videos/upload', formData, {
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setProgress(percentCompleted);
        },
        timeout: 300000, // 5 minute timeout for large files
      });
      
      if (!response.data || !response.data.video_id) {
        throw new Error('Invalid response from server');
      }
      
      processingVideoIdRef.current = response.data.video_id;
      setVideoData(response.data);
      setTranscriptionStatus('processing');
      
      // Explicitly request transcription after upload
      if (socket && connected) {
        socket.emit('start_transcription', { videoId: response.data.video_id });
      }
      
      return response.data.video_id;
    } catch (err) {
      console.error('Error uploading video:', err);
      setError(err.message || 'Failed to upload video');
      setLoading(false);
      setTranscriptionStatus('error');
      return null;
    }
  }, [socket, connected, resetState]);

  // Fetch visual analysis data
  const fetchVisualData = useCallback(async (videoId) => {
    if (!videoId) return null;
    
    try {
      console.log(`Fetching visual data for: ${videoId}`);
      // Try to fetch existing visual data first
      const response = await api.get(`/api/v1/videos/${videoId}/visual-data`);
      
      if (response.data && response.data.data && Object.keys(response.data.data).length > 0) {
        console.log('Visual data found:', response.data);
        setVisualData(response.data.data);
        return response.data.data;
      }
      
      // If no data found, request visual analysis
      console.log('No visual data available, requesting visual analysis');
      const analysisResponse = await api.get(`/api/v1/visual-analysis/${videoId}`);
      
      if (analysisResponse.data && analysisResponse.data.data) {
        console.log('Visual analysis response:', analysisResponse.data);
        setVisualData(analysisResponse.data.data);
        return analysisResponse.data.data;
      }
      
      return null;
    } catch (err) {
      console.warn('Error fetching visual data:', err.message);
      // Don't set error state here, as this is often expected to fail initially
      return null;
    }
  }, []);

  // Added: Function to explicitly request transcription for a video
  const requestTranscription = useCallback((videoId) => {
    if (!videoId || !socket || !connected) return false;
    
    try {
      console.log(`Explicitly requesting transcription for: ${videoId}`);
      setTranscriptionStatus('processing');
      
      socket.emit('start_transcription', { 
        videoId, 
        force: true, // Force retranscription even if already exists
        requestTimestamp: new Date().toISOString()
      });
      
      // Also make an API call as fallback
      api.post('/api/v1/videos/transcribe', { videoId })
        .then(res => console.log('Transcription request API response:', res.data))
        .catch(err => console.warn('Transcription API request failed:', err));
      
      return true;
    } catch (err) {
      console.error('Error requesting transcription:', err);
      return false;
    }
  }, [socket, connected]);

  // Check status of processing
  const checkProcessingStatus = useCallback(async (videoId) => {
    if (!videoId) return;
    
    try {
      const response = await api.get(`/api/v1/videos/${videoId}/status`);
      
      if (response.data && response.data.status) {
        console.log(`Processing status for ${videoId}:`, response.data);
        
        if (response.data.transcription) {
          setTranscription(response.data.transcription);
        }
        
        if (response.data.status === 'completed') {
          setLoading(false);
          setAnalysisComplete(true);
          setTranscriptionStatus('completed');
        } else if (response.data.status === 'failed') {
          setError(response.data.error || 'Processing failed');
          setLoading(false);
          setTranscriptionStatus('error');
        } else {
          setTranscriptionStatus('processing');
        }
      }
    } catch (err) {
      console.warn('Error checking processing status:', err);
    }
  }, []);

  // Set up socket event listeners
  useEffect(() => {
    if (!socket) return;
    
    const handleTranscriptionUpdate = (data) => {
      console.log('Transcription update received:', data);
      
      // Only update if this is for the current video
      if (processingVideoIdRef.current && data.videoId && 
          data.videoId !== processingVideoIdRef.current) {
        console.warn('Received transcription for different video ID, ignoring');
        return;
      }
      
      setTranscription(data);
      setTranscriptionStatus('updating');
    };
    
    const handleVisualUpdate = (data) => {
      console.log('Visual update received:', data);
      
      // Only update if this is for the current video
      if (processingVideoIdRef.current && data.videoId && 
          data.videoId !== processingVideoIdRef.current) {
        console.warn('Received visual update for different video ID, ignoring');
        return;
      }
      
      setVisualData(data);
    };
    
    const handleProcessingComplete = (data) => {
      console.log('Processing complete:', data);
      
      // Only update if this is for the current video
      if (processingVideoIdRef.current && data.videoId && 
          data.videoId !== processingVideoIdRef.current) {
        console.warn('Received completion for different video ID, ignoring');
        return;
      }
      
      setLoading(false);
      setAnalysisComplete(true);
      setTranscriptionStatus('completed');
    };
    
    const handleProcessingError = (data) => {
      console.error('Processing error:', data);
      
      // Only update if this is for the current video
      if (processingVideoIdRef.current && data.videoId && 
          data.videoId !== processingVideoIdRef.current) {
        console.warn('Received error for different video ID, ignoring');
        return;
      }
      
      setError(data.message || 'An error occurred during processing');
      setLoading(false);
      setTranscriptionStatus('error');
    };
    
    const handleUploadProgress = (data) => {
      console.log('Upload progress:', data);
      setProgress(data.progress);
    };
    
    const handleTranscriptionStatus = (data) => {
      console.log('Transcription status update:', data);
      
      // Only update if this is for the current video
      if (processingVideoIdRef.current && data.videoId && 
          data.videoId !== processingVideoIdRef.current) {
        return;
      }
      
      setTranscriptionStatus(data.status);
    };
    
    const handleSocketReconnect = () => {
      console.log('Socket reconnected');
      
      // If we have a video being processed, check its status
      if (processingVideoIdRef.current) {
        checkProcessingStatus(processingVideoIdRef.current);
      }
    };
    
    // Register event listeners
    socket.on('transcription_update', handleTranscriptionUpdate);
    socket.on('visual_update', handleVisualUpdate);
    socket.on('processing_complete', handleProcessingComplete);
    socket.on('processing_error', handleProcessingError);
    socket.on('upload_progress', handleUploadProgress);
    socket.on('transcription_status', handleTranscriptionStatus);
    socket.on('connect', handleSocketReconnect);
    
    // Clean up listeners
    return () => {
      socket.off('transcription_update', handleTranscriptionUpdate);
      socket.off('visual_update', handleVisualUpdate);
      socket.off('processing_complete', handleProcessingComplete);
      socket.off('processing_error', handleProcessingError);
      socket.off('upload_progress', handleUploadProgress);
      socket.off('transcription_status', handleTranscriptionStatus);
      socket.off('connect', handleSocketReconnect);
    };
  }, [socket, checkProcessingStatus]);
  
  // Periodically check status of current processing job
  useEffect(() => {
    if (!loading || !processingVideoIdRef.current) return;
    
    const statusCheckInterval = setInterval(() => {
      checkProcessingStatus(processingVideoIdRef.current);
    }, 10000); // Check every 10 seconds
    
    return () => clearInterval(statusCheckInterval);
  }, [loading, checkProcessingStatus]);

  return {
    loading,
    error,
    progress,
    videoData,
    transcription,
    visualData,
    analysisComplete,
    transcriptionStatus,
    processYoutubeURL,
    uploadVideo,
    fetchVisualData,
    requestTranscription, // Added new function
    resetState
  };
};

export default useVideoProcessing;