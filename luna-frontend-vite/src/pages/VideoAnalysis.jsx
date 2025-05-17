// src/pages/VideoAnalysis.jsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useVideo } from '../context/VideoContext';
import VideoPlayer from '../components/VideoPlayer';
import VisualQAInterface from '../components/VisualQAInterface';
import LoadingIndicator from '../components/LoadingIndicator';
import TimestampDisplay from '../components/TimestampDisplay';
import { marked } from 'marked';
import '../styles/videoAnalysis.css';
import socketService from '../services/socket';
import axios from 'axios';
// AI-themed Loading Component
const AILoadingIndicator = ({ message = "Processing transcription" }) => {
  return (
    <div className="loading-container">
      {/* AI-Themed Circle Loading Indicator */}
      <div className="ai-loading-circle">
        {/* Neural nodes */}
        <div className="neural-node"></div>
        <div className="neural-node"></div>
        <div className="neural-node"></div>
        <div className="neural-node"></div>
        <div className="neural-node"></div>
        <div className="neural-node"></div>
        <div className="neural-node"></div>
        <div className="neural-node"></div>
        {/* Center pulsing core */}
        <div className="ai-loading-core"></div>
      </div>
      {/* AI-Themed Loading Text */}
      <div className="ai-loading-text">
        {message}
        <div className="ai-dots-container">
          <div className="ai-dot"></div>
          <div className="ai-dot"></div>
          <div className="ai-dot"></div>
        </div>
      </div>
    </div>
  );
};
// Enhanced utility function to validate YouTube URL with better regex
const isValidYouTubeUrl = (url) => {
  if (!url) return false;
  url = url.trim();
  const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|shorts\/)|youtu\.be\/)[a-zA-Z0-9_-]+(\?.*)?$/i;
  return youtubeRegex.test(url);
};
// Improved YouTube video ID extraction function with multiple patterns
const getYouTubeVideoId = (url) => {
  if (!url) return null;
  // Comprehensive regex patterns to match all YouTube URL formats
  const patterns = [
    /(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)/i,
    /(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^/?]+)/i,
    /(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^/?]+)/i,
    /(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([^/?]+)/i,
    /(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^/?]+)/i
  ];
  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match && match[1]) {
      return match[1];
    }
  }
  // Fallback to the original regex if none of the patterns match
  const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
  const match = url.match(regExp);
  return (match && match[2].length === 11) ? match[2] : null;
};
// Helper function to determine if a file is an audio file
const isAudioFile = (file) => {
  if (!file) return false;
  // Check MIME type first
  if (file.type.startsWith('audio/')) return true;
  // Check file extension as fallback
  const audioExtensions = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'];
  const fileName = file.name.toLowerCase();
  return audioExtensions.some(ext => fileName.endsWith(ext));
};
const VideoAnalysis = () => {
  const navigate = useNavigate();
  const [authInitialized, setAuthInitialized] = useState(false);
  const { user, logout, isAuthenticated } = useAuth();
  const {
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
    currentVideoId,
    requestVisualAnalysis,
    navigateToTimestamp,
    getVideoSummary,
    getVideoTopicAnalysis,
    switchView,
    isAIThinking,
    setIsAIThinking,
    resetVideoState,
    detectTimestampQuestion,
    extractTimestampFromQuestion,
    askAboutVisibleContent,
    visibleTimestamps,
    processingStage,
    setTranscriptionLoading,
    setProcessingYoutubeUrl
  } = useVideo();
  const [userInput, setUserInput] = useState('');
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [showInputForm, setShowInputForm] = useState(true);
  const [processingStatus, setProcessingStatus] = useState("");
  const [activeTab, setActiveTab] = useState('transcription');
  const [isDarkMode, setIsDarkMode] = useState(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
  const [showVideoPlayer, setShowVideoPlayer] = useState(false);
  const [urlError, setUrlError] = useState('');
  // Add state for drag and drop functionality
  const [dragActive, setDragActive] = useState(false);
  const [isTimestampDetected, setIsTimestampDetected] = useState(false);
  const [detectedTimestamp, setDetectedTimestamp] = useState(null);
  const [showTimestampBubble, setShowTimestampBubble] = useState(false);
  const [recentTimestamps, setRecentTimestamps] = useState([]);
  const [capturedFrame, setCapturedFrame] = useState(null);
  const [visualError, setVisualError] = useState(null);
  const [isRetrying, setIsRetrying] = useState(false);
  const [timeoutOccurred, setTimeoutOccurred] = useState(false);
  const [timestamps, setTimestamps] = useState([]);
  const [isLoadingTimestamps, setIsLoadingTimestamps] = useState(false);
  const [hasTimestamps, setHasTimestamps] = useState(false);
  const [socketReady, setSocketReady] = useState(false);
  // Add state for improved timestamp features
  const [timestampSearchQuery, setTimestampSearchQuery] = useState('');
  const [groupedTimestamps, setGroupedTimestamps] = useState({});
  const [currentProgress, setCurrentProgress] = useState(0);
  const [showTimestampTooltip, setShowTimestampTooltip] = useState(false);
  const [tooltipIndex, setTooltipIndex] = useState(null);
  const [videoDuration, setVideoDuration] = useState(0);
  const [videoPlayerError, setVideoPlayerError] = useState(null);
  const [transcriptionDebug, setTranscriptionDebug] = useState({
    hasTranscription: false,
    length: 0,
    isLoading: true,
    status: 'initializing',
    displayTime: Date.now()
  });
  // NEW: Add state to track which message was copied
  const [copiedMessageId, setCopiedMessageId] = useState(null);
  
  const chatAreaRef = useRef(null);
  const userInputRef = useRef(null);
  const videoPlayerRef = useRef(null);
  const timestampVideoRef = useRef(null);
  const sidebarVideoRef = useRef(null);
  const youtubeInputRef = useRef(null);
  const isComponentMounted = useRef(true);
  const timestampRequestAttempts = useRef(0);
  const socketListenersRegistered = useRef(false);
  const transcriptContainerRef = useRef(null);
  const fileInputRef = useRef(null);
  const timelineRef = useRef(null);
  const [hasNavigated, setHasNavigated] = useState(false);
  // Debug effect for video ID
  useEffect(() => {
    if (videoData?.youtubeUrl) {
      const videoId = getYouTubeVideoId(videoData.youtubeUrl);
      console.log("YouTube URL:", videoData.youtubeUrl);
      console.log("Extracted Video ID:", videoId);
    }
  }, [videoData]);
  // Add document-level drag and drop event handlers
  useEffect(() => {
    if (showInputForm) {
      // Define the handlers for document-level drag events
      const handleDocumentDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        // Only react to drag events that contain files
        if (e.dataTransfer.types && e.dataTransfer.types.includes('Files')) {
          if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
          }
        }
      };
      const handleDocumentDragLeave = (e) => {
        e.preventDefault();
        // Check if drag left the document (to the browser chrome or another window)
        if (e.clientX === 0 && e.clientY === 0) {
          setDragActive(false);
        }
      };
      const handleDocumentDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
          setSelectedFile(e.dataTransfer.files[0]);
        }
      };
      // Add event listeners to the document
      document.addEventListener("dragenter", handleDocumentDrag);
      document.addEventListener("dragover", handleDocumentDrag);
      document.addEventListener("dragleave", handleDocumentDragLeave);
      document.addEventListener("drop", handleDocumentDrop);
      // Clean up
      return () => {
        document.removeEventListener("dragenter", handleDocumentDrag);
        document.removeEventListener("dragover", handleDocumentDrag);
        document.removeEventListener("dragleave", handleDocumentDragLeave);
        document.removeEventListener("drop", handleDocumentDrop);
      };
    }
  }, [showInputForm]);
  // Handle element-level drag events
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);
  // Handle element-level drop event
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  }, []);
  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleThemeChange = (e) => {
      setIsDarkMode(e.matches);
    };
    mediaQuery.addEventListener('change', handleThemeChange);
    return () => mediaQuery.removeEventListener('change', handleThemeChange);
  }, []);
  // Apply theme class to body
  useEffect(() => {
    document.body.className = isDarkMode ? 'dark-theme' : 'light-theme';
  }, [isDarkMode]);
  // Check for transcription on load
  useEffect(() => {
    setAuthInitialized(true);
    if (transcription && transcription.length > 0) {
      console.log('Transcription detected on initial load, transitioning to chat interface');
      setShowInputForm(false);
    }
    return () => {
      isComponentMounted.current = false;
    };
  }, [transcription]);
  // Effect to auto-switch to timestamps tab when clicking "Show Video"
  useEffect(() => {
    if (showVideoPlayer && !activeTab === 'timestamps') {
      setActiveTab('timestamps');
      if (!isSidebarVisible) {
        setIsSidebarVisible(true);
      }
    }
  }, [showVideoPlayer, activeTab, isSidebarVisible]);
  // Update UI based on transcription status with improved transition logic
  useEffect(() => {
    if (!isComponentMounted.current) return;
    console.log('Transcription status changed:', transcriptionStatus, 'Processing stage:', processingStage);
    setProcessingStatus(getStatusMessage());
    if ((transcriptionStatus === 'loaded' || transcriptionStatus === 'completed') && transcription && transcription.length > 0) {
      console.log('Transcription loaded/completed, showing chat interface');
      setShowInputForm(false);
      setTimeoutOccurred(false);
      setActiveTab('transcription');
      if (currentVideoId) {
        console.log('Requesting timestamps for newly transcribed video');
        requestTimestamps(currentVideoId);
      }
    }
    else if (transcriptionStatus === 'error') {
      setProcessingStatus("Error processing video");
      if (error && (error.includes('Timeout') || error.includes('timeout'))) {
        setTimeoutOccurred(true);
      }
    }
  }, [transcriptionStatus, transcription, error, currentVideoId, processingStage]);
  // Force transition to chat interface when transcription is loaded
  useEffect(() => {
    if (transcription && transcription.length > 0 && showInputForm) {
      console.log('Forcing transition to chat interface based on transcription availability');
      setTimeout(() => {
        if (isComponentMounted.current) {
          setShowInputForm(false);
          setActiveTab('transcription');
        }
      }, 300);
    }
  }, [transcription, showInputForm]);
  // Process and group timestamps when they change
  useEffect(() => {
    if (!timestamps || timestamps.length === 0) return;
    // Set hasTimestamps flag
    setHasTimestamps(true);
    // Get the maximum video duration from timestamps
    const maxDuration = Math.max(...timestamps.map(ts => {
      if (typeof ts.end_time === 'number') return ts.end_time;
      if (typeof ts.time === 'number') return ts.time;
      return 0;
    }));
    setVideoDuration(maxDuration);
    // Create timestamp groups (every 5 minutes)
    const groupSize = 5 * 60; // 5 minutes in seconds
    const groups = {};
    // Group by time ranges (0-5 min, 5-10 min, etc.)
    timestamps.forEach(ts => {
      const time = typeof ts.start_time === 'number' ? ts.start_time :
        typeof ts.time === 'number' ? ts.time : 0;
      const groupIndex = Math.floor(time / groupSize);
      const groupStart = groupIndex * groupSize;
      const groupEnd = Math.min((groupIndex + 1) * groupSize, maxDuration);
      const groupKey = `${formatTimeFromSeconds(groupStart)}-${formatTimeFromSeconds(groupEnd)}`;
      if (!groups[groupKey]) {
        groups[groupKey] = [];
      }
      groups[groupKey].push(ts);
    });
    setGroupedTimestamps(groups);
  }, [timestamps]);
  // Update current progress in the video player
  useEffect(() => {
    if (!videoPlayerRef.current || !timelineRef.current) return;
    const updateProgress = () => {
      try {
        const player = videoPlayerRef.current.getInternalPlayer();
        if (!player) return;
        const currentTime = player.getCurrentTime();
        const duration = player.getDuration();
        if (typeof currentTime === 'number' && typeof duration === 'number') {
          const progressPercentage = (currentTime / duration) * 100;
          setCurrentProgress(progressPercentage);
          if (duration > 0 && (!videoDuration || videoDuration === 0)) {
            setVideoDuration(duration);
          }
        }
      } catch (error) {
        console.error("Error updating progress:", error);
      }
    };
    const intervalId = setInterval(updateProgress, 1000);
    return () => clearInterval(intervalId);
  }, [showVideoPlayer, videoDuration]);
  // Improved request timestamps function with fallback to direct API
  const requestTimestamps = useCallback((videoId) => {
    if (!videoId) return;
    console.log("Requesting timestamps for video:", videoId);
    setIsLoadingTimestamps(true);
    // First try direct API call to avoid socket issues
    const fetchDirectTimestamps = async () => {
      try {
        const baseApiUrl = window.location.origin;
        const endpoints = [
          `/api/v1/videos/${videoId}/timestamps`,
          `/api/videos/${videoId}/timestamps`,
          `/api/timestamps?videoId=${videoId}`
        ];
        let success = false;
        for (const endpoint of endpoints) {
          try {
            const response = await axios.get(`${baseApiUrl}${endpoint}`, {
              timeout: 5000,
              validateStatus: (status) => status < 500
            });
            if (response.status === 200 && response.data) {
              if (Array.isArray(response.data)) {
                console.log(`Found ${response.data.length} timestamps via direct API`);
                setTimestamps(response.data);
                setIsLoadingTimestamps(false);
                success = true;
                break;
              } else if (response.data.timestamps && Array.isArray(response.data.timestamps)) {
                console.log(`Found ${response.data.timestamps.length} timestamps via direct API`);
                setTimestamps(response.data.timestamps);
                setIsLoadingTimestamps(false);
                success = true;
                break;
              }
            } else {
              console.log(`Endpoint ${endpoint} returned status ${response.status}`);
            }
          } catch (error) {
            console.log(`Error with endpoint ${endpoint}:`, error.message);
          }
        }
        if (!success) {
          attemptSocketTimestamps();
        }
      } catch (error) {
        console.error("Error in direct API timestamp fetch:", error);
        attemptSocketTimestamps();
      }
    };
    // Socket-based timestamp retrieval
    const attemptSocketTimestamps = (retryCount = 0) => {
      try {
        if (!socketService.isSocketConnected()) {
          console.log("Socket not connected, retrying timestamp request...");
          if (retryCount < 3) {
            setTimeout(() => attemptSocketTimestamps(retryCount + 1), 2000);
          } else {
            console.error("Failed to connect socket for timestamps after retries");
            setIsLoadingTimestamps(false);
            generateTimestampsFromTranscript();
          }
          return;
        }
        socketService.emit('get_timestamps', {
          video_id: videoId,
          tabId: Date.now().toString()
        });
        setTimeout(() => {
          if (socketService.isSocketConnected()) {
            socketService.emit('get_transcript_timestamps', {
              video_id: videoId,
              tabId: Date.now().toString()
            });
          }
        }, 500);
        setTimeout(() => {
          if (isLoadingTimestamps && isComponentMounted.current) {
            console.log("Timestamp request timed out");
            setIsLoadingTimestamps(false);
            generateTimestampsFromTranscript();
            timestampRequestAttempts.current += 1;
            if (timestampRequestAttempts.current < 3 && isComponentMounted.current) {
              console.log(`Auto-retrying timestamp request (attempt ${timestampRequestAttempts.current + 1})`);
              attemptSocketTimestamps(retryCount + 1);
            } else {
              timestampRequestAttempts.current = 0;
            }
          }
        }, 10000);
      } catch (error) {
        console.error("Error requesting timestamps:", error);
        if (retryCount < 3) {
          setTimeout(() => attemptSocketTimestamps(retryCount + 1), 2000);
        } else {
          setIsLoadingTimestamps(false);
          generateTimestampsFromTranscript();
        }
      }
    };
    fetchDirectTimestamps();
  }, []);
  // Generate timestamps from transcript directly
  const generateTimestampsFromTranscript = useCallback(() => {
    if (!transcription) return;
    console.log('Generating timestamps from transcript as fallback');
    try {
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
      const estimatedDuration = 600;
      const charsPerSecond = 6;
      const generatedTimestamps = sentences.map((sentence, index) => {
        const startPosition = transcription.indexOf(sentence);
        if (startPosition === -1) {
          const approxPosition = (index / sentences.length) * totalLength;
          const startTime = (approxPosition / totalLength) * estimatedDuration;
          const endTime = ((index + 1) / sentences.length) * estimatedDuration;
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
        const startTime = (startPosition / totalLength) * estimatedDuration;
        let endTime;
        if (index < sentences.length - 1) {
          const nextSentence = sentences[index + 1];
          const nextSentencePos = transcription.indexOf(nextSentence);
          if (nextSentencePos !== -1) {
            endTime = (nextSentencePos / totalLength) * estimatedDuration;
          } else {
            endTime = startTime + (sentence.length / charsPerSecond);
          }
        } else {
          endTime = estimatedDuration;
        }
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
        console.log(`Generated ${generatedTimestamps.length} timestamps from transcript`);
        setTimestamps(generatedTimestamps);
      }
    } catch (error) {
      console.error('Error generating timestamps:', error);
    }
  }, [transcription]);
  // Auto-resize input field
  const autoResizeInput = useCallback(() => {
    if (userInputRef.current) {
      userInputRef.current.style.height = 'auto';
      userInputRef.current.style.height = Math.min(userInputRef.current.scrollHeight, 150) + 'px';
    }
  }, []);
  // Toggle sidebar visibility
  const toggleSidebar = useCallback(() => {
    setIsSidebarVisible(!isSidebarVisible);
  }, [isSidebarVisible]);
  // Toggle dark mode
  const toggleTheme = useCallback(() => {
    setIsDarkMode(!isDarkMode);
  }, [isDarkMode]);
  // Toggle video player visibility - UPDATED to properly show in sidebar
  const toggleVideoPlayer = useCallback(() => {
    // Switch to timestamps tab and make sidebar visible
    setActiveTab('timestamps');
    setShowVideoPlayer(true);
    if (!isSidebarVisible) {
      setIsSidebarVisible(true);
    }
    // For debugging
    console.log("Toggling video player:", {
      videoUrl: videoData?.youtubeUrl,
      videoId: getYouTubeVideoId(videoData?.youtubeUrl)
    });
    // Reset any previous video errors
    setVideoPlayerError(null);
    // Force a delay before attempting to reload the iframe
    setTimeout(() => {
      const iframe = document.getElementById('youtube-player-iframe');
      if (iframe && videoData?.youtubeUrl) {
        const videoId = getYouTubeVideoId(videoData.youtubeUrl);
        if (videoId) {
          console.log("Refreshing YouTube iframe with ID:", videoId);
          iframe.src = `https://www.youtube.com/embed/${videoId}?enablejsapi=1&origin=${window.location.origin}`;
        }
      }
    }, 300);
  }, [isSidebarVisible, videoData]);
  // Reset the app state and clear the current video
  const handleResetState = useCallback(() => {
    resetVideoState();
    setShowInputForm(true);
    setProcessingStatus("");
    setYoutubeUrl("");
    setSelectedFile(null);
    setTimeoutOccurred(false);
    setIsRetrying(false);
    setTimestamps([]);
    setHasTimestamps(false);
    setUrlError('');
    setVideoPlayerError(null);
    timestampRequestAttempts.current = 0;
  }, [resetVideoState]);
  // Handle retry for timeout errors with improved error handling
  const handleRetryProcessing = useCallback(async () => {
    try {
      setIsRetrying(true);
      resetVideoState();
      if (youtubeUrl) {
        let modifiedUrl = youtubeUrl;
        if (!youtubeUrl.includes('/shorts/') && !youtubeUrl.includes('&t=') && !youtubeUrl.includes('?t=')) {
          if (youtubeUrl.includes('?')) {
            modifiedUrl = `${youtubeUrl}&t=0s`;
          } else {
            modifiedUrl = `${youtubeUrl}?t=0s`;
          }
        }
        const result = await processYoutubeUrl(modifiedUrl);
        if (!result?.success) {
          throw new Error(result?.error || 'Failed to process YouTube URL during retry');
        }
        setIsRetrying(false);
        setTimeoutOccurred(false);
      } else if (selectedFile) {
        // Determine if it's an audio file and use the appropriate upload function
        const isAudio = isAudioFile(selectedFile);
        if (isAudio) {
          await uploadAudioFile(selectedFile);
        } else {
          await uploadVideoFile(selectedFile);
        }
        setIsRetrying(false);
        setTimeoutOccurred(false);
      }
    } catch (error) {
      console.error("Error during retry:", error);
      setIsRetrying(false);
      console.error('Retry attempt failed: ' + (error.message || 'Unknown error'));
    }
  }, [youtubeUrl, selectedFile, resetVideoState, processYoutubeUrl, uploadVideoFile, uploadAudioFile]);
  // Process YouTube URL or upload file with improved error handling
  const handleProcessVideo = useCallback(async () => {
    setTimeoutOccurred(false);
    setUrlError('');
    if (youtubeUrl) {
      if (!isValidYouTubeUrl(youtubeUrl)) {
        setUrlError('Please enter a valid YouTube URL (e.g., https://www.youtube.com/watch?v=...)');
        return;
      }
      let processUrl = youtubeUrl.trim();
      if (!processUrl.startsWith('http')) {
        processUrl = 'https://' + processUrl;
      }
      console.log("Processing YouTube URL:", processUrl);
      try {
        const result = await processYoutubeUrl(processUrl);
        if (result?.success) {
          setShowInputForm(false);
        } else {
          setUrlError(result?.error || 'Failed to process YouTube URL');
        }
      } catch (error) {
        console.error("Error processing YouTube URL:", error);
        if (error.message && (
          error.message.includes('URL') ||
          error.message.includes('youtube') ||
          error.message.includes('invalid')
        )) {
          setUrlError(error.message);
        } else {
          setUrlError('Failed to process YouTube URL. Please try again.');
        }
      }
    } else if (selectedFile) {
      try {
        // Determine if it's an audio file and use the appropriate upload function
        const isAudio = isAudioFile(selectedFile);
        console.log(`Uploading file as ${isAudio ? 'audio' : 'video'} type:`, selectedFile.name);
        
        const result = isAudio 
          ? await uploadAudioFile(selectedFile)
          : await uploadVideoFile(selectedFile);
          
        if (result?.success) {
          setShowInputForm(false);
        } else {
          setUrlError(result?.error || 'Failed to upload file');
        }
      } catch (error) {
        console.error("Error uploading file:", error);
        setUrlError('Failed to upload file: ' + (error.message || 'Unknown error'));
      }
    }
  }, [youtubeUrl, selectedFile, processYoutubeUrl, uploadVideoFile, uploadAudioFile]);
  // Send message to AI with enhanced timestamp handling
  const sendMessage = useCallback(async () => {
    if (!userInput.trim() || isAIThinking) return;
    try {
      setIsAIThinking(true);
      if (isTimestampDetected && detectedTimestamp) {
        if (videoPlayerRef.current) {
          const totalSeconds = detectedTimestamp.totalSeconds ||
            (detectedTimestamp.minutes * 60 + detectedTimestamp.secondsComponent);
          videoPlayerRef.current.seekTo(totalSeconds);
          setTimeout(() => {
            if (isComponentMounted.current) {
              sendMessageToAI(userInput);
              setUserInput('');
            }
          }, 300);
        } else {
          await sendMessageToAI(userInput);
          setUserInput('');
        }
      }
      else if (capturedFrame) {
        await askAboutVisibleContent?.(userInput);
        setUserInput('');
        setCapturedFrame(null);
      }
      else {
        await sendMessageToAI(userInput);
        setUserInput('');
      }
      if (userInputRef.current) {
        userInputRef.current.style.height = 'auto';
      }
    } catch (error) {
      console.error("Error sending message:", error);
      if (isComponentMounted.current) {
        setIsAIThinking(false);
      }
    }
  }, [userInput, isAIThinking, isTimestampDetected, detectedTimestamp, capturedFrame, sendMessageToAI, askAboutVisibleContent]);
  // Handle input keypress (Enter to send)
  const handleKeyPress = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }, [sendMessage]);
  // Copy text to clipboard - UPDATED with improved notification
  const copyToClipboard = useCallback((text, messageId) => {
    navigator.clipboard.writeText(text)
      .then(() => {
        console.log("Copied to clipboard");
        setCopiedMessageId(messageId);
        setTimeout(() => {
          setCopiedMessageId(null);
        }, 2000); // Remove notification after 2 seconds
      })
      .catch(err => console.error('Failed to copy: ', err));
  }, []);
  // Get status message based on transcription status
  const getStatusMessage = useCallback(() => {
    switch (processingStage) {
      case 'socket-connecting':
        return 'Connecting to processing server...';
      case 'socket-analysis':
        return 'Starting video analysis...';
      case 'downloading-video':
        return 'Downloading YouTube video...';
      case 'transcribing-audio':
        return 'Transcribing audio...';
      case 'transcription-complete':
        return 'Transcription complete!';
      case 'transcription-error':
        return timeoutOccurred
          ? 'Timeout occurred during transcription. Try a shorter video.'
          : 'Error processing video.';
      default:
        break;
    }
    switch (transcriptionStatus) {
      case 'received':
        return 'Request received, starting processing...';
      case 'queued':
        return 'Video has been queued for processing...';
      case 'downloading':
        return 'Downloading YouTube video...';
      case 'extracting_audio':
        return 'Extracting audio from video...';
      case 'transcribing':
        return 'Transcribing audio...';
      case 'processing':
        return 'Processing transcription...';
      case 'loaded':
      case 'completed':
        return 'Transcription completed!';
      case 'error':
        return timeoutOccurred
          ? 'Timeout occurred during transcription. Try a shorter video.'
          : 'Error processing video.';
      default:
        return processingYoutubeUrl ? "Processing..." : "";
    }
  }, [processingStage, transcriptionStatus, timeoutOccurred, processingYoutubeUrl]);
  // Handle sidebar tab changes
  const handleTabChange = useCallback((tab) => {
    setActiveTab(tab);
    if (tab === 'topics' && videoTopics.length === 0) {
      getVideoTopicAnalysis?.();
    } else if (tab === 'timestamps' && currentVideoId && timestamps.length === 0 && !isLoadingTimestamps) {
      requestTimestamps(currentVideoId);
    }
    // Reset video error when switching to timestamps tab
    if (tab === 'timestamps') {
      setVideoPlayerError(null);
      setShowVideoPlayer(true);
      // Force refresh the YouTube iframe with a slight delay
      setTimeout(() => {
        const iframe = document.getElementById('youtube-player-iframe');
        if (iframe && videoData?.youtubeUrl) {
          const videoId = getYouTubeVideoId(videoData.youtubeUrl);
          if (videoId) {
            console.log("Refreshing YouTube iframe with ID after tab change:", videoId);
            iframe.src = `https://www.youtube.com/embed/${videoId}?enablejsapi=1&origin=${window.location.origin}`;
          }
        }
      }, 300);
    }
  }, [videoTopics, currentVideoId, timestamps, isLoadingTimestamps, getVideoTopicAnalysis, requestTimestamps, videoData]);
  // Handle YouTube iframe errors
  const handleYouTubeError = useCallback((error) => {
    console.error("YouTube iframe error:", error);
    setVideoPlayerError("Failed to load video. Please check if the video exists and is publicly available.");
    // Try to refresh the iframe with different parameters
    setTimeout(() => {
      const iframe = document.getElementById('youtube-player-iframe');
      if (iframe && videoData?.youtubeUrl) {
        const videoId = getYouTubeVideoId(videoData.youtubeUrl);
        if (videoId) {
          console.log("Attempting to refresh iframe after error");
          iframe.src = `https://www.youtube.com/embed/${videoId}?enablejsapi=1&origin=${window.location.origin}`;
        }
      }
    }, 1000);
  }, [videoData]);
  // Capture current frame
  const handleCaptureFrame = useCallback(() => {
    if (!videoPlayerRef.current || !videoPlayerRef.current.getInternalPlayer) {
      setVisualError("Video player not available. Please ensure the video is loaded.");
      return;
    }
    try {
      const player = videoPlayerRef.current.getInternalPlayer();
      if (!player) {
        setVisualError("Could not access video player");
        return;
      }
      const currentTime = player.getCurrentTime();
      if (typeof currentTime !== 'number') {
        setVisualError("Could not determine current video position");
        return;
      }
      const canvas = document.createElement('canvas');
      const video = player;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 360;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      try {
        const dataUrl = canvas.toDataURL('image/jpeg');
        setCapturedFrame({
          dataUrl,
          timestamp: currentTime,
          width: canvas.width,
          height: canvas.height
        });
        setUserInput("What can you see in this frame?");
        if (userInputRef.current) {
          userInputRef.current.focus();
        }
      } catch (canvasError) {
        console.error("Error creating canvas image:", canvasError);
        setVisualError("Could not capture frame. The video might be from a different domain.");
      }
    } catch (error) {
      console.error("Error capturing frame:", error);
      setVisualError("Failed to capture current frame");
    }
  }, []);
  // Format seconds into MM:SS or HH:MM:SS
  const formatTimeFromSeconds = useCallback((seconds) => {
    if (isNaN(seconds)) return '00:00';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
      return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }
  }, []);
  // Filter timestamps based on search query
  const getFilteredTimestamps = useCallback((timestamps) => {
    if (!timestampSearchQuery) return timestamps;
    const query = timestampSearchQuery.toLowerCase();
    return timestamps.filter(ts => {
      const text = ts.text?.toLowerCase() || '';
      return text.includes(query);
    });
  }, [timestampSearchQuery]);
  // Improved timestamp click handler with better normalization and reliability
  const handleTimestampClick = useCallback((timestamp) => {
    if (!timestamp) return;
    let timeInSeconds;
    if (typeof timestamp.time === 'number') {
      timeInSeconds = timestamp.time;
    } else if (typeof timestamp.start_time === 'number') {
      timeInSeconds = timestamp.start_time;
    } else if (typeof timestamp.time === 'string') {
      const timeParts = timestamp.time.split(':').map(Number);
      if (timeParts.length === 3) {
        timeInSeconds = (timeParts[0] * 3600) + (timeParts[1] * 60) + timeParts[2];
      } else if (timeParts.length === 2) {
        timeInSeconds = (timeParts[0] * 60) + timeParts[1];
      } else {
        timeInSeconds = Number(timestamp.time);
      }
    } else {
      console.error('Invalid timestamp format:', timestamp);
      return;
    }
    if (isNaN(timeInSeconds)) {
      console.error('Invalid timestamp value:', timestamp);
      return;
    }
    const normalizedTimestamp = {
      time: timeInSeconds,
      time_formatted: timestamp.time_formatted || timestamp.display_time || formatTimeFromSeconds(timeInSeconds)
    };
    console.log('Navigating to timestamp:', normalizedTimestamp);
    // Switch to timestamps tab
    setActiveTab('timestamps');
    if (!isSidebarVisible) {
      setIsSidebarVisible(true);
    }
    // Try to navigate to the timestamp in the embedded YouTube player
    const videoId = getYouTubeVideoId(videoData?.youtubeUrl);
    if (videoId) {
      const embeddedPlayer = document.querySelector('#youtube-player-iframe');
      if (embeddedPlayer) {
        try {
          // Update YouTube player to navigate to specific time
          embeddedPlayer.src = `https://www.youtube.com/embed/${videoId}?start=${Math.floor(timeInSeconds)}&autoplay=1&enablejsapi=1&origin=${window.location.origin}`;
        } catch (error) {
          console.error("Error navigating YouTube iframe to timestamp:", error);
        }
      }
    }
    setTimeout(() => {
      if (isComponentMounted.current) {
        navigateToTimestamp?.(normalizedTimestamp);
      }
    }, 50);
  }, [navigateToTimestamp, formatTimeFromSeconds, isSidebarVisible, videoData]);
  // Generate timeline markers based on timestamps
  const generateTimelineMarkers = useCallback(() => {
    if (!timestamps || timestamps.length === 0 || !videoDuration) return [];
    // Get only unique timestamp points (avoid cluttering timeline)
    const allPoints = timestamps.map(ts =>
      typeof ts.start_time === 'number' ? ts.start_time :
        typeof ts.time === 'number' ? ts.time : 0
    );
    // Limit to a reasonable number of markers
    const maxMarkers = 15;
    let selectedPoints = [];
    if (allPoints.length <= maxMarkers) {
      selectedPoints = [...allPoints];
    } else {
      // Select points evenly distributed across the timeline
      const step = allPoints.length / maxMarkers;
      for (let i = 0; i < maxMarkers; i++) {
        const index = Math.min(Math.floor(i * step), allPoints.length - 1);
        selectedPoints.push(allPoints[index]);
      }
    }
    // Sort points and remove duplicates
    selectedPoints = [...new Set(selectedPoints)].sort((a, b) => a - b);
    return selectedPoints.map((time, index) => {
      const percent = (time / videoDuration) * 100;
      // Find the corresponding timestamp for tooltip text
      const nearestTimestamp = timestamps.find(ts => {
        const tsTime = typeof ts.start_time === 'number' ? ts.start_time :
          typeof ts.time === 'number' ? ts.time : 0;
        return Math.abs(tsTime - time) < 1; // Within 1 second
      });
      const tooltipText = nearestTimestamp?.text || formatTimeFromSeconds(time);
      return (
        <div
          key={`marker-${index}`}
          className="timeline-marker"
          style={{ left: `${percent}%` }}
          data-time={formatTimeFromSeconds(time)}
          onClick={() => handleTimestampClick({ time })}
          onMouseEnter={() => {
            setTooltipIndex(index);
            setShowTimestampTooltip(true);
          }}
          onMouseLeave={() => {
            setShowTimestampTooltip(false);
          }}
          title={tooltipText}
        />
      );
    });
  }, [timestamps, videoDuration, handleTimestampClick, formatTimeFromSeconds]);
  // Handle direct timeline click for navigation
  const handleTimelineClick = useCallback((e) => {
    if (!timelineRef.current || !videoDuration) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const trackWidth = rect.width;
    const percentage = clickX / trackWidth;
    const timeInSeconds = percentage * videoDuration;
    handleTimestampClick({ time: timeInSeconds });
  }, [videoDuration, handleTimestampClick]);
  // Safe logout handler
  const handleLogout = useCallback(() => {
    console.log('User initiated logout');
    try {
      logout?.();
    } catch (error) {
      console.error('Error during logout:', error);
      navigate('/login');
    }
  }, [logout, navigate]);
  // Render timestamp item with new design
  const renderTimestampItem = useCallback((timestamp, index) => {
    const startTime = typeof timestamp.start_time === 'number' ? timestamp.start_time :
      typeof timestamp.time === 'number' ? timestamp.time : 0;
    const endTime = typeof timestamp.end_time === 'number' ? timestamp.end_time : startTime + 2;
    const startFormatted = formatTimeFromSeconds(startTime);
    const endFormatted = formatTimeFromSeconds(endTime);
    const displayTime = `${startFormatted} - ${endFormatted}`;
    const text = timestamp.text || "No text available";
    return (
      <div
        key={`timestamp-${index}`}
        className="timestamp-item"
        onClick={() => handleTimestampClick(timestamp)}
      >
        <div className="timestamp-time">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
          </svg>
          {startFormatted}
        </div>
        <div className="timestamp-text">{text}</div>
      </div>
    );
  }, [handleTimestampClick, formatTimeFromSeconds]);
  // Render timestamp options with new design
  const renderTimestampOptions = useCallback(() => {
    if (isLoadingTimestamps) {
      return (
        <div className="loading-indicator">
          <AILoadingIndicator message="Loading timestamps..." />
        </div>
      );
    }
    if (!timestamps || timestamps.length === 0) {
      if (currentVideoId && !hasTimestamps) {
        return (
          <div className="empty-timestamps">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
            <p>No timestamps available for this video</p>
            <button
              className="refresh-timestamps-btn"
              onClick={() => requestTimestamps(currentVideoId)}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"></path>
              </svg>
              Request Timestamps
            </button>
          </div>
        );
      }
      return (
        <div className="empty-timestamps">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
          </svg>
          <p>No timestamps available</p>
        </div>
      );
    }
    // If there's a search query or fewer than 10 timestamps, show a flat list
    if (timestampSearchQuery || Object.keys(groupedTimestamps).length <= 1 || timestamps.length < 10) {
      const filteredTimestamps = getFilteredTimestamps(timestamps);
      if (filteredTimestamps.length === 0) {
        return (
          <div className="empty-timestamps">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M11 17.25a6.25 6.25 0 110-12.5 6.25 6.25 0 010 12.5zM16 16l4.5 4.5"></path>
            </svg>
            <p>No timestamps match your search</p>
            <button
              className="refresh-timestamps-btn"
              onClick={() => setTimestampSearchQuery('')}
            >
              Clear Search
            </button>
          </div>
        );
      }
      return (
        <div className="timestamp-list">
          {filteredTimestamps.map((ts, index) => renderTimestampItem(ts, index))}
        </div>
      );
    }
    // Show grouped timestamps
    return (
      <div className="timestamp-list">
        {Object.entries(groupedTimestamps).map(([groupKey, groupTimestamps], groupIndex) => {
          const filteredGroupTimestamps = getFilteredTimestamps(groupTimestamps);
          if (filteredGroupTimestamps.length === 0) return null;
          return (
            <div key={`group-${groupIndex}`} className="timestamp-group">
              <div className="timestamp-group-header">
                <div className="timestamp-group-title">{groupKey}</div>
                <div className="timestamp-group-line"></div>
              </div>
              {filteredGroupTimestamps.map((ts, tsIndex) => renderTimestampItem(ts, `${groupIndex}-${tsIndex}`))}
            </div>
          );
        })}
      </div>
    );
  }, [
    isLoadingTimestamps,
    timestamps,
    currentVideoId,
    hasTimestamps,
    requestTimestamps,
    timestampSearchQuery,
    groupedTimestamps,
    getFilteredTimestamps,
    renderTimestampItem
  ]);
  // Render quick prompts buttons for common questions
  const renderQuickPrompts = useCallback(() => {
    const prompts = [
      { text: "Summarize", prompt: "Please summarize this video" },
      { text: "Key points", prompt: "What are the key points in this video?" },
      { text: "Topics", prompt: "What topics are covered in this video?" },
    ];
    return (
      <div className="quick-prompts">
        {prompts.map((item, index) => (
          <button
            key={index}
            className="prompt-button"
            onClick={() => {
              setUserInput(item.prompt);
              setTimeout(() => {
                if (isComponentMounted.current) {
                  sendMessage();
                }
              }, 300);
            }}
          >
            {item.text}
          </button>
        ))}
      </div>
    );
  }, [sendMessage]);
  // Show loading indicator if loading and no specific status message
  if (loading && !processingStatus && !processingYoutubeUrl) {
    return (
      <div className="loading-container">
        <div className="loading-content">
          <AILoadingIndicator message="Loading..." />
        </div>
      </div>
    );
  }
  return (
    <div className="video-analysis-container">
      {/* Page-level drop indicator */}
      {dragActive && showInputForm && (
        <div className="page-level-drop-indicator">
          <div className="drop-indicator-content">
            <svg className="drop-indicator-icon" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
            </svg>
            <span className="drop-indicator-text">Drop your file here</span>
          </div>
        </div>
      )}
      {/* Left sidebar with transcription */}
      <div className={`sidebar ${!isSidebarVisible ? 'sidebar-hidden' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <span className="sidebar-logo-icon">✨</span>
            Luna AI
            <button
              className="sidebar-toggle"
              onClick={toggleSidebar}
              aria-label="Toggle sidebar"
              style={{ marginLeft: '200px', marginTop: '27px' }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M11 19l-7-7 7-7m8 14l-7-7 7-7"></path>
              </svg>
            </button>
          </div>
          <div className="sidebar-controls">
            <button
              className="theme-toggle-btn"
              onClick={toggleTheme}
              aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
            >
              {isDarkMode ? (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"></path>
                </svg>
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"></path>
                </svg>
              )}
            </button>
          </div>
        </div>
        {/* Tabs for transcription, etc. */}
        <div className="tab-navigation">
          <button
            className={`tab-button ${activeTab === 'transcription' ? 'active' : ''}`}
            onClick={() => handleTabChange('transcription')}
          >
            <span className="tab-content">
              <svg className="tab-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
              </svg>
              Transcript
            </span>
          </button>
          <button
            className={`tab-button ${activeTab === 'timestamps' ? 'active' : ''}`}
            onClick={() => handleTabChange('timestamps')}
          >
            <span className="tab-content">
              <svg className="tab-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
              </svg>
              Time
            </span>
          </button>
        </div>
        {/* Transcription content */}
        <div className="tab-content-container">
          {activeTab === 'transcription' && (
            <div className="transcript-container" ref={transcriptContainerRef}>
              {(processingYoutubeUrl || transcriptionLoading) && !transcription ? (
                <div className="transcript-loading">
                  <AILoadingIndicator message={processingStatus || 'Processing transcription...'} />
                </div>
              ) : (
                <div
                  className="transcript-content"
                  onClick={(e) => {
                    const text = window.getSelection().toString();
                    const timestampMatch = text.match(/(?:(\d{1,2}):)?(\d{1,2}):(\d{2})/);
                    if (timestampMatch) {
                      let hours = 0;
                      let minutes = 0;
                      let seconds = 0;
                      if (timestampMatch[1] !== undefined) {
                        hours = parseInt(timestampMatch[1], 10);
                        minutes = parseInt(timestampMatch[2], 10);
                        seconds = parseInt(timestampMatch[3], 10);
                      } else {
                        minutes = parseInt(timestampMatch[2], 10);
                        seconds = parseInt(timestampMatch[3], 10);
                      }
                      const time = (hours * 3600) + (minutes * 60) + seconds;
                      console.log(`Detected timestamp: ${timestampMatch[0]} -> ${time}s`);
                      handleTimestampClick({
                        time: time,
                        time_formatted: timestampMatch[0]
                      });
                    }
                  }}
                >
                  {transcription || (
                    <div className="transcript-empty">
                      <svg className="transcript-empty-icon" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                      </svg>
                      <p>No transcription available yet for this video.</p>
                      {transcriptionDebug.isLoading && (
                        <div className="transcript-loading-indicator">
                          <AILoadingIndicator message="Transcription is still loading..." />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
          {/* Timestamps tab content - IMPROVED IMPLEMENTATION */}
          {activeTab === 'timestamps' && (
            <div className="timestamps-container">
              <div className="timestamps-header">
                <h3 className="timestamps-title">Video Timestamps</h3>
                <p className="timestamps-description">
                  Click on a timestamp to jump to that point in the video
                </p>
              </div>
              {/* Debug output for development */}
              {process.env.NODE_ENV === 'development' && (
                <div id="video-id-debug" className="debug-output">
                  Video ID: {getYouTubeVideoId(videoData?.youtubeUrl) || 'Not detected'}
                </div>
              )}
              {/* Embed the video at the top of timestamps tab - FIXED IMPLEMENTATION */}
              {videoData?.youtubeUrl ? (
                <div className="video-timestamp-player">
                  <div className="video-player-container">
                    {videoPlayerError ? (
                      <div className="video-error">
                        <p>
                          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="8" x2="12" y2="12"></line>
                            <line x1="12" y1="16" x2="12.01" y2="16"></line>
                          </svg>
                          {videoPlayerError}
                        </p>
                        <button
                          onClick={() => {
                            setVideoPlayerError(null);
                            const videoId = getYouTubeVideoId(videoData.youtubeUrl);
                            if (videoId) {
                              const iframe = document.getElementById('youtube-player-iframe');
                              if (iframe) {
                                iframe.src = `https://www.youtube.com/embed/${videoId}?enablejsapi=1&origin=${window.location.origin}`;
                              }
                            }
                          }}
                          className="video-retry-button"
                        >
                          Try Again
                        </button>
                      </div>
                    ) : (
                      <iframe
                        id="youtube-player-iframe"
                        width="100%"
                        height="100%"
                        src={`https://www.youtube.com/embed/${getYouTubeVideoId(videoData.youtubeUrl) || ''}?enablejsapi=1&origin=${window.location.origin}`}
                        title="YouTube video player"
                        frameBorder="0"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowFullScreen
                        onError={handleYouTubeError}
                      ></iframe>
                    )}
                  </div>
                </div>
              ) : (
                <div className="video-placeholder">
                  <p>No video URL available</p>
                </div>
              )}
              {/* Interactive Timeline */}
              {!isLoadingTimestamps && timestamps?.length > 0 && (
                <div className="timeline-container">
                  <div
                    className="timeline-track"
                    ref={timelineRef}
                    onClick={handleTimelineClick}
                  >
                    <div
                      className="timeline-progress"
                      style={{ width: `${currentProgress}%` }}
                    ></div>
                  </div>
                  <div className="timeline-markers">
                    {generateTimelineMarkers()}
                  </div>
                  {/* Current time indicator */}
                  <div className="current-time-display">
                    {formatTimeFromSeconds(videoDuration * (currentProgress / 100))}
                  </div>
                </div>
              )}
              {/* Search input for timestamps */}
              <div className="timestamps-search">
                <svg className="timestamps-search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8"></circle>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
                <input
                  type="text"
                  className="timestamps-search-input"
                  placeholder="Search timestamps..."
                  value={timestampSearchQuery}
                  onChange={(e) => setTimestampSearchQuery(e.target.value)}
                />
              </div>
              {/* Timestamp list */}
              {renderTimestampOptions()}
            </div>
          )}
        </div>
        {/* User info section - Updated with greeting */}
        <div className="user-info">
          <div className="user-avatar">
            {user?.fullname?.charAt(0) || 'U'}
          </div>
          <div className="user-info-text">
            <div className="user-greeting">Hello,</div>
            <div className="user-name">
              {user?.fullname || 'User'}
            </div>
          </div>
          <button
            className="logout-button"
            onClick={handleLogout}
            aria-label="Logout"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"></path>
            </svg>
          </button>
        </div>
      </div>
      {/* Main content area */}
      <div className="main-content">
        {/* Toggle sidebar button when sidebar is hidden */}
        {!isSidebarVisible && (
          <button
            className="sidebar-toggle-button"
            onClick={toggleSidebar}
            aria-label="Show sidebar"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 6h16M4 12h16M4 18h16"></path>
            </svg>
          </button>
        )}
        {/* YouTube URL / File Upload Form */}
        {showInputForm ? (
          <div className="upload-form-container">
            <div className="upload-form">
              <h1 className="upload-title">Analyze a Video</h1>
              <p className="upload-subtitle">Upload a video file or enter a YouTube URL to analyze</p>
              <div className="form-group">
                <label className="form-label">YouTube URL</label>
                <div className="url-input-wrapper">
                  <div className="url-input-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"></path>
                    </svg>
                  </div>
                  <input
                    type="text"
                    ref={youtubeInputRef}
                    value={youtubeUrl}
                    onChange={(e) => {
                      setYoutubeUrl(e.target.value);
                      if (urlError) setUrlError('');
                    }}
                    onBlur={() => {
                      if (youtubeUrl && !isValidYouTubeUrl(youtubeUrl)) {
                        setUrlError('Please enter a valid YouTube URL');
                      }
                    }}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && youtubeUrl) {
                        handleProcessVideo();
                      }
                    }}
                    placeholder="https://www.youtube.com/watch?v=..."
                    className={`url-input ${urlError ? 'error' : ''}`}
                    disabled={processingYoutubeUrl}
                  />
                </div>
                {urlError && (
                  <div className="error-message">
                    <svg className="error-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <span>{urlError}</span>
                  </div>
                )}
                <div className="input-tip">
                  <svg className="tip-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                  <span>For best results, use YouTube Shorts or video clips under 5 minutes</span>
                </div>
              </div>
              {/* Modified file upload section with drag and drop functionality */}
              <div className="form-group">
                <label className="form-label">OR Upload a Video or Audio File</label>
                <div
                  className={`file-upload-container ${dragActive ? "drag-active" : ""}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    id="video-file"
                    accept="video/*,audio/*"
                    onChange={(e) => setSelectedFile(e.target.files[0])}
                    className="hidden-file-input"
                    disabled={processingYoutubeUrl}
                  />
                  <label
                    htmlFor="video-file"
                    className="file-upload-area"
                  >
                    <svg className="file-upload-icon" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                    </svg>
                    <span className={`file-upload-text ${selectedFile ? 'has-file' : ''}`}>
                      {selectedFile ? selectedFile.name : 'Choose a video or audio file or drag and drop'}
                    </span>
                  </label>
                  {dragActive && (
                    <div
                      className="drag-file-element"
                      onDragEnter={handleDrag}
                      onDragLeave={handleDrag}
                      onDragOver={handleDrag}
                      onDrop={handleDrop}
                    ></div>
                  )}
                </div>
              </div>
              <button
                onClick={handleProcessVideo}
                disabled={loading || processingYoutubeUrl || (!youtubeUrl && !selectedFile)}
                className={`submit-button ${loading || processingYoutubeUrl || (!youtubeUrl && !selectedFile) ? 'disabled' : ''}`}
              >
                {processingYoutubeUrl ? (
                  <>
                    <AILoadingIndicator message="Processing..." />
                  </>
                ) : (
                  <>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="submit-icon">
                      <path d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path>
                      <path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <span>Analyze {selectedFile && isAudioFile(selectedFile) ? 'Audio' : 'Video'}</span>
                  </>
                )}
              </button>
              {error && !timeoutOccurred && !urlError && (
                <div className="error-status">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="error-status-icon">
                    <path d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                  <span>{error}</span>
                </div>
              )}
              {processingStatus && !timeoutOccurred && (
                <div className="processing-status">
                  <AILoadingIndicator message={processingStatus} />
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Chat interface */
          <div className="chat-interface">
            {/* Ask Luna header */}
            <div className="main-header">
              Ask Luna
            </div>
            {/* Captured frame display */}
            {capturedFrame && (
              <div className="captured-frame-container">
                <div className="captured-frame-header">
                  <span className="captured-frame-timestamp">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10"></circle>
                      <polyline points="12 6 12 12 16 14"></polyline>
                    </svg>
                    Captured at {formatTimeFromSeconds(capturedFrame.timestamp)}
                  </span>
                  <button
                    className="close-frame-button"
                    onClick={() => setCapturedFrame(null)}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                  </button>
                </div>
                <div className="captured-frame-image-container">
                  <img
                    src={capturedFrame.dataUrl}
                    alt={`Frame at ${formatTimeFromSeconds(capturedFrame.timestamp)}`}
                    className="captured-frame-image"
                  />
                </div>
              </div>
            )}
            {/* Chat area - UPDATED MESSAGE RENDERING for improved copy button */}
            <div className="chat-area" ref={chatAreaRef}>
              {conversations.length === 0 ? (
                <div className="welcome-container">
                  <div className="welcome-icon">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"></path>
                    </svg>
                  </div>
                  <h2 className="welcome-title">How can I help you with this video?</h2>
                  <p className="welcome-description">Ask me anything about the content, including what's visually shown or what happens at specific timestamps.</p>
                  {renderQuickPrompts()}
                </div>
              ) : (
                <>
                  {conversations.map((message, index) => (
                    <div
                      key={index}
                      className={`message-container ${message.role === 'user' ? 'user' : 'ai'}`}
                    >
                      <div className={`message-bubble ${message.role === 'user' ? 'user' : 'ai'}`}>
                        <div className={`message-header ${message.role === 'user' ? 'user' : 'ai'}`}>
                          {/* Header content, if needed */}
                        </div>
                        <div className="message-content">
                          {message.content}
                        </div>
                        {message.role === 'ai' && (
                          <button
                            className={`copy-button ${copiedMessageId === index ? 'copied' : ''}`}
                            onClick={() => copyToClipboard(message.content, index)}
                            aria-label="Copy to clipboard"
                          >
                            {copiedMessageId === index ? (
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M20 6L9 17l-5-5"></path>
                              </svg>
                            ) : (
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                              </svg>
                            )}
                            <span className={`copy-tooltip ${copiedMessageId === index ? 'visible' : ''}`}>
                              Copied!
                            </span>
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                  {/* AI thinking indicator - UPDATED to remove avatar */}
                  {isAIThinking && (
                    <div className="message-container ai">
                      <div className="message-bubble ai">
                        <div className="message-header ai">
                          {/* Avatar removed here */}
                        </div>
                        <div className="thinking-indicator">
                          <div className="thinking-dots">
                            <div className="thinking-dot"></div>
                            <div className="thinking-dot"></div>
                            <div className="thinking-dot"></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
            {/* Input area */}
            <div className="input-container">
              <div className="input-wrapper">
                {/* Detected timestamp bubble */}
                {showTimestampBubble && detectedTimestamp && (
                  <div className="timestamp-bubble">
                    <svg className="timestamp-bubble-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10"></circle>
                      <polyline points="12 6 12 12 16 14"></polyline>
                    </svg>
                    <span>Found timestamp: {detectedTimestamp.formatted}</span>
                  </div>
                )}
                <textarea
                  ref={userInputRef}
                  value={userInput}
                  onChange={(e) => {
                    setUserInput(e.target.value);
                    autoResizeInput();
                  }}
                  onKeyDown={handleKeyPress}
                  placeholder={capturedFrame ? "Ask about this frame..." : "Ask about the video or a specific timestamp..."}
                  rows="1"
                  disabled={isAIThinking}
                  className="input-textarea"
                />
                <button
                  className={`send-button ${isAIThinking || !userInput.trim() ? 'disabled' : ''}`}
                  onClick={sendMessage}
                  disabled={isAIThinking || !userInput.trim()}
                  aria-label="Send message"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                  </svg>
                </button>
              </div>
              {/* Video controls when player is hidden */}
              <div className="video-controls-bar">
                <button
                  className="video-control"
                  onClick={toggleVideoPlayer}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                  </svg>
                  Show Video
                </button>
                <button
                  className="video-control"
                  onClick={handleResetState}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                  </svg>
                  New Video
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
export default VideoAnalysis;

