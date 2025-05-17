// src/components/TranscriptionDisplay.jsx
import React, { useEffect, useState, useRef } from 'react';
import { useVideo } from '../context/VideoContext';

const TranscriptionDisplay = () => {
  const { 
    transcription, 
    transcriptionLoading, 
    transcriptionStatus,
    currentVideoId,
    fetchTranscription
  } = useVideo();
  
  const [localTranscription, setLocalTranscription] = useState('');
  const [retryCount, setRetryCount] = useState(0);
  const [manualFetchStarted, setManualFetchStarted] = useState(false);
  const [pollAttempts, setPollAttempts] = useState(0);
  
  // For debugging - remove in production
  const debugRef = useRef({
    lastTranscription: null,
    lastTranscriptionLength: 0,
    attempts: 0
  });
  
  // Debug logging to diagnose the issue
  useEffect(() => {
    console.log("TranscriptionDisplay state:", {
      hasContextTranscription: !!transcription,
      contextTranscriptionLength: transcription?.length || 0,
      hasLocalTranscription: !!localTranscription,
      localTranscriptionLength: localTranscription?.length || 0,
      status: transcriptionStatus,
      isLoading: transcriptionLoading,
      videoId: currentVideoId
    });
    
    // Update debug ref
    debugRef.current = {
      ...debugRef.current,
      lastTranscription: transcription,
      lastTranscriptionLength: transcription?.length || 0,
      attempts: debugRef.current.attempts + 1
    };
  }, [transcription, localTranscription, transcriptionStatus, transcriptionLoading, currentVideoId]);
  
  // Use transcription from context if available - with additional checks
  useEffect(() => {
    // Crucial fix: Only update if we have actual content and it's different
    if (transcription && (!localTranscription || transcription !== localTranscription)) {
      console.log(`Setting local transcription from context (${transcription.length} chars)`);
      setLocalTranscription(transcription);
    }
  }, [transcription, localTranscription]);
  
  // Handle manual fetch
  const handleManualFetch = async () => {
    // Prevent multiple fetches
    if (manualFetchStarted) return;
    
    setManualFetchStarted(true);
    setRetryCount(prev => prev + 1);
    
    try {
      console.log('Manually fetching transcription...');
      
      if (currentVideoId) {
        await fetchTranscription(currentVideoId);
        
        // Added timeout to ensure state updates
        setTimeout(() => {
          // Check if we received data from context after fetch
          if (transcription) {
            console.log(`Manual fetch successful, got ${transcription.length} chars`);
            setLocalTranscription(transcription);
          } else {
            console.log('Manual fetch completed but no transcription received');
          }
          
          // Reset fetch state
          setManualFetchStarted(false);
        }, 1000);
      } else {
        console.error('No video ID available for fetching');
        setManualFetchStarted(false);
      }
    } catch (error) {
      console.error('Error manually fetching transcription:', error);
      setManualFetchStarted(false);
    }
  };
  
  // Polling mechanism for transcription - simplified
  useEffect(() => {
    // Skip all this if we already have transcription data
    if (transcription || localTranscription) {
      return;
    }
    
    // Don't start polling if no video ID
    if (!currentVideoId) return;
    
    // Don't poll if manual fetch already started
    if (manualFetchStarted) return;
    
    // Don't continue if max attempts reached
    const maxAttempts = 10;
    if (pollAttempts >= maxAttempts) return;
    
    // Set up polling timer
    console.log(`Setting up polling (attempt ${pollAttempts + 1}/${maxAttempts})`);
    const pollTimer = setTimeout(async () => {
      console.log(`Executing poll attempt ${pollAttempts + 1}`);
      
      try {
        // Try to fetch transcription
        await fetchTranscription(currentVideoId);
        
        // Check if we got data
        if (transcription) {
          console.log(`Poll successful, received ${transcription.length} chars`);
          setLocalTranscription(transcription);
        } else {
          // Increment attempts only if we didn't get data
          setPollAttempts(current => current + 1);
        }
      } catch (error) {
        console.error('Error during transcription poll:', error);
        setPollAttempts(current => current + 1);
      }
    }, 5000); // Poll every 5 seconds
    
    // Clean up timer
    return () => clearTimeout(pollTimer);
  }, [
    currentVideoId, 
    transcription, 
    localTranscription, 
    manualFetchStarted, 
    pollAttempts, 
    fetchTranscription
  ]);
  
  // CRITICAL FIXES: Improved rendering logic
  
  // Display transcription content if we have it in either state
  const displayText = transcription || localTranscription;
  
  // If we have content to display, show it immediately regardless of loading state
  if (displayText) {
    return (
      <div className="transcription-display">
        <div className="transcription-header">
          <h3>Transcription</h3>
          {transcriptionLoading && <span className="loading-indicator">Updating...</span>}
        </div>
        <div className="transcription-content">
          <pre className="transcription-text">{displayText}</pre>
        </div>
      </div>
    );
  }
  
  // Still loading, no content yet
  if (transcriptionLoading) {
    return (
      <div className="transcription-display transcription-loading">
        <div className="transcription-spinner"></div>
        <p>Transcribing your video...</p>
        
        {/* Show retry button after some time has passed */}
        {retryCount < 3 && pollAttempts > 2 && (
          <button 
            onClick={handleManualFetch} 
            disabled={manualFetchStarted}
            className="transcription-retry-button"
          >
            {manualFetchStarted ? 'Fetching...' : 'Retry Fetching Transcription'}
          </button>
        )}
      </div>
    );
  }
  
  // No content and not loading
  return (
    <div className="transcription-display transcription-empty">
      <p>No transcription available yet.</p>
      <button 
        onClick={handleManualFetch}
        disabled={manualFetchStarted}
        className="transcription-fetch-button"
      >
        {manualFetchStarted ? 'Fetching...' : 'Fetch Transcription'}
      </button>
    </div>
  );
};

export default TranscriptionDisplay;