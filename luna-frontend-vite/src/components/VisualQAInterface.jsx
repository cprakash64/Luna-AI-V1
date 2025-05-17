import React, { useState, useEffect, forwardRef, useRef, useCallback } from 'react';
import { useVideo } from '../context/VideoContext';
import { marked } from 'marked';
import '../styles/videoAnalysis.css';

/**
 * Enhanced component for visual question answering interface
 * Allows users to ask questions about visual content in videos with improved timestamp understanding
 * and color analysis capabilities
 */
const VisualQAInterface = forwardRef(({ videoId }, videoRef) => {
  const {
    sendMessageToAI,
    isAIThinking,
    addMessage,
    conversations,
    visualAnalysisAvailable,
    visualAnalysisInProgress,
    requestVisualAnalysis,
    currentVideoId,
    navigateToTimestamp,
    detectedScenes,
    videoHighlights,
    fetchVisualData: contextFetchVisualData, // Renamed to avoid confusion
    socket
  } = useVideo();
  
  const [question, setQuestion] = useState('');
  const [visualQuestions, setVisualQuestions] = useState([
    'What objects appear in this video?',
    'What is happening at 0:15?',
    'What colors are most prominent in this video?',
    'When does the main subject first appear?',
    'Is there a bull painting in this video?'
  ]);
  
  const [showTimestampPicker, setShowTimestampPicker] = useState(false);
  const [selectedTime, setSelectedTime] = useState('');
  const [recentTimestamps, setRecentTimestamps] = useState([]);
  const [isImageBasedQuery, setIsImageBasedQuery] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [colorAnalysisMode, setColorAnalysisMode] = useState(false);
  const [showObjectsList, setShowObjectsList] = useState(false);
  const [detectedObjects, setDetectedObjects] = useState([]);
  
  const questionInputRef = useRef(null);
  
  // Create a local implementation of fetchVisualData in case it's not provided by context
  const fetchVisualData = useCallback(async (vidId) => {
    if (!vidId) return;
    
    // Use the context function if available
    if (typeof contextFetchVisualData === 'function') {
      return contextFetchVisualData(vidId);
    }
    
    // Local implementation as fallback
    console.log("Using fallback fetchVisualData implementation for video:", vidId);
    
    try {
      // Try API endpoint first
      const apiUrl = `/api/v1/videos/${vidId}/visual-data`;
      const response = await fetch(apiUrl);
      
      if (response.ok) {
        const data = await response.json();
        return data;
      } else {
        // Fallback to socket.io
        return requestVisualAnalysisViaSocket(vidId);
      }
    } catch (error) {
      console.error("Error fetching visual data:", error);
      return requestVisualAnalysisViaSocket(vidId);
    }
  }, [contextFetchVisualData]);
  
  // Socket.io fallback for visual analysis
  const requestVisualAnalysisViaSocket = useCallback((vidId) => {
    if (!socket || !vidId) return null;
    
    return new Promise((resolve) => {
      // Try multiple event names to ensure one works
      const eventNames = [
        'analyze_visual', 
        'request_visual_analysis',
        'analyze_visual_content',
        'process_visual'
      ];
      
      // Send multiple events to increase chance of success
      eventNames.forEach(eventName => {
        socket.emit(eventName, {
          video_id: vidId,
          tab_id: window.tabId || 'default_tab',
          timestamp: Date.now()
        });
      });
      
      // Set up a listener for the response
      const handleVisualData = (data) => {
        if (data.video_id === vidId) {
          socket.off('visual_analysis_data', handleVisualData);
          resolve(data);
        }
      };
      
      // Listen for the response
      socket.on('visual_analysis_data', handleVisualData);
      
      // Timeout after 5 seconds
      setTimeout(() => {
        socket.off('visual_analysis_data', handleVisualData);
        resolve(null); // Resolve with null on timeout
      }, 5000);
    });
  }, [socket]);
  
  // Load visual data if available
  useEffect(() => {
    if (currentVideoId && !visualAnalysisAvailable && !visualAnalysisInProgress) {
      console.log("Auto-requesting visual analysis for video:", currentVideoId);
      fetchVisualData(currentVideoId).catch(err => {
        console.warn("Failed to fetch visual data:", err);
      });
    }
  }, [currentVideoId, visualAnalysisAvailable, visualAnalysisInProgress, fetchVisualData]);
  
  // Extract timestamps from scenes for the timestamp picker
  useEffect(() => {
    if (detectedScenes && detectedScenes.length > 0) {
      const timestamps = detectedScenes.map(scene => ({
        time: scene.start_time,
        formattedTime: scene.start_time_str,
        description: scene.description || "Scene start"
      }));
      
      // Add highlights as timestamps
      if (videoHighlights && videoHighlights.length > 0) {
        videoHighlights.forEach(highlight => {
          timestamps.push({
            time: highlight.timestamp,
            formattedTime: highlight.timestamp_str,
            description: highlight.description || "Highlight moment"
          });
        });
      }
      
      // Sort by time and remove duplicates
      const uniqueTimestamps = timestamps
        .sort((a, b) => a.time - b.time)
        .filter((ts, index, self) => 
          index === self.findIndex(t => t.time === ts.time)
        );
      
      setRecentTimestamps(uniqueTimestamps);
    }
  }, [detectedScenes, videoHighlights]);
  
  // Process user's question about visual content with improved timestamp handling
  const handleVisualQuestion = async (questionText) => {
    if (!questionText.trim() || isAIThinking) return;
    
    try {
      // Add user message to conversation
      addMessage('user', questionText);
      
      // Check if question is timestamp-specific
      const timestampMatch = questionText.match(/at\s+(\d+:?\d*)/i);
      let enhancedQuestion = questionText;
      
      if (timestampMatch && selectedTime) {
        // If specific timestamp mentioned, enhance the question
        enhancedQuestion = `${questionText} (specifically focus on what's happening at timestamp ${selectedTime})`;
        
        // Reset selected time
        setSelectedTime('');
        setShowTimestampPicker(false);
        
        // If there's a video reference, navigate to the timestamp
        if (videoRef && videoRef.current) {
          const timeComponents = selectedTime.split(':').map(Number);
          const timeInSeconds = timeComponents.length > 1 
            ? timeComponents[0] * 60 + timeComponents[1]
            : parseInt(timeComponents[0], 10);
            
          navigateToTimestamp({
            time: timeInSeconds,
            time_formatted: selectedTime
          });
        }
      } 
      else if (isImageBasedQuery && currentFrame) {
        // For image-based queries, include frame context
        enhancedQuestion = `${questionText} (specifically about the visual content at ${currentFrame.timestamp_formatted})`;
        
        // Navigate to the frame timestamp
        if (videoRef && videoRef.current) {
          navigateToTimestamp({
            time: currentFrame.timestamp,
            time_formatted: currentFrame.timestamp_formatted
          });
        }
        
        // Reset image query state
        setIsImageBasedQuery(false);
        setCurrentFrame(null);
      }
      else if (colorAnalysisMode) {
        // For color analysis queries
        enhancedQuestion = `${questionText} (focus on colors and visual appearance in the video)`;
        setColorAnalysisMode(false);
      }
      else {
        // For general visual questions, enhance with visual context request
        enhancedQuestion = `${questionText} (focus on what is visually shown in the video)`;
      }
      
      // Send question to AI
      await sendMessageToAI(enhancedQuestion);
      
      // Clear input
      setQuestion('');
    } catch (error) {
      console.error('Error asking visual question:', error);
    }
  };
  
  // Handle timestamp click
  const handleTimestampClick = (timestamp) => {
    if (videoRef && videoRef.current && timestamp && timestamp.time) {
      navigateToTimestamp(timestamp);
    }
  };
  
  // Handle timestamp selection
  const handleTimestampSelect = (timestamp) => {
    // Format the time as MM:SS
    const minutes = Math.floor(timestamp.time / 60);
    const seconds = Math.floor(timestamp.time % 60);
    const formattedTime = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    
    setSelectedTime(formattedTime);
    
    // Add the timestamp reference to the question
    if (questionInputRef.current && question.trim()) {
      // Only add "at X:XX" if not already present
      if (!question.toLowerCase().includes(' at ')) {
        setQuestion(`${question} at ${formattedTime}`);
      } else {
        // Replace existing timestamp
        setQuestion(question.replace(/at\s+\d+:?\d*/i, `at ${formattedTime}`));
      }
    } else if (questionInputRef.current) {
      // If no question yet, set a default
      setQuestion(`What is happening at ${formattedTime}?`);
    }
    
    setShowTimestampPicker(false);
  };
  
  // Handle current frame capture for image-based questions
  const handleCaptureFrame = () => {
    if (videoRef && videoRef.current) {
      const captureMethod = videoRef.current.captureCurrentFrame || 
                            (videoRef.current.current && videoRef.current.current.captureCurrentFrame);
      
      if (typeof captureMethod === 'function') {
        const capturedFrame = captureMethod();
        if (capturedFrame) {
          setCurrentFrame({
            dataUrl: capturedFrame.dataUrl,
            timestamp: capturedFrame.timestamp,
            timestamp_formatted: formatTimeFromSeconds(capturedFrame.timestamp)
          });
          setIsImageBasedQuery(true);
          
          // Set default question for the current frame
          setQuestion(`What is shown in this frame?`);
          
          // If we have object detection capability, detect objects in this frame
          const detectMethod = videoRef.current.detectObjectsInFrame || 
                              (videoRef.current.current && videoRef.current.current.detectObjectsInFrame);
                              
          if (typeof detectMethod === 'function') {
            detectMethod(capturedFrame.dataUrl)
              .then(objects => {
                if (objects && objects.length > 0) {
                  setDetectedObjects(objects);
                  setShowObjectsList(true);
                }
              })
              .catch(err => console.error('Error detecting objects:', err));
          }
        }
      } else {
        console.warn("captureCurrentFrame method not available on videoRef");
      }
    }
  };
  
  // Toggle color analysis mode
  const toggleColorAnalysisMode = () => {
    setColorAnalysisMode(!colorAnalysisMode);
    if (!colorAnalysisMode) {
      setQuestion('What colors are most prominent in this video?');
    }
  };
  
  // Format seconds to MM:SS
  const formatTimeFromSeconds = (seconds) => {
    if (typeof seconds !== 'number' || isNaN(seconds)) {
      return "00:00";
    }
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };
  
  // Generate timestamp options
  const renderTimestampOptions = () => {
    if (!recentTimestamps || recentTimestamps.length === 0) {
      return <p>No timestamps available</p>;
    }
    
    return (
      <div className="visual-qa-timestamps">
        {recentTimestamps.map((ts, index) => (
          <button 
            key={index} 
            className="timestamp-option"
            onClick={() => handleTimestampSelect(ts)}
          >
            {ts.formattedTime} - {ts.description}
          </button>
        ))}
      </div>
    );
  };
  
  // Render detected objects list
  const renderDetectedObjects = () => {
    if (!detectedObjects || detectedObjects.length === 0) {
      return null;
    }
    
    return (
      <div className="detected-objects-list">
        <div className="detected-objects-header">
          <h4>Detected Objects</h4>
          <button 
            className="close-button"
            onClick={() => setShowObjectsList(false)}
          >
            ✕
          </button>
        </div>
        <ul>
          {detectedObjects.map((obj, index) => (
            <li key={index} className="detected-object-item">
              <span className="object-label">{obj.label}</span>
              <span className="object-confidence">{Math.round(obj.confidence * 100)}%</span>
              {obj.dominant_color && (
                <span 
                  className="object-color"
                  style={{ backgroundColor: obj.dominant_color }}
                  title={`Dominant color: ${obj.dominant_color}`}
                />
              )}
            </li>
          ))}
        </ul>
        <button 
          className="ask-about-objects-button"
          onClick={() => {
            const objectLabels = detectedObjects.map(obj => obj.label).join(', ');
            setQuestion(`Tell me about the ${objectLabels} in this frame`);
            setShowObjectsList(false);
          }}
        >
          Ask about these objects
        </button>
      </div>
    );
  };
  
  // If visual analysis is not available, show option to run it
  if (!visualAnalysisAvailable && !visualAnalysisInProgress) {
    return (
      <div className="visual-qa-container">
        <div className="visual-qa-header">
          <h3>Visual Understanding</h3>
        </div>
        <div className="visual-qa-content">
          <p>
            To enable visual question answering, Luna needs to analyze the visual
            content of your video first.
          </p>
          <button
            className="analysis-button"
            onClick={() => {
              console.log("Manual request for visual analysis");
              if (typeof requestVisualAnalysis === 'function') {
                requestVisualAnalysis();
              }
              // Also try to fetch any existing data
              fetchVisualData(currentVideoId);
            }}
            disabled={!currentVideoId}
          >
            Analyze Visual Content
          </button>
        </div>
      </div>
    );
  }
  
  // Show loading state while analysis is in progress
  if (visualAnalysisInProgress) {
    return (
      <div className="visual-qa-container">
        <div className="visual-qa-header">
          <h3>Analyzing Visual Content</h3>
        </div>
        <div className="visual-qa-content">
          <div className="loader-container">
            <div className="loader"></div>
            <p>
              Analyzing video frames and objects... This may take a few minutes
              depending on the video length.
            </p>
          </div>
        </div>
      </div>
    );
  }
  
  // Show the main question answering interface
  return (
    <div className="visual-qa-container">
      <div className="visual-qa-header">
        <h3>Ask About What's in the Video</h3>
      </div>
      <div className="visual-qa-content">
        {isImageBasedQuery && currentFrame && (
          <div className="current-frame-container">
            <div className="current-frame-header">
              <h4>Asking about this frame: {currentFrame.timestamp_formatted}</h4>
              <button 
                className="cancel-frame-button"
                onClick={() => {
                  setIsImageBasedQuery(false);
                  setCurrentFrame(null);
                  setShowObjectsList(false);
                  setDetectedObjects([]);
                }}
              >
                ✕
              </button>
            </div>
            <img 
              src={currentFrame.dataUrl} 
              alt={`Frame at ${currentFrame.timestamp_formatted}`}
              className="current-frame-image"
            />
            {showObjectsList && renderDetectedObjects()}
          </div>
        )}
        
        {colorAnalysisMode && (
          <div className="color-analysis-banner">
            <div className="color-analysis-icon">
              <i className="fas fa-palette"></i>
            </div>
            <div className="color-analysis-message">
              Color analysis mode active
            </div>
            <button 
              className="cancel-color-button"
              onClick={() => setColorAnalysisMode(false)}
            >
              ✕
            </button>
          </div>
        )}
        
        <div className="question-input-container">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={
              isImageBasedQuery 
                ? "Ask about this frame..." 
                : colorAnalysisMode 
                  ? "Ask about colors in the video..." 
                  : "Ask about what you see in the video..."
            }
            className="question-input"
            disabled={isAIThinking}
            ref={questionInputRef}
          />
          
          {/* Timestamp picker toggle */}
          <button
            className={`timestamp-button ${showTimestampPicker ? 'active' : ''}`}
            onClick={() => setShowTimestampPicker(!showTimestampPicker)}
            disabled={isAIThinking}
            title="Ask about a specific timestamp"
          >
            <i className="fas fa-clock"></i>
          </button>
          
          {/* Frame capture button */}
          <button
            className={`frame-button ${isImageBasedQuery ? 'active' : ''}`}
            onClick={handleCaptureFrame}
            disabled={isAIThinking || !videoRef || !videoRef.current}
            title="Ask about current video frame"
          >
            <i className="fas fa-camera"></i>
          </button>
          
          {/* Color analysis button */}
          <button
            className={`color-button ${colorAnalysisMode ? 'active' : ''}`}
            onClick={toggleColorAnalysisMode}
            disabled={isAIThinking}
            title="Ask about colors in the video"
          >
            <i className="fas fa-palette"></i>
          </button>
          
          <button
            className="ask-button"
            onClick={() => handleVisualQuestion(question)}
            disabled={!question.trim() || isAIThinking}
          >
            Ask
          </button>
        </div>
        
        {/* Timestamp picker dropdown */}
        {showTimestampPicker && (
          <div className="timestamp-picker">
            <h4>Select a timestamp to ask about:</h4>
            {renderTimestampOptions()}
          </div>
        )}
        
        <div className="suggested-questions">
          <div className="suggested-label">Try asking:</div>
          <div className="questions-list">
            {visualQuestions.map((q, index) => (
              <button
                key={index}
                className="question-chip"
                onClick={() => handleVisualQuestion(q)}
                disabled={isAIThinking}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
        
        {/* Recent visual questions and answers */}
        <div className="recent-visual-qa">
          {conversations && Array.isArray(conversations) && conversations
            .filter((msg, index) => {
              // Filter for visual Q&A pairs
              if (msg.role !== 'user') return false;
              
              // Check if the next message is an AI response
              const nextMsg = conversations[index + 1];
              return nextMsg && nextMsg.role === 'ai';
            })
            .slice(-3) // Show only last 3 Q&A pairs
            .map((userMsg, index) => {
              // Find the corresponding AI response
              const aiMsg = conversations.find(
                (msg, i) => i > conversations.indexOf(userMsg) && msg.role === 'ai'
              );
              
              // Find timestamps message if any
              const timestampsMsg = conversations.find(
                (msg, i) => 
                  i > conversations.indexOf(aiMsg) && 
                  i <= conversations.indexOf(aiMsg) + 1 && 
                  msg.role === 'timestamps'
              );
              
              return (
                <div key={index} className="qa-pair">
                  <div className="question">
                    <strong>Q:</strong> {userMsg.content}
                  </div>
                  {aiMsg && (
                    <div className="answer">
                      <strong>A:</strong>{' '}
                      <span
                        dangerouslySetInnerHTML={{
                          __html: marked.parse(aiMsg.content)
                        }}
                      />
                    </div>
                  )}
                  {timestampsMsg && (
                    <div 
                      className="timestamps"
                      dangerouslySetInnerHTML={{
                        __html: marked.parse(timestampsMsg.content)
                      }}
                      onClick={(e) => {
                        // Handle timestamp clicks
                        if (e.target.tagName === 'LI' || e.target.closest('li')) {
                          const text = e.target.textContent || '';
                          const match = text.match(/(\d+:\d+)/);
                          if (match) {
                            const timeStr = match[1];
                            const [minutes, seconds] = timeStr.split(':').map(Number);
                            const timeInSeconds = minutes * 60 + seconds;
                            
                            handleTimestampClick({
                              time: timeInSeconds,
                              time_formatted: timeStr
                            });
                          }
                        }
                      }}
                    />
                  )}
                </div>
              );
            })}
        </div>
      </div>
      <div className="visual-qa-footer">
        <span className="visual-qa-badge">
          <i className="fas fa-eye"></i> Enhanced Visual Understanding
        </span>
      </div>
    </div>
  );
});

export default VisualQAInterface;