// luna-frontend-vite/src/components/VideoPlayer.jsx
import React, { useState, useRef, useEffect } from 'react';
import PropTypes from 'prop-types';
import '../styles/videoPlayer.css';

const VideoPlayer = ({ 
  src, 
  posterUrl, 
  timestamps = [],
  onTimeUpdate,
  autoPlay = false
}) => {
  const videoRef = useRef(null);
  const progressRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [buffered, setBuffered] = useState(0);
  const [showControls, setShowControls] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSettingsPanel, setShowSettingsPanel] = useState(false);
  const [showSpeedOptions, setShowSpeedOptions] = useState(false);
  const [showQualityOptions, setShowQualityOptions] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  // Control visibility timer
  let controlsTimer = null;

  // Format seconds to MM:SS
  const formatTime = (seconds) => {
    if (isNaN(seconds)) return '00:00';
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // Handle video metadata loaded
  const handleMetadataLoaded = () => {
    setDuration(videoRef.current.duration);
    setIsLoading(false);
  };

  // Handle play/pause
  const togglePlay = () => {
    if (videoRef.current.paused) {
      videoRef.current.play();
    } else {
      videoRef.current.pause();
    }
  };

  // Handle fullscreen toggle
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      videoRef.current.parentElement.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  // Handle volume change
  const handleVolumeChange = (e) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    videoRef.current.volume = newVolume;
    setIsMuted(newVolume === 0);
  };

  // Toggle mute
  const toggleMute = () => {
    const newMutedState = !isMuted;
    setIsMuted(newMutedState);
    videoRef.current.muted = newMutedState;
  };

  // Handle progress bar click
  const handleProgressClick = (e) => {
    const progressBar = progressRef.current;
    const rect = progressBar.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    
    // Ensure pos is between 0 and 1
    const seekPos = Math.max(0, Math.min(1, pos));
    
    // Set current time
    videoRef.current.currentTime = seekPos * duration;
  };

  // Set playback speed
  const changePlaybackSpeed = (speed) => {
    setPlaybackSpeed(speed);
    videoRef.current.playbackRate = speed;
    setShowSpeedOptions(false);
  };

  // Reset controls timer
  const resetControlsTimer = () => {
    if (controlsTimer) {
      clearTimeout(controlsTimer);
    }
    
    setShowControls(true);
    
    controlsTimer = setTimeout(() => {
      if (isPlaying) {
        setShowControls(false);
      }
    }, 3000);
  };

  // Update buffered amount
  const updateBuffered = () => {
    if (videoRef.current && videoRef.current.buffered.length > 0) {
      const bufferedEnd = videoRef.current.buffered.end(videoRef.current.buffered.length - 1);
      const bufferedPercent = (bufferedEnd / videoRef.current.duration) * 100;
      setBuffered(bufferedPercent);
    }
  };

  // Jump to specific timestamp
  const jumpToTimestamp = (time) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      if (videoRef.current.paused) {
        videoRef.current.play();
      }
    }
  };

  // Event listeners
  useEffect(() => {
    const video = videoRef.current;
    
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
      if (onTimeUpdate) onTimeUpdate(video.currentTime);
      updateBuffered();
    };
    const handleWaiting = () => setIsLoading(true);
    const handlePlaying = () => setIsLoading(false);
    const handleFullscreenChange = () => setIsFullscreen(!!document.fullscreenElement);
    
    // Add event listeners
    video.addEventListener('loadedmetadata', handleMetadataLoaded);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('waiting', handleWaiting);
    video.addEventListener('playing', handlePlaying);
    video.addEventListener('progress', updateBuffered);
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    
    // Cleanup
    return () => {
      video.removeEventListener('loadedmetadata', handleMetadataLoaded);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('waiting', handleWaiting);
      video.removeEventListener('playing', handlePlaying);
      video.removeEventListener('progress', updateBuffered);
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      
      if (controlsTimer) {
        clearTimeout(controlsTimer);
      }
    };
  }, [onTimeUpdate]);

  // Mouse move event to show controls
  useEffect(() => {
    const handleMouseMove = () => {
      resetControlsTimer();
    };
    
    const playerContainer = videoRef.current?.parentElement;
    if (playerContainer) {
      playerContainer.addEventListener('mousemove', handleMouseMove);
    }
    
    return () => {
      if (playerContainer) {
        playerContainer.removeEventListener('mousemove', handleMouseMove);
      }
    };
  }, [isPlaying]);

  // Close settings panels on click outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (showSettingsPanel && !e.target.closest('.video-settings') && !e.target.closest('.settings-panel')) {
        setShowSettingsPanel(false);
      }
      
      if (showSpeedOptions && !e.target.closest('.speed-selector')) {
        setShowSpeedOptions(false);
      }
      
      if (showQualityOptions && !e.target.closest('.quality-selector')) {
        setShowQualityOptions(false);
      }
    };
    
    document.addEventListener('click', handleClickOutside);
    
    return () => {
      document.removeEventListener('click', handleClickOutside);
    };
  }, [showSettingsPanel, showSpeedOptions, showQualityOptions]);

  return (
    <div className="video-player-container">
      <video
        ref={videoRef}
        className="video-element"
        src={src}
        poster={posterUrl}
        preload="metadata"
        onClick={togglePlay}
        autoPlay={autoPlay}
      />
      
      {/* Big play button overlay */}
      {!isPlaying && !isLoading && (
        <div className="play-button-overlay" onClick={togglePlay}>
          <div className="big-play-button">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
          </div>
        </div>
      )}
      
      {/* Loading overlay */}
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner" />
        </div>
      )}
      
      {/* Settings button */}
      <div className="video-settings" onClick={() => setShowSettingsPanel(!showSettingsPanel)}>
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="3"></circle>
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
        </svg>
      </div>
      
      {/* Settings panel */}
      {showSettingsPanel && (
        <div className="settings-panel active">
          <div className="settings-title">Settings</div>
          
          <div className="settings-option">
            <div className="settings-option-title">Playback Speed</div>
            <div className="settings-option-value" onClick={() => setShowSpeedOptions(!showSpeedOptions)}>
              {playbackSpeed}x
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </div>
            
            {showSpeedOptions && (
              <div className="speed-options">
                {[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2].map((speed) => (
                  <div 
                    key={speed} 
                    className={`speed-option ${playbackSpeed === speed ? 'active' : ''}`}
                    onClick={() => changePlaybackSpeed(speed)}
                  >
                    {speed}x
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Video controls */}
      <div className={`video-controls ${showControls ? 'active' : ''}`}>
        {/* Progress bar with timestamp markers */}
        <div 
          className="progress-container" 
          ref={progressRef} 
          onClick={handleProgressClick}
        >
          <div 
            className="buffer-bar" 
            style={{ width: `${buffered}%` }} 
          />
          <div 
            className="progress-bar" 
            style={{ width: `${(currentTime / duration) * 100}%` }} 
          />
          <div 
            className="progress-handle" 
            style={{ left: `${(currentTime / duration) * 100}%` }} 
          />
          
          {/* Timestamp markers */}
          {timestamps.length > 0 && (
            <div className="timestamp-markers">
              {timestamps.map((timestamp, index) => (
                <div 
                  key={index} 
                  className="timestamp-marker" 
                  style={{ left: `${(timestamp.time / duration) * 100}%` }} 
                  data-tooltip={timestamp.label}
                  onClick={(e) => {
                    e.stopPropagation();
                    jumpToTimestamp(timestamp.time);
                  }}
                />
              ))}
            </div>
          )}
        </div>
        
        {/* Control buttons */}
        <div className="controls-row">
          <div className="left-controls">
            {/* Play/Pause button */}
            <button className="playback-button" onClick={togglePlay}>
              {isPlaying ? (
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8 5v14l11-7z" />
                </svg>
              )}
            </button>
            
            {/* Volume control */}
            <div className="volume-control">
              <button className="volume-button" onClick={toggleMute}>
                {isMuted ? (
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z" />
                  </svg>
                ) : volume > 0.5 ? (
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" />
                  </svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M18.5 12c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM5 9v6h4l5 5V4L9 9H5z" />
                  </svg>
                )}
              </button>
              
              <div className="volume-slider-container">
                <div className="volume-slider">
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={volume}
                    onChange={handleVolumeChange}
                    style={{ 
                      position: 'absolute', 
                      width: '100%', 
                      height: '100%', 
                      opacity: 0,
                      cursor: 'pointer' 
                    }}
                  />
                  <div 
                    className="volume-level" 
                    style={{ width: `${volume * 100}%` }} 
                  />
                </div>
              </div>
            </div>
            
            {/* Time display */}
            <div className="time-display">
              <span className="current-time">{formatTime(currentTime)}</span>
              <span className="time-separator">/</span>
              <span className="total-time">{formatTime(duration)}</span>
            </div>
          </div>
          
          <div className="right-controls">
            {/* Playback speed */}
            <div className={`speed-selector ${showSpeedOptions ? 'active' : ''}`}>
              <button 
                className="speed-button"
                onClick={() => setShowSpeedOptions(!showSpeedOptions)}
              >
                {playbackSpeed}x
              </button>
              
              {showSpeedOptions && (
                <div className="speed-options">
                  {[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2].map((speed) => (
                    <div 
                      key={speed} 
                      className={`speed-option ${playbackSpeed === speed ? 'active' : ''}`}
                      onClick={() => changePlaybackSpeed(speed)}
                    >
                      {speed}x
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            {/* Fullscreen button */}
            <button className="fullscreen-button" onClick={toggleFullscreen}>
              {isFullscreen ? (
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-11V5h-2v5h5V8h-3z" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

VideoPlayer.propTypes = {
  src: PropTypes.string.isRequired,
  posterUrl: PropTypes.string,
  timestamps: PropTypes.arrayOf(
    PropTypes.shape({ 
      time: PropTypes.number.isRequired,
      label: PropTypes.string.isRequired
    })
  ),
  onTimeUpdate: PropTypes.func,
  autoPlay: PropTypes.bool
};

export default VideoPlayer;