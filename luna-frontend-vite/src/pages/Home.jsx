// src/pages/Home.jsx
import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useVideo } from '../context/VideoContext';
import LoadingIndicator from '../components/LoadingIndicator';

const Home = () => {
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);
  
  const { user, logout } = useAuth();
  const { processYoutubeUrl, uploadVideoFile, uploadAudioFile, loading } = useVideo();
  const navigate = useNavigate();
  
  // Apply body styles for the background gradient
  useEffect(() => {
    // Apply styling directly to body
    document.body.style.display = 'flex';
    document.body.style.justifyContent = 'center';
    document.body.style.alignItems = 'center';
    document.body.style.minHeight = '100vh';
    document.body.style.margin = '0';
    document.body.style.padding = '0';
    document.body.style.background = 'linear-gradient(-45deg, #1B263B, #415A77, #778DA9)';
    document.body.style.fontFamily = '"Roboto", sans-serif';
    document.body.style.color = '#E6E6E6';
    document.body.style.overflow = 'hidden';
    
    // Set up the tabId in localStorage
    const tabId = localStorage.getItem("tabId") || Date.now().toString();
    localStorage.setItem("tabId", tabId);
    
    // Hide any header elements
    const headers = document.querySelectorAll('header, nav, .navbar, .nav-bar, .header');
    headers.forEach(header => {
      header.style.display = 'none';
    });
    
    // Cleanup function
    return () => {
      document.body.style.display = '';
      document.body.style.justifyContent = '';
      document.body.style.alignItems = '';
      document.body.style.minHeight = '';
      document.body.style.margin = '';
      document.body.style.padding = '';
      document.body.style.background = '';
      document.body.style.fontFamily = '';
      document.body.style.color = '';
      document.body.style.overflow = '';
      
      // Restore header elements
      headers.forEach(header => {
        header.style.display = '';
      });
    };
  }, []);
  
  // YouTube URL validation
  const isValidYoutubeUrl = (url) => {
    const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/;
    return youtubeRegex.test(url);
  };
  
  // Handle YouTube URL submission
  const handleYoutubeSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (!youtubeUrl.trim() && !fileInputRef.current?.files?.length) {
      setError('Please enter a YouTube URL or select a file');
      return;
    }
    
    if (youtubeUrl.trim() && !isValidYoutubeUrl(youtubeUrl)) {
      setError('Please enter a valid YouTube URL');
      return;
    }
    
    try {
      if (youtubeUrl.trim()) {
        await processYoutubeUrl(youtubeUrl);
      } else if (fileInputRef.current?.files?.length) {
        const file = fileInputRef.current.files[0];
        await handleFileUpload(file);
      }
      navigate('/analysis');
    } catch (err) {
      console.error('Error processing input:', err);
      setError(err.message || 'Failed to process. Please try again.');
    }
  };
  
  // Handle file upload
  const handleFileUpload = async (file) => {
    if (!file) return;
    setError('');
    
    const isVideoOrAudio = file.type.startsWith('video/') || file.type.startsWith('audio/');
    if (!isVideoOrAudio) {
      setError('Please select a valid video or audio file');
      return;
    }
    
    try {
      // Use appropriate upload function based on file type
      if (file.type.startsWith('audio/')) {
        await uploadAudioFile(file);
      } else {
        await uploadVideoFile(file);
      }
      return true;
    } catch (err) {
      console.error('Error uploading file:', err);
      setError(err.message || 'Failed to upload file. Please try again.');
      return false;
    }
  };
  
  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = (e) => {
    e.preventDefault();
    if (e.target === document || e.target.id === 'dragOverlay') {
      setIsDragging(false);
    }
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('video/') || file.type.startsWith('audio/')) {
        handleFileUpload(file);
      } else {
        setError('Please drop a valid video or audio file.');
      }
    }
  };
  
  const containerStyle = {
    backgroundColor: 'rgba(10, 20, 30, 0.9)',
    padding: '60px 40px',
    borderRadius: '40px',
    boxShadow: '0 15px 35px rgba(0, 0, 0, 0.7), 0 0 20px rgba(0, 180, 250, 0.15)',
    textAlign: 'center',
    width: '600px',
    maxWidth: '90%',
    backdropFilter: 'blur(10px)',
    border: '2px solid rgba(255, 255, 255, 0.1)',
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    zIndex: 100
  };
  
  const titleStyle = {
    fontSize: '42px',
    marginBottom: '60px',
    fontWeight: '700',
    color: '#FFFFFF',
    textShadow: '0 0 10px rgba(0, 180, 250, 0.3)'
  };
  
  const formStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    width: '100%'
  };
  
  const inputStyle = {
    width: '100%',
    padding: '20px',
    margin: '10px 0',
    borderRadius: '30px',
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    border: '2px solid #00B4D8',
    color: '#E6E6E6',
    fontSize: '18px',
    textAlign: 'center',
    outline: 'none',
    boxSizing: 'border-box'
  };
  
  const fileInputContainerStyle = {
    width: '100%',
    padding: '15px',
    margin: '10px 0',
    borderRadius: '30px',
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    border: '2px solid #00B4D8',
    color: '#E6E6E6',
    fontSize: '18px',
    textAlign: 'center',
    cursor: 'pointer',
    position: 'relative',
    overflow: 'hidden'
  };
  
  const hiddenFileInputStyle = {
    position: 'absolute',
    top: 0,
    left: 0,
    opacity: 0,
    width: '100%',
    height: '100%',
    cursor: 'pointer'
  };
  
  const orSeparatorStyle = {
    color: '#A8DADC',
    margin: '30px 0',
    fontSize: '18px',
    fontWeight: 'bold'
  };
  
  const buttonStyle = {
    padding: '16px 50px',
    marginTop: '20px',
    borderRadius: '30px',
    background: 'linear-gradient(to right, #00B4D8, #80FFDB)',
    color: 'white',
    fontSize: '20px',
    fontWeight: '600',
    border: 'none',
    cursor: 'pointer',
    boxShadow: '0 10px 20px rgba(0, 180, 216, 0.5), 0 0 15px rgba(0, 180, 250, 0.3)',
    transition: 'all 0.3s ease'
  };
  
  const buttonHoverStyle = {
    background: 'linear-gradient(to right, #80FFDB, #00B4D8)',
    transform: 'translateY(-3px)',
    boxShadow: '0 15px 25px rgba(0, 180, 216, 0.6), 0 0 20px rgba(0, 180, 250, 0.4)'
  };
  
  const [isButtonHovered, setIsButtonHovered] = useState(false);
  
  const dragOverlayStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    display: isDragging ? 'flex' : 'none',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    color: '#FFFFFF',
    fontSize: '24px',
    zIndex: 9999
  };
  
  const errorMessageStyle = {
    marginTop: '20px',
    padding: '15px',
    borderRadius: '10px',
    backgroundColor: 'rgba(231, 76, 60, 0.3)',
    border: '1px solid rgba(231, 76, 60, 0.5)',
    color: '#FFFFFF',
    fontSize: '16px',
    width: '100%',
    boxSizing: 'border-box',
    textAlign: 'center'
  };
  
  const loadingBarStyle = {
    position: 'fixed',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    zIndex: 9999
  };
  
  return (
    <>
      {/* Drag overlay for file drop */}
      <div 
        style={dragOverlayStyle}
        id="dragOverlay"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        Drop your video or audio file here
      </div>
      
      {/* Main content container */}
      <div 
        style={containerStyle}
        id="searchContainer"
        onDragOver={handleDragOver}
      >
        <h1 style={titleStyle}>Luna AI</h1>
        
        <form style={formStyle} onSubmit={handleYoutubeSubmit}>
          <input 
            style={inputStyle}
            type="text" 
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)} 
            placeholder="Enter YouTube URL"
            disabled={loading}
          />
          
          <div style={orSeparatorStyle}>OR</div>
          
          <div style={fileInputContainerStyle}>
            {fileInputRef.current?.files?.[0]?.name || "Choose File"}
            <input 
              style={hiddenFileInputStyle}
              type="file" 
              id="videoFile" 
              ref={fileInputRef}
              onChange={(e) => setFileInputRef(current => ({...current, files: e.target.files}))}
              accept="video/*,audio/*"
              disabled={loading}
            />
          </div>
          
          <button 
            type="submit" 
            disabled={loading} 
            style={{
              ...buttonStyle,
              ...(isButtonHovered && !loading ? buttonHoverStyle : {}),
              opacity: loading ? 0.7 : 1,
              cursor: loading ? 'not-allowed' : 'pointer'
            }}
            onMouseEnter={() => setIsButtonHovered(true)}
            onMouseLeave={() => setIsButtonHovered(false)}
          >
            {loading ? 'Processing...' : 'Get Answer'}
          </button>
          
          {error && (
            <div style={errorMessageStyle}>
              {error}
            </div>
          )}
        </form>
      </div>
      
      {/* Loading indicator */}
      {loading && (
        <div style={loadingBarStyle} className="loading-bar">
          <div className="bar top" style={{
            background: '#3498db',
            height: '5px',
            width: '0%',
            position: 'absolute',
            top: 0,
            animation: 'loadTop 3s linear infinite'
          }}></div>
          <div className="bar right" style={{
            background: '#3498db',
            height: '0%',
            width: '5px',
            position: 'absolute',
            right: 0,
            animation: 'loadRight 3s linear infinite',
            animationDelay: '0.75s'
          }}></div>
          <div className="bar bottom" style={{
            background: '#3498db',
            height: '5px',
            width: '0%',
            position: 'absolute',
            bottom: 0,
            animation: 'loadBottom 3s linear infinite',
            animationDelay: '1.5s'
          }}></div>
          <div className="bar left" style={{
            background: '#3498db',
            height: '0%',
            width: '5px',
            position: 'absolute',
            left: 0,
            animation: 'loadLeft 3s linear infinite',
            animationDelay: '2.25s'
          }}></div>
        </div>
      )}
      
      {/* Define animations for loading bar */}
      <style>
        {`
          @keyframes loadTop {
            from { left: 100%; width: 0; }
            to { left: 0; width: 100%; }
          }
          @keyframes loadRight {
            from { top: 100%; height: 0; }
            to { top: 0; height: 100%; }
          }
          @keyframes loadBottom {
            from { right: 100%; width: 0; }
            to { right: 0; width: 100%; }
          }
          @keyframes loadLeft {
            from { bottom: 100%; height: 0; }
            to { bottom: 0; height: 100%; }
          }
        `}
      </style>
    </>
  );
};

export default Home;