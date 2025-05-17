import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSocket } from '../hooks/useSocket';
import { useVideoContext } from '../context/VideoContext';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import './ChatInterface.css';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showWelcomeMessage, setShowWelcomeMessage] = useState(true);
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);
  const chatAreaRef = useRef(null);
  const inputRef = useRef(null);
  const socket = useSocket();
  const { transcription, tabId } = useVideoContext();
  const navigate = useNavigate();

  // Welcome message typing animation
  const welcomeMessages = [
    "How can I help you with?",
    "I can provide answers to your questions.",
    "I can analyze the content of the video.",
    "Need help finding specific timestamps?"
  ];
  const [currentWelcomeIndex, setCurrentWelcomeIndex] = useState(0);
  const [displayedWelcome, setDisplayedWelcome] = useState('');
  const [isTypingWelcome, setIsTypingWelcome] = useState(true);
  const [isErasingWelcome, setIsErasingWelcome] = useState(false);

  // Typing animation for welcome message
  useEffect(() => {
    if (!showWelcomeMessage) return;
    const currentMessage = welcomeMessages[currentWelcomeIndex];
    if (isTypingWelcome) {
      if (displayedWelcome.length < currentMessage.length) {
        const timeoutId = setTimeout(() => {
          setDisplayedWelcome(currentMessage.slice(0, displayedWelcome.length + 1));
        }, 100); // Typing speed
        return () => clearTimeout(timeoutId);
      } else {
        setIsTypingWelcome(false);
        const timeoutId = setTimeout(() => {
          setIsErasingWelcome(true);
        }, 2000); // Wait before erasing
        return () => clearTimeout(timeoutId);
      }
    }
    if (isErasingWelcome) {
      if (displayedWelcome.length > 0) {
        const timeoutId = setTimeout(() => {
          setDisplayedWelcome(displayedWelcome.slice(0, -1));
        }, 50); // Erasing speed
        return () => clearTimeout(timeoutId);
      } else {
        setIsErasingWelcome(false);
        setIsTypingWelcome(true);
        setCurrentWelcomeIndex((currentWelcomeIndex + 1) % welcomeMessages.length);
      }
    }
  }, [showWelcomeMessage, displayedWelcome, isTypingWelcome, isErasingWelcome, currentWelcomeIndex, welcomeMessages]);

  // Socket connection and message handling
  useEffect(() => {
    if (!socket) return;
    // Handle incoming AI responses
    socket.on('ai_response', (data) => {
      setIsTyping(false);
      
      // Add AI message to chat
      setMessages(prev => [...prev, { role: 'ai', content: data.answer }]);
      
      // Scroll to bottom
      scrollToBottom();
    });
    return () => {
      socket.off('ai_response');
    };
  }, [socket]);

  // Auto-scroll to bottom when messages change
  const scrollToBottom = () => {
    setTimeout(() => {
      if (chatAreaRef.current) {
        chatAreaRef.current.scrollTo({
          top: chatAreaRef.current.scrollHeight,
          behavior: 'smooth'
        });
      }
    }, 100);
  };

  // Send message function
  const sendMessage = () => {
    if (!input.trim() || isTyping) return;
    
    // Hide welcome message
    if (showWelcomeMessage) {
      setShowWelcomeMessage(false);
    }
    
    // Add user message to chat
    setMessages(prev => [...prev, { role: 'user', content: input }]);
    
    // Show typing indicator
    setIsTyping(true);
    
    // Send message to server
    if (socket && tabId) {
      socket.emit('ask_ai', { question: input, tabId });
    }
    
    // Clear input field and reset height
    setInput('');
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
    }
    
    scrollToBottom();
  };

  // Handle Enter key to send message
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    } else if (e.key === 'Enter' && e.shiftKey) {
      // Allow Shift + Enter to create a new line
      autoGrow(e.target);
    }
  };

  // Auto-grow textarea
  const autoGrow = (element) => {
    element.style.height = 'auto';
    element.style.height = (element.scrollHeight) + 'px';
    
    // Limit height to 9 lines
    const maxHeight = 1.5 * 16 * 9;
    if (element.scrollHeight > maxHeight) {
      element.style.height = maxHeight + 'px';
      element.style.overflowY = 'auto';
    } else {
      element.style.overflowY = 'hidden';
    }
  };

  // Handle input change
  const handleInputChange = (e) => {
    setInput(e.target.value);
    autoGrow(e.target);
  };

  // Toggle sidebar visibility
  const toggleSidebar = () => {
    setIsSidebarVisible(!isSidebarVisible);
  };

  // Copy text to clipboard
  const copyToClipboard = (text, element) => {
    navigator.clipboard.writeText(text)
      .then(() => {
        // Visual feedback for copy success
        const originalInnerHTML = element.innerHTML;
        element.innerHTML = "Copied!";
        element.classList.add('copied');
        
        setTimeout(() => {
          element.innerHTML = originalInnerHTML;
          element.classList.remove('copied');
        }, 1500);
      })
      .catch(err => console.error('Could not copy text: ', err));
  };

  // Render message content with properly sanitized HTML from Markdown
  const renderMessageContent = (message) => {
    if (message.role === 'user') {
      // For user messages, just replace newlines with <br> tags
      return <div dangerouslySetInnerHTML={{ 
        __html: DOMPurify.sanitize(message.content.replace(/\n/g, '<br>')) 
      }} />;
    } else {
      // For AI messages, parse markdown and apply syntax highlighting
      const rawHtml = marked.parse(message.content);
      const sanitizedHtml = DOMPurify.sanitize(rawHtml, {
        ADD_ATTR: ['target', 'rel'], // Allow target="_blank" for links
      });
      
      return (
        <div className="ai-message-content">
          <div dangerouslySetInnerHTML={{ __html: sanitizedHtml }} />
          <div className="icon-actions">
            <i 
              className="fas fa-clipboard" 
              title="Copy to clipboard"
              onClick={(e) => copyToClipboard(message.content, e.target)}
            ></i>
          </div>
        </div>
      );
    }
  };

  // Process code blocks to add language labels and copy buttons
  useEffect(() => {
    // Add language labels and copy buttons to code blocks after rendering
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(block => {
      // Skip already processed blocks
      if (block.parentElement.classList.contains('code-block-container')) return;
      
      const container = block.parentElement;
      container.classList.add('code-block-container');
      // Find language class
      const languageClass = Array.from(block.classList).find(cls => cls.startsWith('language-'));
      const language = languageClass ? languageClass.replace('language-', '').toUpperCase() : 'CODE';
      // Create header with language label and copy button
      const header = document.createElement('div');
      header.classList.add('code-block-header');
      
      const languageLabel = document.createElement('span');
      languageLabel.classList.add('code-language-label');
      languageLabel.textContent = language;
      header.appendChild(languageLabel);
      
      const copyIcon = document.createElement('i');
      copyIcon.classList.add('fas', 'fa-clipboard', 'copy-icon');
      header.appendChild(copyIcon);
      
      // Insert header before code block
      container.insertBefore(header, block);
      
      // Add click handler for copying code
      copyIcon.addEventListener('click', () => {
        const codeText = block.textContent;
        navigator.clipboard.writeText(codeText)
          .then(() => {
            copyIcon.classList.replace('fa-clipboard', 'fa-check');
            setTimeout(() => copyIcon.classList.replace('fa-check', 'fa-clipboard'), 2000);
          })
          .catch(err => console.error('Failed to copy: ', err));
      });
    });
  }, [messages]); // Re-run when messages change

  return (
    <div className="chat-container">
      {/* The sidebar with conditional class for visibility */}
      <div className={`sidebar ${!isSidebarVisible ? 'sidebar-hidden' : ''}`}>
        <div className="sidebar-header">
          <div className="logo-container">
            <div className="luna-logo">🌙</div>
            <div className="logo-text">Luna AI</div>
          </div>
          <button className="theme-toggle">
            <svg width="24" height="24" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" 
                d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z">
              </path>
            </svg>
          </button>
        </div>

        <div className="tab-navigation">
          <button className="tab-button active">
            <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" 
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z">
              </path>
            </svg>
            <span>Transcript</span>
          </button>
          <button className="tab-button">
            <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" 
                d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z">
              </path>
            </svg>
            <span>Time</span>
          </button>
        </div>

        <div className="transcribed-text-container">
          <div className="transcribed-text-box">
            <div className="text-content">{transcription || "No transcription available."}</div>
          </div>
        </div>
        
        <div className="user-info">
          <div className="user-avatar">U</div>
          <div className="user-greeting">hi</div>
        </div>
      </div>
      
      {/* Main content area with conditional class for centering */}
      <div className={`main-content ${!isSidebarVisible ? 'main-content-full' : ''}`}>
        {/* Sidebar toggle button - positioned consistently */}
        <button id="toggleButton" className={`sidebar-toggle ${!isSidebarVisible ? 'toggle-right' : ''}`} onClick={toggleSidebar}>
          <i className={`fas fa-chevron-${isSidebarVisible ? 'left' : 'right'}`}></i>
        </button>
        
        {/* Welcome message at the beginning */}
        {showWelcomeMessage && (
          <div className="welcome-message">
            {displayedWelcome}
            <span className="cursor"></span>
          </div>
        )}
        
        {/* Chat area with messages - UPDATED to match screenshot with no avatars */}
        <div 
          className={`chat-area ${!isSidebarVisible ? 'chat-area-centered' : ''}`} 
          ref={chatAreaRef}
        >
          {messages.map((message, index) => (
            <div key={index} className={`message-bubble ${message.role}`}>
              {renderMessageContent(message)}
            </div>
          ))}
          
          {/* Typing indicator */}
          {isTyping && (
            <div className="message-bubble ai thinking">
              <div className="thinking-animation">
                <div className="thinking-dot"></div>
                <div className="thinking-dot"></div>
                <div className="thinking-dot"></div>
              </div>
            </div>
          )}
        </div>
        
        {/* Input area */}
        <div className="input-container">
          <textarea
            id="userInput"
            ref={inputRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the video or a specific timestamp..."
            rows="1"
          ></textarea>
          <div className="input-actions">
            <button onClick={sendMessage} className="send-button">
              <svg width="24" height="24" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
              </svg>
            </button>
          </div>
        </div>
        
        {/* Bottom controls */}
        <div className="bottom-controls">
          <button className="control-button">
            <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
            </svg>
            <span>Show Video</span>
          </button>
          <button className="control-button">
            <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
            </svg>
            <span>New Video</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;