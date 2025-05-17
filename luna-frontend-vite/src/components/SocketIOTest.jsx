import React, { useState, useEffect, useRef } from 'react';
import socketService from '../services/socket';

function SocketIOTest() {
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Attempt to connect when component mounts
    const connectSocket = async () => {
      try {
        await socketService.connect();
        setConnected(true);
        addMessage('System', 'Connected to Socket.IO server', 'success');
      } catch (error) {
        addMessage('System', `Connection error: ${error.message}`, 'error');
      }
    };

    connectSocket();

    // Set up event listeners
    const handleConnectionEstablished = (data) => {
      addMessage('Server', `Connection established: ${JSON.stringify(data)}`, 'success');
    };

    const handleEchoResponse = (data) => {
      addMessage('Server', `Echo response: ${JSON.stringify(data)}`, 'info');
    };

    const handleAIResponse = (data) => {
      addMessage('AI', data.response || JSON.stringify(data), 'ai');
    };

    const handleError = (data) => {
      addMessage('Error', data.message || JSON.stringify(data), 'error');
    };

    // Register event listeners
    socketService.on('connection_established', handleConnectionEstablished);
    socketService.on('echo_response', handleEchoResponse);
    socketService.on('ai_response', handleAIResponse);
    socketService.on('error', handleError);

    // Clean up event listeners when component unmounts
    return () => {
      socketService.off('connection_established', handleConnectionEstablished);
      socketService.off('echo_response', handleEchoResponse);
      socketService.off('ai_response', handleAIResponse);
      socketService.off('error', handleError);
    };
  }, []);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const addMessage = (sender, text, type = 'info') => {
    setMessages((prev) => [...prev, { sender, text, type, timestamp: new Date() }]);
  };

  const handleSendEcho = () => {
    if (inputText.trim()) {
      addMessage('You', `Sending echo: ${inputText}`, 'sent');
      socketService
        .emit('echo', { message: inputText })
        .then(() => setInputText(''))
        .catch((error) => addMessage('System', `Error sending message: ${error.message}`, 'error'));
    }
  };

  const handleAskAI = () => {
    if (inputText.trim()) {
      addMessage('You', `Asking AI: ${inputText}`, 'sent');
      socketService
        .askAI('test-video-id', inputText)
        .then(() => setInputText(''))
        .catch((error) => addMessage('System', `Error asking AI: ${error.message}`, 'error'));
    }
  };

  return (
    <div style={{ maxWidth: '600px', margin: '0 auto', padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2>Socket.IO Connection Test</h2>
      
      <div style={{ marginBottom: '10px' }}>
        <span 
          style={{ 
            display: 'inline-block', 
            width: '10px', 
            height: '10px', 
            borderRadius: '50%', 
            backgroundColor: connected ? 'green' : 'red',
            marginRight: '5px'
          }} 
        />
        <span>{connected ? 'Connected' : 'Disconnected'}</span>
      </div>
      
      <div 
        style={{ 
          border: '1px solid #ccc', 
          borderRadius: '4px', 
          height: '300px', 
          overflowY: 'auto',
          padding: '10px',
          marginBottom: '10px',
          backgroundColor: '#f5f5f5'
        }}
      >
        {messages.map((msg, index) => (
          <div 
            key={index} 
            style={{ 
              marginBottom: '8px', 
              padding: '8px', 
              borderRadius: '4px',
              backgroundColor: 
                msg.type === 'error' ? '#ffebee' : 
                msg.type === 'success' ? '#e8f5e9' : 
                msg.type === 'sent' ? '#e3f2fd' :
                msg.type === 'ai' ? '#f3e5f5' : '#fff',
              borderLeft: `4px solid ${
                msg.type === 'error' ? '#f44336' : 
                msg.type === 'success' ? '#4caf50' : 
                msg.type === 'sent' ? '#2196f3' :
                msg.type === 'ai' ? '#9c27b0' : '#9e9e9e'
              }`
            }}
          >
            <div style={{ fontWeight: 'bold' }}>
              {msg.sender} <span style={{ fontSize: '0.8em', color: '#666' }}>
                {msg.timestamp.toLocaleTimeString()}
              </span>
            </div>
            <div>{msg.text}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <div style={{ display: 'flex', marginBottom: '10px' }}>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSendEcho()}
          placeholder="Type a message..."
          style={{ 
            flex: 1, 
            padding: '8px', 
            borderRadius: '4px',
            border: '1px solid #ccc',
            marginRight: '10px'
          }}
        />
      </div>
      
      <div>
        <button 
          onClick={handleSendEcho}
          disabled={!connected}
          style={{ 
            padding: '8px 16px', 
            backgroundColor: '#2196f3', 
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            marginRight: '10px',
            cursor: connected ? 'pointer' : 'not-allowed',
            opacity: connected ? 1 : 0.7
          }}
        >
          Send Echo
        </button>
        
        <button 
          onClick={handleAskAI}
          disabled={!connected}
          style={{ 
            padding: '8px 16px', 
            backgroundColor: '#9c27b0', 
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: connected ? 'pointer' : 'not-allowed',
            opacity: connected ? 1 : 0.7
          }}
        >
          Ask AI
        </button>
      </div>
    </div>
  );
}

export default SocketIOTest;