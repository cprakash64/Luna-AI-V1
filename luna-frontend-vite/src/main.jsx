import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './styles/index.css'

// Debug Socket.IO connection
console.log('Initializing Socket.IO debugging...');

// Check if io is available (socket.io-client)
if (typeof io === 'undefined') {
  console.error("ERROR: Socket.IO client library is not loaded!");
} else {
  console.log("Socket.IO client library is loaded correctly");
  
  try {
    // Create direct Socket.IO connection
    const socket = io(window.location.origin, {
      path: '/socket.io/',
      transports: ['websocket', 'polling'],
      reconnectionAttempts: 3,
      reconnectionDelay: 1000,
      timeout: 10000
    });
    
    socket.on('connect', () => {
      console.log('Socket.IO Connected successfully!', socket.id);
    });
    
    socket.on('connect_error', (err) => {
      console.error('Socket.IO Connect Error:', err);
    });
    
    socket.on('error', (err) => {
      console.error('Socket.IO Error:', err);
    });
    
    socket.on('disconnect', (reason) => {
      console.log('Socket.IO Disconnected:', reason);
    });
    
    // Make socket available globally for debugging
    window.socket = socket;
  } catch (e) {
    console.error('Failed to initialize Socket.IO:', e);
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)