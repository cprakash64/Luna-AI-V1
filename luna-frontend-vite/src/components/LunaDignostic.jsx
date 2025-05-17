import React, { useEffect, useState } from 'react';
import { useSocket } from '../hooks/useSocket';

const LunaDiagnostic = () => {
  const [diagnosticInfo, setDiagnosticInfo] = useState({
    socketConnected: false,
    events: [],
    error: null
  });
  
  const socket = useSocket();
  
  useEffect(() => {
    if (!socket) {
      setDiagnosticInfo(prev => ({
        ...prev,
        error: "Socket not initialized"
      }));
      return;
    }
    
    setDiagnosticInfo(prev => ({
      ...prev,
      socketConnected: socket.connected
    }));
    
    // Listen for connection
    const onConnect = () => {
      console.log("Socket connected in diagnostic");
      setDiagnosticInfo(prev => ({
        ...prev,
        socketConnected: true,
        events: [...prev.events, "Connected"]
      }));
    };
    
    // Listen for disconnection
    const onDisconnect = () => {
      console.log("Socket disconnected in diagnostic");
      setDiagnosticInfo(prev => ({
        ...prev,
        socketConnected: false,
        events: [...prev.events, "Disconnected"]
      }));
    };
    
    // Listen for welcome event
    const onWelcome = (data) => {
      console.log("Received welcome event:", data);
      setDiagnosticInfo(prev => ({
        ...prev,
        events: [...prev.events, `Welcome: ${JSON.stringify(data)}`]
      }));
    };
    
    // Register event listeners
    socket.on('connect', onConnect);
    socket.on('disconnect', onDisconnect);
    socket.on('welcome', onWelcome);
    
    // Send test echo
    socket.emit('echo', { message: 'Diagnostic test' });
    
    // Cleanup
    return () => {
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);
      socket.off('welcome', onWelcome);
    };
  }, [socket]);
  
  return (
    <div style={{
      padding: '20px',
      margin: '20px',
      border: '1px solid #ddd',
      borderRadius: '5px',
      backgroundColor: '#f5f5f5'
    }}>
      <h2>Luna AI Diagnostic</h2>
      
      <div style={{ marginBottom: '10px' }}>
        <strong>Socket Status:</strong> {diagnosticInfo.socketConnected ? 
          '✅ Connected' : '❌ Disconnected'}
      </div>
      
      {diagnosticInfo.error && (
        <div style={{ color: 'red', marginBottom: '10px' }}>
          <strong>Error:</strong> {diagnosticInfo.error}
        </div>
      )}
      
      <div>
        <strong>Events:</strong>
        <ul style={{ maxHeight: '200px', overflow: 'auto' }}>
          {diagnosticInfo.events.map((event, index) => (
            <li key={index}>{event}</li>
          ))}
        </ul>
      </div>
      
      <button 
        onClick={() => socket && socket.emit('echo', { message: 'Button test' })}
        style={{
          padding: '10px 15px',
          backgroundColor: '#4CAF50',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          marginTop: '10px'
        }}
      >
        Test Socket Connection
      </button>
    </div>
  );
};

export default LunaDiagnostic;