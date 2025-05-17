import React from 'react';
import LunaDiagnostic from './components/LunaDiagnostic';

const TestApp = () => {
  return (
    <div style={{ 
      padding: '20px',
      maxWidth: '800px',
      margin: '0 auto'
    }}>
      <h1>Luna AI Test Page</h1>
      <p>If you can see this text, basic React rendering is working.</p>
      
      <LunaDiagnostic />
      
      <div style={{ marginTop: '20px' }}>
        <h2>Troubleshooting Steps</h2>
        <ol>
          <li>Check if this component renders (if you see this, it does)</li>
          <li>Check if Socket.IO is connecting (see diagnostic panel above)</li>
          <li>Open React DevTools to inspect component hierarchy</li>
          <li>Look for any errors in browser console beyond the ones shown</li>
        </ol>
      </div>
    </div>
  );
};

export default TestApp;