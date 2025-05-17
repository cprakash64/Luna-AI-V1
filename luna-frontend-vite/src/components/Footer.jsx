// src/components/Footer.jsx
import React from 'react';
import { useTheme } from '../context/ThemeContext';

const Footer = () => {
  const { isDarkMode } = useTheme();
  
  return (
    <footer className={`py-6 border-t ${isDarkMode ? 'border-gray-800 bg-gray-900' : 'border-gray-200 bg-white'}`}>
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <div className="flex items-center">
              <span className="text-lg font-bold mr-2">Luna AI</span>
              <span className={`text-xs px-2 py-1 rounded-full ${isDarkMode ? 'bg-indigo-900 text-indigo-200' : 'bg-indigo-100 text-indigo-800'}`}>
                Video Analysis
              </span>
            </div>
            <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Your AI-powered video analysis tool
            </p>
          </div>
          
          <div className="flex space-x-8">
            <div>
              <h4 className="font-medium mb-2">Features</h4>
              <ul className={`text-sm space-y-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                <li>Transcription</li>
                <li>Content Search</li>
                <li>Visual Analysis</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium mb-2">Support</h4>
              <ul className={`text-sm space-y-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                <li>Documentation</li>
                <li>Feedback</li>
                <li>Privacy Policy</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className={`mt-8 pt-4 border-t text-center text-sm ${isDarkMode ? 'border-gray-800 text-gray-500' : 'border-gray-100 text-gray-400'}`}>
          &copy; {new Date().getFullYear()} Luna AI. All rights reserved.
        </div>
      </div>
    </footer>
  );
};

export default Footer;