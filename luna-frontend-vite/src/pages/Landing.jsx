// src/pages/Landing.jsx
import React from 'react';
import { Link } from 'react-router-dom';

function Landing() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100">
      <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-8 mb-4">
        <h1 className="text-3xl font-bold text-center mb-6">Welcome to Luna AI</h1>
        <p className="text-gray-600 mb-8 text-center">
          Your AI-powered video analysis tool
        </p>
        
        <div className="flex flex-col space-y-4">
          <Link 
            to="/login" 
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg text-center"
          >
            Login
          </Link>
          
          <Link 
            to="/signup" 
            className="w-full bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-3 px-4 rounded-lg text-center"
          >
            Create Account
          </Link>
        </div>
      </div>
      
      <p className="text-sm text-gray-500">
        © {new Date().getFullYear()} Luna AI. All rights reserved.
      </p>
    </div>
  );
}

export default Landing;