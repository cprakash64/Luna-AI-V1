// src/pages/Login.jsx
import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  
  const { login, loading, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  
  // If already authenticated, redirect to analysis page instead of home
  useEffect(() => {
    if (isAuthenticated()) {
      navigate('/analysis');
    }
  }, [isAuthenticated, navigate]);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!email || !password) {
      setErrorMessage('Please fill in all fields');
      return;
    }
    
    // Attempt login and capture the result.
    const result = await login(email, password);
    
    if (result.success) {
      // Check if there's a redirectUrl in the response and use it
      if (result.redirectUrl) {
        navigate(result.redirectUrl);
      } else {
        // Default to analysis page if no redirectUrl is provided
        navigate('/analysis');
      }
    } else {
      // If login fails, display the error message.
      setErrorMessage(result.error || 'Login failed. Please try again.');
    }
  };
  
  return (
    <div className="auth-container">
      <form className="auth-form" onSubmit={handleSubmit}>
        <h1>Login to Luna</h1>
        
        {errorMessage && (
          <div className="flash-message">
            {errorMessage}
          </div>
        )}
        
        <div className="form-group">
          <label htmlFor="email">Email</label>
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email"
            required
            disabled={loading}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="******"
            required
            disabled={loading}
          />
        </div>
        
        <div className="forgot-password-link">
          <Link to="/forgot-password">Forgot Password?</Link>
        </div>
        
        <button 
          type="submit" 
          className="auth-button"
          disabled={loading}
        >
          {loading ? 'Logging in...' : 'Login'}
        </button>
        
        <div className="auth-redirect">
          Don't have an account? <Link to="/signup">Sign Up</Link>
        </div>
                
      </form>
    </div>
  );
};

export default Login;