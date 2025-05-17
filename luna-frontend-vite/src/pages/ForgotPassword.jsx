import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const ForgotPassword = () => {
  const [email, setEmail] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  
  const { forgotPassword, loading } = useAuth();
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!email) {
      setErrorMessage('Please enter your email address');
      return;
    }
    
    try {
      const response = await forgotPassword(email);
      setSuccessMessage(response.message || 'If your email exists in our system, you will receive a password reset link.');
      setEmail('');
    } catch (err) {
      setErrorMessage(err.response?.data?.detail || 'Failed to process request. Please try again.');
    }
  };
  
  return (
    <div className="auth-container">
      <form className="auth-form" onSubmit={handleSubmit}>
        <h1>Forgot Password</h1>
        
        {errorMessage && (
          <div className="flash-message error">
            {errorMessage}
          </div>
        )}
        
        {successMessage && (
          <div className="flash-message success">
            {successMessage}
          </div>
        )}
        
        <p>Enter the email address associated with your account.</p>
        
        <div className="form-group">
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email Address"
            required
            disabled={loading || successMessage}
          />
        </div>
        
        <button 
          type="submit" 
          className="auth-button"
          disabled={loading || successMessage}
        >
          {loading ? 'Sending...' : 'Send Reset Link'}
        </button>
        
        <div className="auth-redirect">
          <Link to="/login">Back to Login</Link>
        </div>
      </form>
    </div>
  );
};

export default ForgotPassword;