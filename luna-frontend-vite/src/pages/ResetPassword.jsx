import React, { useState } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const ResetPassword = () => {
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  
  const { token } = useParams();
  const { resetPassword, loading } = useAuth();
  const navigate = useNavigate();
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!newPassword || !confirmPassword) {
      setErrorMessage('Please fill in all fields');
      return;
    }
    
    if (newPassword.length < 6) {
      setErrorMessage('Password must be at least 6 characters');
      return;
    }
    
    if (newPassword !== confirmPassword) {
      setErrorMessage('Passwords do not match');
      return;
    }
    
    try {
      const response = await resetPassword(token, newPassword, confirmPassword);
      setSuccessMessage(response.message || 'Password reset successfully! Please log in with your new password.');
      
      // Redirect to login after 3 seconds
      setTimeout(() => {
        navigate('/login');
      }, 3000);
    } catch (err) {
      setErrorMessage(err.response?.data?.detail || 'Failed to reset password. The link may be invalid or expired.');
    }
  };
  
  return (
    <div className="auth-container">
      <form className="auth-form" onSubmit={handleSubmit}>
        <h1>Reset Password</h1>
        
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
        
        <p>Enter your new password below.</p>
        
        <div className="form-group">
          <label htmlFor="new-password">New Password</label>
          <input
            type="password"
            id="new-password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            placeholder="New Password"
            required
            disabled={loading || successMessage}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="confirm-password">Confirm Password</label>
          <input
            type="password"
            id="confirm-password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            placeholder="Confirm Password"
            required
            disabled={loading || successMessage}
          />
        </div>
        
        <button 
          type="submit" 
          className="auth-button"
          disabled={loading || successMessage}
        >
          {loading ? 'Resetting...' : 'Reset Password'}
        </button>
        
        {successMessage && (
          <div className="auth-redirect">
            Redirecting to login page...
          </div>
        )}
      </form>
    </div>
  );
};

export default ResetPassword;