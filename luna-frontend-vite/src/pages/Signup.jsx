import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Signup = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  
  // Use register instead of signup - this is the key fix
  const { register, loading, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  
  // If already authenticated, redirect to analysis page
  useEffect(() => {
    if (isAuthenticated()) {
      navigate('/analysis');
    }
    
    // Check for any success message from previous navigation
    if (location.state?.message) {
      setSuccessMessage(location.state.message);
    }
  }, [isAuthenticated, navigate, location.state]);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMessage('');
    setSuccessMessage('');
    
    // Basic validation
    if (!name || !email || !password || !confirmPassword) {
      setErrorMessage('Please fill in all fields');
      return;
    }
    
    if (password.length < 6) {
      setErrorMessage('Password must be at least 6 characters');
      return;
    }
    
    if (password !== confirmPassword) {
      setErrorMessage('Passwords do not match');
      return;
    }
    
    try {
      // Build user data object that matches what the backend expects
      const userData = {
        name,
        email,
        password
      };
      
      // Call register (not signup) with userData object
      const result = await register(userData);
      
      if (result.success) {
        setSuccessMessage('Account created successfully!');
        
        // Clear form
        setName('');
        setEmail('');
        setPassword('');
        setConfirmPassword('');
        
        if (result.autoLogin) {
          // If auto-login happened, redirect directly to analysis page
          navigate('/analysis');
        } else {
          // Otherwise redirect to login after a short delay
          setTimeout(() => {
            navigate('/login', { 
              state: { message: 'Account created successfully! Please log in.' }
            });
          }, 1500);
        }
      } else {
        // Handle any error in the result
        setErrorMessage(result.error || 'Failed to create account');
      }
    } catch (err) {
      console.error("Signup error:", err);
      
      // Improved error handling that's more flexible with different API formats
      let errorMsg = 'Failed to create account. Please try again.';
      
      if (err.response) {
        // Get error message from various possible locations in the response
        errorMsg = 
          err.response.data?.detail || 
          err.response.data?.message || 
          err.response.data?.error || 
          errorMsg;
        
        // Handle specific status codes
        if (err.response.status === 409 || err.response.status === 400) {
          // Look for common patterns in error messages
          const errText = JSON.stringify(err.response.data).toLowerCase();
          if (errText.includes('email') && (errText.includes('exists') || errText.includes('taken'))) {
            errorMsg = "This email is already registered. Please log in instead.";
          } else if (errText.includes('user') && errText.includes('exists')) {
            errorMsg = "User already exists. Please log in instead.";
          }
        }
      }
      
      setErrorMessage(errorMsg);
    }
  };
  
  return (
    <div className="auth-container">
      <form className="auth-form" onSubmit={handleSubmit}>
        <h1>Sign Up for Luna</h1>
        
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
        
        <div className="form-group">
          <label htmlFor="name">Full Name</label>
          <input
            type="text"
            id="name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Enter your full name"
            required
            disabled={loading || successMessage}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="email">Email</label>
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email"
            required
            disabled={loading || successMessage}
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
            disabled={loading || successMessage}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="confirmPassword">Confirm Password</label>
          <input
            type="password"
            id="confirmPassword"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            placeholder="******"
            required
            disabled={loading || successMessage}
          />
        </div>
        
        <button 
          type="submit" 
          className="auth-button"
          disabled={loading || successMessage}
        >
          {loading ? 'Creating Account...' : 'Sign Up'}
        </button>
        
        <div className="auth-redirect">
          Already have an account? <Link to="/login" style={{ color: '#00CCFF', textDecoration: 'none', fontWeight: 'bold', marginLeft: '5px' }}>Login</Link>
        </div>
        
      </form>
    </div>
  );
};

export default Signup;