// src/context/AuthContext.jsx
import React, { createContext, useState, useEffect, useCallback } from 'react';
import api from '../services/api';

// Create auth context
export const AuthContext = createContext();

export function AuthProvider({ children }) {
  // State for user authentication
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Default redirect path - centralized for consistency
  const DEFAULT_REDIRECT = '/analysis';
  
  // Load user from local storage on initial mount
  useEffect(() => {
    const loadUserFromStorage = () => {
      try {
        // Try to get user from localStorage
        const storedUser = localStorage.getItem('user');
        const token = localStorage.getItem('authToken');
        
        if (storedUser) {
          // If we have a stored user, parse and set it
          const userData = JSON.parse(storedUser);
          setUser(userData);
        } else if (token) {
          // If we only have a token, create a minimal user object
          setUser({ token });
        }
      } catch (err) {
        console.error('Error loading user from storage:', err);
      } finally {
        // Always set loading to false when done
        setLoading(false);
      }
    };
    
    loadUserFromStorage();
  }, []);
  
  // Login function
  const login = useCallback(async (email, password) => {
    setLoading(true);
    setError(null);
    
    try {
      // Use optional chaining to safely access response data
      const response = await api.post('/api/auth/login', { email, password });
      const userData = response?.data;
      
      if (!userData) {
        throw new Error('Invalid response from server');
      }
      
      // Save token separately for easy access
      if (userData.token || userData.access_token) {
        localStorage.setItem('authToken', userData.token || userData.access_token);
      }
      
      // Save full user data
      localStorage.setItem('user', JSON.stringify(userData));
      setUser(userData);
      
      // Always redirect to analysis page as default, unless specified by server
      const redirectUrl = userData.redirectUrl || DEFAULT_REDIRECT;
      
      return { 
        success: true, 
        user: userData,
        redirectUrl
      };
    } catch (err) {
      // Safely handle error by checking various properties that might exist
      const errorMessage = 
        err?.response?.data?.message || 
        err?.response?.data?.error || 
        err?.message || 
        'An error occurred during login';
      
      setError(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Register function
  const register = useCallback(async (userData) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.post('/api/auth/register', userData);
      
      // Check if response exists and has data
      if (!response || !response.data) {
        throw new Error('Invalid response from server');
      }
      
      const responseData = response.data;
      
      // Some APIs automatically log in after registration, handle both cases
      if (responseData.token || responseData.access_token) {
        // Save token separately
        localStorage.setItem('authToken', responseData.token || responseData.access_token);
        
        // Save user data
        localStorage.setItem('user', JSON.stringify(responseData));
        setUser(responseData);
        
        // Always redirect to analysis page as default, unless specified by server
        const redirectUrl = responseData.redirectUrl || DEFAULT_REDIRECT;
        
        return { 
          success: true, 
          user: responseData, 
          autoLogin: true,
          redirectUrl
        };
      }
      
      // Registration successful but no auto-login
      return { 
        success: true, 
        autoLogin: false,
        redirectUrl: DEFAULT_REDIRECT
      };
    } catch (err) {
      // Safely extract error message
      const errorMessage = 
        err?.response?.data?.message || 
        err?.response?.data?.error || 
        err?.message || 
        'An error occurred during registration';
      
      setError(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Logout function
  const logout = useCallback(() => {
    try {
      // Clear local storage
      localStorage.removeItem('user');
      localStorage.removeItem('authToken');
      
      // Reset state
      setUser(null);
      
      // Optional: Call logout endpoint if your API requires it
      api.post('/api/auth/logout').catch(err => {
        console.warn('Error during logout API call:', err);
        // Continue with client-side logout regardless of API success
      });
      
      return { 
        success: true,
        redirectUrl: '/login' // Redirect to login after logout
      };
    } catch (err) {
      console.error('Error during logout:', err);
      return { 
        success: false, 
        error: err.message,
        redirectUrl: '/login' // Still redirect to login on error
      };
    }
  }, []);
  
  // Check if user is authenticated
  const isAuthenticated = useCallback(() => {
    return !!user;
  }, [user]);
  
  // Get auth token
  const getToken = useCallback(() => {
    if (user?.token) return user.token;
    if (user?.access_token) return user.access_token;
    
    // Try to get from localStorage as fallback
    const token = localStorage.getItem('authToken');
    if (token) return token;
    
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        const userData = JSON.parse(storedUser);
        return userData.token || userData.access_token || null;
      } catch (err) {
        return null;
      }
    }
    
    return null;
  }, [user]);
  
  // Reset password request
  const requestPasswordReset = useCallback(async (email) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.post('/api/auth/reset-password', { email });
      
      // Check if response exists before accessing properties
      if (response?.data) {
        return { 
          success: true, 
          message: response.data.message || 'Password reset email sent',
          redirectUrl: '/login'
        };
      }
      
      return { 
        success: true, 
        message: 'Password reset request submitted',
        redirectUrl: '/login'
      };
    } catch (err) {
      const errorMessage = 
        err?.response?.data?.message || 
        err?.response?.data?.error || 
        err?.message || 
        'An error occurred during password reset request';
      
      setError(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Confirm password reset
  const confirmPasswordReset = useCallback(async (token, newPassword) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.post('/api/auth/reset-password/confirm', {
        token,
        password: newPassword
      });
      
      // Check if response exists
      if (response?.data) {
        return { 
          success: true, 
          message: response.data.message || 'Password has been reset successfully',
          redirectUrl: '/login'
        };
      }
      
      return { 
        success: true, 
        message: 'Password has been reset successfully',
        redirectUrl: '/login'
      };
    } catch (err) {
      const errorMessage = 
        err?.response?.data?.message || 
        err?.response?.data?.error || 
        err?.message || 
        'An error occurred during password reset';
      
      setError(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Refresh token
  const refreshToken = useCallback(async () => {
    try {
      const response = await api.post('/api/auth/refresh');
      
      // Check if response exists and has token
      if (response?.data?.token || response?.data?.access_token) {
        const token = response.data.token || response.data.access_token;
        
        // Update localStorage
        localStorage.setItem('authToken', token);
        
        // Update user object if it exists
        if (user) {
          const updatedUser = { 
            ...user, 
            token: token,
            access_token: token
          };
          localStorage.setItem('user', JSON.stringify(updatedUser));
          setUser(updatedUser);
        }
        
        return { success: true, token };
      }
      
      throw new Error('No token received from refresh endpoint');
    } catch (err) {
      console.error('Token refresh failed:', err);
      
      // If refresh fails, log the user out
      if (err?.response?.status === 401) {
        logout();
      }
      
      return { 
        success: false, 
        error: err.message,
        redirectUrl: '/login' // Redirect to login if refresh fails
      };
    }
  }, [user, logout]);
  
  // Context value
  const contextValue = {
    user,
    loading,
    error,
    login,
    logout,
    register,
    isAuthenticated,
    getToken,
    requestPasswordReset,
    confirmPasswordReset,
    refreshToken,
    DEFAULT_REDIRECT // Export the default redirect path for use in components
  };
  
  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
}

// Custom hook for using auth context
export function useAuth() {
  const context = React.useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export default AuthContext;