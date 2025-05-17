// src/App.jsx
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { VideoProvider } from './context/VideoContext';
// Pages
import Home from './pages/Home';
import VideoAnalysis from './pages/VideoAnalysis';
import Login from './pages/Login';
import Signup from './pages/Signup';
import ForgotPassword from './pages/ForgotPassword';
import ResetPassword from './pages/ResetPassword';

// Public routes - accessible to everyone
const PublicRoute = ({ children }) => {
  const { isAuthenticated } = useAuth();
  
  // We don't need location.pathname check anymore since "/" is no longer public
  return children;
};

// Protected routes - only for authenticated users
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();
  const location = useLocation();
  
  if (loading) {
    return <div className="loading">Loading...</div>;
  }
  
  if (!isAuthenticated()) {
    return <Navigate to={`/login?from=${encodeURIComponent(location.pathname)}`} replace />;
  }
  
  return children;
};

// App component needs to be refactored since we're using useAuth hook inside ProtectedRoute
const AppRoutes = () => {
  const { isAuthenticated } = useAuth();
  
  return (
    <Routes>
      {/* Root route now shows the VideoAnalysis page and is protected */}
      <Route path="/" element={
        <ProtectedRoute>
          <VideoAnalysis />
        </ProtectedRoute>
      } />
      
      {/* Keep analysis route for backward compatibility - also protected */}
      <Route path="/analysis" element={
        <ProtectedRoute>
          <VideoAnalysis />
        </ProtectedRoute>
      } />
      
      {/* Move the old home page to /home if you still need it */}
      <Route path="/home" element={<PublicRoute><Home /></PublicRoute>} />
      
      {/* Auth routes - for login and registration */}
      <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
      <Route path="/signup" element={<PublicRoute><Signup /></PublicRoute>} />
      <Route path="/forgot-password" element={<PublicRoute><ForgotPassword /></PublicRoute>} />
      <Route path="/reset-password" element={<PublicRoute><ResetPassword /></PublicRoute>} />
      
      {/* Fallback route - redirect to root */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
};

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <VideoProvider>
          <AppRoutes />
        </VideoProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;