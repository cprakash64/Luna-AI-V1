// src/components/ProtectedRoute.jsx
import React, { useEffect } from 'react';
import { Navigate } from 'react-router-dom';

const ProtectedRoute = ({ children }) => {
  // Direct token check - no complex state or hooks
  const token = localStorage.getItem('authToken');
  
  useEffect(() => {
    console.log("AUTH DEBUG:", { 
      hasToken: !!token,
      tokenValue: token ? token.substring(0, 15) + "..." : "null",
      path: window.location.pathname
    });
  }, [token]);
  
  // No token = immediate redirect with no other logic
  if (!token) {
    console.log("🔒 NO TOKEN FOUND - REDIRECTING TO LOGIN");
    return <Navigate to="/login" replace />;
  }
  
  console.log("✅ TOKEN FOUND - RENDERING PROTECTED ROUTE");
  return children;
};

export default ProtectedRoute;