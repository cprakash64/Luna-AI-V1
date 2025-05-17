// src/hooks/useAuth.js
import { useAuth as useAuthFromContext } from '../context/AuthContext';

/**
 * Custom hook for authentication
 * This is a wrapper around the useAuth hook from AuthContext
 * to maintain proper separation of concerns
 * 
 * @returns {Object} Authentication state and methods
 */
export const useAuth = () => {
  return useAuthFromContext();
};

// Default export for flexibility in imports
export default useAuth;