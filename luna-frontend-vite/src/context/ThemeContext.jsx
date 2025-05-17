// src/context/ThemeContext.jsx
import React, { createContext, useState, useEffect, useContext } from 'react';

// Create context
export const ThemeContext = createContext();

// Custom hook for using theme
export const useTheme = () => useContext(ThemeContext);

export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState('system');
  const [isDarkMode, setIsDarkMode] = useState(false);
  
  // Initialize theme based on system preference
  useEffect(() => {
    // Check for saved preference
    const savedTheme = localStorage.getItem('luna-theme') || 'system';
    setTheme(savedTheme);
    
    // Set initial theme based on system or saved preference
    if (savedTheme === 'system') {
      const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setIsDarkMode(systemPrefersDark);
      document.documentElement.classList.toggle('dark', systemPrefersDark);
    } else {
      const isDark = savedTheme === 'dark';
      setIsDarkMode(isDark);
      document.documentElement.classList.toggle('dark', isDark);
    }
  }, []);
  
  // Handle system theme preference changes
  useEffect(() => {
    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      
      const handleChange = (e) => {
        setIsDarkMode(e.matches);
        document.documentElement.classList.toggle('dark', e.matches);
      };
      
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }
  }, [theme]);
  
  // Toggle between light and dark mode
  const toggleTheme = () => {
    const newTheme = isDarkMode ? 'light' : 'dark';
    setIsDarkMode(!isDarkMode);
    setTheme(newTheme);
    localStorage.setItem('luna-theme', newTheme);
    document.documentElement.classList.toggle('dark', !isDarkMode);
  };
  
  // Set a specific theme (light, dark, or system)
  const setSpecificTheme = (newTheme) => {
    setTheme(newTheme);
    localStorage.setItem('luna-theme', newTheme);
    
    if (newTheme === 'system') {
      const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setIsDarkMode(systemPrefersDark);
      document.documentElement.classList.toggle('dark', systemPrefersDark);
    } else {
      const isDark = newTheme === 'dark';
      setIsDarkMode(isDark);
      document.documentElement.classList.toggle('dark', isDark);
    }
  };
  
  return (
    <ThemeContext.Provider 
      value={{ 
        theme, 
        isDarkMode, 
        setTheme: setSpecificTheme, 
        toggleTheme 
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
};

export default ThemeProvider;