import React, { useContext, useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { 
  Moon, 
  Sun, 
  Menu, 
  X, 
  Video, 
  Search, 
  Home, 
  LogOut, 
  User, 
  FileText,
  ClipboardList
} from 'lucide-react';

const Navbar = ({ darkMode, setDarkMode }) => {
  const location = useLocation();
  const { user, logout } = useAuth();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  const toggleTheme = () => {
    setDarkMode(!darkMode);
  };
  
  const closeMenu = () => {
    setMobileMenuOpen(false);
  };
  
  const isActive = (path) => {
    return location.pathname === path;
  };
  
  // Function to get user's initial from name or email
  const getUserInitial = () => {
    if (!user) return 'U';
    
    if (user.name) {
      return user.name.charAt(0).toUpperCase();
    } else if (user.email) {
      return user.email.charAt(0).toUpperCase();
    } else {
      return 'U';
    }
  };
  
  // Function to get user's display name
  const getUserName = () => {
    if (!user) return 'User';
    
    if (user.name) {
      return user.name.split(' ')[0]; // Get first name
    } else if (user.email) {
      return user.email.split('@')[0]; // Get username part of email
    } else {
      return 'User';
    }
  };
  
  const navbarClasses = `fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
    scrolled 
      ? `${darkMode ? 'bg-gray-900/95 backdrop-blur-md' : 'bg-white/95 backdrop-blur-md shadow-sm'}`
      : `${darkMode ? 'bg-gray-900' : 'bg-white'}`
  }`;
  
  // Updated mainMenuItems - changed Home to point to "/analysis" and removed the redundant Video Analysis item
  const mainMenuItems = [
    {
      name: 'Home',
      path: '/analysis', // Changed from '/' to '/analysis'
      icon: <Home size={18} />
    },
    {
      name: 'My Library',
      path: '/library',
      icon: <FileText size={18} />
    },
    {
      name: 'Search',
      path: '/search',
      icon: <Search size={18} />
    }
  ];
  
  return (
    <nav className={navbarClasses}>
      <div className="container mx-auto px-4 md:px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo - updated to point to /analysis */}
          <Link to="/analysis" className="flex items-center space-x-2">
            <div className={`h-8 w-8 rounded-full bg-gradient-to-br ${darkMode ? 'from-indigo-500 to-purple-700' : 'from-indigo-400 to-purple-600'} flex items-center justify-center`}>
              <span className="text-white font-bold text-lg">L</span>
            </div>
            <span className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Luna<span className="text-indigo-500">AI</span>
            </span>
          </Link>
          
          {/* Desktop Navigation */}
          <div className="hidden md:flex md:items-center md:space-x-6">
            <div className="flex items-center space-x-4">
              {mainMenuItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive(item.path)
                      ? darkMode
                        ? 'bg-indigo-600/20 text-indigo-400'
                        : 'bg-indigo-50 text-indigo-700'
                      : darkMode
                      ? 'text-gray-300 hover:bg-gray-800 hover:text-white'
                      : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
                  }`}
                >
                  {item.icon}
                  <span>{item.name}</span>
                </Link>
              ))}
            </div>
            <div className="flex items-center space-x-3">
              <button
                onClick={toggleTheme}
                className={`p-2 rounded-full transition-colors ${
                  darkMode 
                    ? 'bg-gray-800 hover:bg-gray-700 text-yellow-300' 
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-800'
                }`}
                aria-label="Toggle theme"
              >
                {darkMode ? <Sun size={18} /> : <Moon size={18} />}
              </button>
              
              {user ? (
                <div className="relative group">
                  <button
                    className={`flex items-center space-x-1 p-1 rounded-full ${
                      darkMode ? 'hover:bg-gray-800' : 'hover:bg-gray-100'
                    }`}
                  >
                    <div className={`h-8 w-8 rounded-full flex items-center justify-center ${
                      darkMode ? 'bg-indigo-600 text-white' : 'bg-indigo-500 text-white'
                    }`}>
                      {/* Display user initial instead of User icon */}
                      <span className="font-semibold text-sm">{getUserInitial()}</span>
                    </div>
                  </button>
                  <div className={`absolute right-0 mt-2 w-48 origin-top-right rounded-md shadow-lg overflow-hidden transition-opacity opacity-0 invisible group-hover:opacity-100 group-hover:visible ${
                    darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'
                  }`}>
                    {/* Add user greeting at the top of dropdown */}
                    <div className={`px-4 py-2 border-b ${
                      darkMode ? 'border-gray-700 text-gray-200' : 'border-gray-200 text-gray-800'
                    }`}>
                      <p className="text-sm font-medium">Hello, {getUserName()}</p>
                    </div>
                    
                    <div className="py-1">
                      <Link
                        to="/profile"
                        className={`block px-4 py-2 text-sm ${
                          darkMode 
                            ? 'text-gray-300 hover:bg-gray-700' 
                            : 'text-gray-700 hover:bg-gray-100'
                        }`}
                      >
                        <span className="flex items-center gap-2">
                          <User size={16} />
                          Profile
                        </span>
                      </Link>
                      <button
                        onClick={logout}
                        className={`w-full text-left block px-4 py-2 text-sm ${
                          darkMode 
                            ? 'text-red-400 hover:bg-gray-700' 
                            : 'text-red-600 hover:bg-gray-100'
                        }`}
                      >
                        <span className="flex items-center gap-2">
                          <LogOut size={16} />
                          Sign out
                        </span>
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <Link
                  to="/login"
                  className={`px-4 py-1.5 rounded-full text-sm font-medium ${
                    darkMode
                      ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
                      : 'bg-indigo-50 hover:bg-indigo-100 text-indigo-700'
                  }`}
                >
                  Sign in
                </Link>
              )}
            </div>
          </div>
          
          {/* Mobile menu button */}
          <div className="flex md:hidden items-center space-x-3">
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-full ${
                darkMode 
                  ? 'bg-gray-800 hover:bg-gray-700 text-yellow-300' 
                  : 'bg-gray-100 hover:bg-gray-200 text-gray-800'
              }`}
              aria-label="Toggle theme"
            >
              {darkMode ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            
            {/* Add user initial for mobile */}
            {user && (
              <div className={`h-8 w-8 rounded-full flex items-center justify-center ${
                darkMode ? 'bg-indigo-600 text-white' : 'bg-indigo-500 text-white'
              }`}>
                <span className="font-semibold text-sm">{getUserInitial()}</span>
              </div>
            )}
            
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className={`p-2 rounded-md transition-colors ${
                darkMode 
                  ? 'text-gray-300 hover:bg-gray-800 hover:text-white' 
                  : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
              }`}
              aria-expanded="false"
            >
              <span className="sr-only">Open main menu</span>
              {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>
      
      {/* Mobile menu */}
      <div
        className={`md:hidden transition-all duration-300 overflow-hidden ${
          mobileMenuOpen ? 'max-h-96' : 'max-h-0'
        } ${darkMode ? 'bg-gray-900 border-t border-gray-800' : 'bg-white border-t border-gray-100'}`}
      >
        {/* Add user greeting at the top of mobile menu when logged in */}
        {user && (
          <div className={`px-4 py-3 ${
            darkMode ? 'border-b border-gray-800 text-gray-200' : 'border-b border-gray-100 text-gray-800'
          }`}>
            <p className="text-sm font-medium">Hello, {getUserName()}</p>
          </div>
        )}
        
        <div className="container mx-auto px-4 py-2 space-y-1">
          {mainMenuItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              onClick={closeMenu}
              className={`flex items-center space-x-3 px-3 py-2 rounded-md text-base font-medium w-full ${
                isActive(item.path)
                  ? darkMode
                    ? 'bg-indigo-600/20 text-indigo-400'
                    : 'bg-indigo-50 text-indigo-700'
                  : darkMode
                  ? 'text-gray-300 hover:bg-gray-800 hover:text-white'
                  : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
              }`}
            >
              {item.icon}
              <span>{item.name}</span>
            </Link>
          ))}
          
          {user ? (
            <>
              <Link
                to="/profile"
                onClick={closeMenu}
                className={`flex items-center space-x-3 px-3 py-2 rounded-md text-base font-medium w-full ${
                  darkMode
                    ? 'text-gray-300 hover:bg-gray-800 hover:text-white'
                    : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
                }`}
              >
                <User size={18} />
                <span>Profile</span>
              </Link>
              <button
                onClick={() => {
                  logout();
                  closeMenu();
                }}
                className={`flex items-center space-x-3 px-3 py-2 rounded-md text-base font-medium w-full text-left ${
                  darkMode
                    ? 'text-red-400 hover:bg-gray-800'
                    : 'text-red-600 hover:bg-gray-100'
                }`}
              >
                <LogOut size={18} />
                <span>Sign out</span>
              </button>
            </>
          ) : (
            <Link
              to="/login"
              onClick={closeMenu}
              className={`flex items-center justify-center px-3 py-2 rounded-md text-base font-medium ${
                darkMode
                  ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
                  : 'bg-indigo-100 hover:bg-indigo-200 text-indigo-700'
              }`}
            >
              Sign in
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;