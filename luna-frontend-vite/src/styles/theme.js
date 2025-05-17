/**
 * Luna AI Theme Configuration
 * 
 * This file contains the theme variables for Luna AI application
 * to maintain consistent styling across components
 */
const theme = {
  // Font families
  fonts: {
    primary: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    secondary: "'Roboto', sans-serif",
    mono: "'Courier New', Courier, monospace"
  },
  // Color palette
  colors: {
    // Primary brand colors
    primary: {
      main: '#5D5FEF', // Updated to a rich indigo for stronger brand identity
      light: '#7879F1',
      dark: '#4A4AEF',
      gradient: 'linear-gradient(135deg, #5D5FEF, #7879F1)'
    },
    
    // Secondary accent colors
    accent: {
      main: '#8C5FEF', // Purple accent for complementary color
      light: '#A77EF2',
      dark: '#7144E8',
      gradient: 'linear-gradient(135deg, #8C5FEF, #A77EF2)'
    },
    
    // Semantic colors
    success: '#34D399', // Updated to a more modern green
    warning: '#FBBF24', 
    error: '#F87171',
    info: '#60A5FA',
    
    // Dark theme (default)
    dark: {
      background: {
        primary: '#121826',
        secondary: '#1E293B',
        tertiary: '#2C3E50',
        gradient: 'linear-gradient(135deg, #0F172A, #1E293B, #334155)'
      },
      surface: {
        primary: 'rgba(30, 41, 59, 0.95)',
        secondary: '#1F2D3D',
        tertiary: '#273746',
        card: 'rgba(30, 41, 59, 0.8)',
        input: '#1E293B'
      },
      text: {
        primary: '#F8FAFC',
        secondary: '#CBD5E1',
        tertiary: '#94A3B8',
        placeholder: '#64748B'
      },
      border: {
        primary: 'rgba(255, 255, 255, 0.1)',
        secondary: '#334155'
      },
      button: {
        primary: '#5D5FEF',
        hover: '#7879F1',
        text: 'white'
      },
      link: {
        primary: '#60A5FA',
        hover: '#93C5FD'
      },
      chat: {
        user: '#334155',
        ai: '#1E293B'
      },
      thinking: {
        dot: 'linear-gradient(45deg, #5D5FEF, #7879F1, #8C5FEF, #A77EF2)'
      }
    },
    
    // Light theme
    light: {
      background: {
        primary: '#F8FAFC',
        secondary: '#FFFFFF',
        tertiary: '#F1F5F9',
        gradient: 'linear-gradient(135deg, #F8FAFC, #FFFFFF, #F1F5F9)'
      },
      surface: {
        primary: '#FFFFFF',
        secondary: '#F8FAFC',
        tertiary: '#F1F5F9',
        card: 'rgba(255, 255, 255, 0.95)',
        input: '#F8FAFC'
      },
      text: {
        primary: '#0F172A',
        secondary: '#334155',
        tertiary: '#64748B',
        placeholder: '#94A3B8'
      },
      border: {
        primary: '#E2E8F0',
        secondary: '#CBD5E1'
      },
      button: {
        primary: '#5D5FEF',
        hover: '#7879F1',
        text: 'white'
      },
      link: {
        primary: '#5D5FEF',
        hover: '#7879F1'
      },
      chat: {
        user: '#F1F5F9',
        ai: '#F8FAFC'
      },
      thinking: {
        dot: 'linear-gradient(45deg, #5D5FEF, #7879F1, #8C5FEF, #A77EF2)'
      }
    }
  },
  
  // Spacing scale
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
    xl: '32px',
    xxl: '48px',
    xxxl: '64px',
    
    // For more precise control
    '1': '4px',
    '2': '8px',
    '3': '12px',
    '4': '16px',
    '5': '20px',
    '6': '24px',
    '8': '32px',
    '10': '40px',
    '12': '48px',
    '16': '64px',
    '20': '80px',
    '24': '96px'
  },
  
  // Border radius
  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '12px',
    xl: '16px',
    xxl: '24px',
    card: '20px',
    circle: '50%',
    pill: '9999px'
  },
  
  // Shadow styles
  shadows: {
    sm: '0 1px 2px rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    button: '0 4px 10px rgba(93, 95, 239, 0.25)',
    buttonHover: '0 6px 15px rgba(93, 95, 239, 0.35)',
    input: '0 1px 2px rgba(0, 0, 0, 0.05)',
    outline: '0 0 0 2px rgba(93, 95, 239, 0.4)'
  },
  
  // Typography
  typography: {
    h1: {
      fontSize: '36px',
      fontWeight: 700,
      lineHeight: 1.2
    },
    h2: {
      fontSize: '30px',
      fontWeight: 700,
      lineHeight: 1.3
    },
    h3: {
      fontSize: '24px',
      fontWeight: 600,
      lineHeight: 1.4
    },
    h4: {
      fontSize: '20px',
      fontWeight: 600,
      lineHeight: 1.4
    },
    body1: {
      fontSize: '16px',
      fontWeight: 400,
      lineHeight: 1.5
    },
    body2: {
      fontSize: '14px',
      fontWeight: 400,
      lineHeight: 1.6
    },
    caption: {
      fontSize: '12px',
      fontWeight: 400,
      lineHeight: 1.5
    },
    button: {
      fontSize: '16px',
      fontWeight: 500,
      lineHeight: 1.5
    }
  },
  
  // Animation/Transition
  animation: {
    fast: '0.15s',
    normal: '0.25s',
    slow: '0.4s',
    easing: 'ease',
    easingInOut: 'cubic-bezier(0.4, 0, 0.2, 1)'
  },
  
  // Z-index values
  zIndex: {
    dropdown: 1000,
    sticky: 1100,
    overlay: 1200,
    modal: 1300,
    popup: 1400,
    tooltip: 1500
  },
  
  // Specific component styling
  components: {
    // Sidebar
    sidebar: {
      width: '25%',
      mobileWidth: '300px',
      padding: '20px'
    },
    
    // Chat components
    chat: {
      messageMaxWidth: '70%',
      inputAreaRadius: '16px',
      inputAreaBackground: '#F8FAFC',
      sendButtonColor: '#5D5FEF',
      sendButtonHover: '#7879F1',
      sendButtonShadow: '0 4px 6px rgba(93, 95, 239, 0.25)'
    },
    
    // Forms
    forms: {
      padding: '40px',
      backdropFilter: 'blur(10px)',
      maxWidth: '450px'
    },
    
    // Code blocks
    code: {
      background: '#1e1e2e',
      headerBackground: '#313244',
      textColor: '#f8f8f2',
      borderRadius: '8px'
    },
    
    // Video Analysis
    videoAnalysis: {
      cardWidth: '550px',
      cardPadding: '32px',
      uploadAreaHeight: '180px',
      inputPadding: '12px 16px',
      buttonHeight: '48px'
    }
  },
  
  // Backdrop blur effect
  backdropBlur: {
    sm: 'blur(4px)',
    md: 'blur(8px)',
    lg: 'blur(16px)'
  },
  
  // Breakpoints for responsive design
  breakpoints: {
    xs: '480px',
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    xxl: '1536px'
  }
};

// Function to get current theme based on system preference
export const getCurrentTheme = () => {
  if (typeof window !== 'undefined') {
    const isDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    return isDarkMode ? 'dark' : 'light';
  }
  return 'dark'; // Default to dark theme on server-side
};

// Helper functions to use theme 
export const getThemeColor = (colorPath, themeMode = getCurrentTheme()) => {
  const paths = colorPath.split('.');
  let result = theme.colors[themeMode];
  
  for (let i = 0; i < paths.length; i++) {
    if (result && result[paths[i]]) {
      result = result[paths[i]];
    } else {
      return null;
    }
  }
  
  return result;
};

export default theme;