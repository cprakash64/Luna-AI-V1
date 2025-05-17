// luna-frontend-vite/src/components/LoadingIndicator.jsx
import React from 'react';
import PropTypes from 'prop-types';

const LoadingIndicator = ({ 
  type = 'spinner', 
  size = 'medium', 
  color = 'primary',
  text = '',
  progress = null,
  className = ''
}) => {
  // Size classes
  const sizeMap = {
    small: {
      spinner: 'w-4 h-4 border-2',
      pulse: 'w-4 h-4',
      dots: 'h-4',
      bar: 'h-1'
    },
    medium: {
      spinner: 'w-8 h-8 border-2',
      pulse: 'w-8 h-8',
      dots: 'h-8',
      bar: 'h-2'
    },
    large: {
      spinner: 'w-12 h-12 border-3',
      pulse: 'w-12 h-12',
      dots: 'h-12',
      bar: 'h-3'
    }
  };

  // Color classes
  const colorMap = {
    primary: {
      main: 'border-[#6366F1]',
      bg: 'bg-[#6366F1]',
      text: 'text-[#6366F1]'
    },
    secondary: {
      main: 'border-[#F59E0B]',
      bg: 'bg-[#F59E0B]',
      text: 'text-[#F59E0B]'
    },
    success: {
      main: 'border-[#10B981]',
      bg: 'bg-[#10B981]',
      text: 'text-[#10B981]'
    },
    error: {
      main: 'border-[#EF4444]',
      bg: 'bg-[#EF4444]',
      text: 'text-[#EF4444]'
    },
    white: {
      main: 'border-white',
      bg: 'bg-white',
      text: 'text-white'
    },
    gray: {
      main: 'border-gray-400',
      bg: 'bg-gray-400',
      text: 'text-gray-400'
    }
  };

  // Spinner Loading Indicator
  const renderSpinner = () => (
    <div 
      className={`
        ${sizeMap[size].spinner} 
        rounded-full 
        border-t-transparent 
        ${colorMap[color].main} 
        animate-spin
      `}
    />
  );

  // Pulsing Loading Indicator
  const renderPulse = () => (
    <div 
      className={`
        ${sizeMap[size].pulse}
        ${colorMap[color].bg}
        rounded-full
        animate-pulse
        opacity-75
      `}
    />
  );

  // Bouncing Dots Loading Indicator
  const renderDots = () => (
    <div className={`flex items-center space-x-2 ${sizeMap[size].dots}`}>
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className={`
            w-2 h-2 
            rounded-full 
            ${colorMap[color].bg}
            animate-bounce
          `}
          style={{
            animationDelay: `${i * 0.1}s`,
            animationDuration: '0.5s',
          }}
        />
      ))}
    </div>
  );

  // Progress Bar Loading Indicator
  const renderProgressBar = () => (
    <div className={`w-full ${sizeMap[size].bar} bg-gray-200 rounded-full overflow-hidden`}>
      <div
        className={`${colorMap[color].bg} h-full transition-all duration-300 ease-in-out`}
        style={{ width: `${progress || 0}%` }}
      />
    </div>
  );

  // Combined Loading Indicator with Text
  const renderWithText = (indicator) => (
    <div className={`flex flex-col items-center space-y-2 ${className}`}>
      {indicator}
      {text && <p className={`text-sm font-medium ${colorMap[color].text}`}>{text}</p>}
    </div>
  );

  // Return the appropriate loading indicator based on type
  switch (type) {
    case 'spinner':
      return renderWithText(renderSpinner());
    case 'pulse':
      return renderWithText(renderPulse());
    case 'dots':
      return renderWithText(renderDots());
    case 'bar':
      return renderWithText(renderProgressBar());
    default:
      return renderWithText(renderSpinner());
  }
};

LoadingIndicator.propTypes = {
  type: PropTypes.oneOf(['spinner', 'pulse', 'dots', 'bar']),
  size: PropTypes.oneOf(['small', 'medium', 'large']),
  color: PropTypes.oneOf(['primary', 'secondary', 'success', 'error', 'white', 'gray']),
  text: PropTypes.string,
  progress: PropTypes.number,
  className: PropTypes.string
};

export default LoadingIndicator;