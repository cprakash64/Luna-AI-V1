// luna-frontend-vite/src/components/TimestampDisplay.jsx
import React from 'react';
import PropTypes from 'prop-types';

const TimestampDisplay = ({ 
  seconds, 
  showHours = false,
  className = '',
  onClick = null
}) => {
  if (seconds === null || seconds === undefined) {
    return <span className={className}>--:--</span>;
  }

  const totalSeconds = Math.floor(seconds);
  
  let hours = 0;
  let mins = 0;
  let secs = 0;
  
  if (showHours || totalSeconds >= 3600) {
    hours = Math.floor(totalSeconds / 3600);
    mins = Math.floor((totalSeconds % 3600) / 60);
    secs = totalSeconds % 60;
  } else {
    mins = Math.floor(totalSeconds / 60);
    secs = totalSeconds % 60;
  }
  
  const formattedTime = hours > 0
    ? `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    : `${mins}:${secs.toString().padStart(2, '0')}`;
  
  if (onClick) {
    return (
      <button
        className={`inline-flex items-center text-indigo-600 hover:text-indigo-800 ${className}`}
        onClick={() => onClick(seconds)}
      >
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          className="h-3.5 w-3.5 mr-1" 
          viewBox="0 0 20 20" 
          fill="currentColor"
        >
          <path 
            fillRule="evenodd" 
            d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" 
            clipRule="evenodd" 
          />
        </svg>
        {formattedTime}
      </button>
    );
  }
  
  return <span className={className}>{formattedTime}</span>;
};

TimestampDisplay.propTypes = {
  seconds: PropTypes.number,
  showHours: PropTypes.bool,
  className: PropTypes.string,
  onClick: PropTypes.func
};

export default TimestampDisplay;