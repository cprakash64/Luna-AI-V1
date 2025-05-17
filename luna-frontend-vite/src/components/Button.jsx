// luna-frontend-vite/src/components/Button.jsx
import React from 'react';
import PropTypes from 'prop-types';

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'medium', 
  fullWidth = false,
  disabled = false,
  rounded = false,
  startIcon = null,
  endIcon = null,
  onClick,
  type = 'button',
  className = '',
  ...props 
}) => {
  // Base button classes
  const baseClasses = 'font-medium flex items-center justify-center transition-all duration-200';
  
  // Variant classes
  const variantClasses = {
    primary: 'bg-[#6366F1] hover:bg-[#4F46E5] text-white shadow-sm',
    secondary: 'bg-[#F59E0B] hover:bg-[#D97706] text-white shadow-sm',
    outline: 'bg-transparent border border-[#6366F1] text-[#6366F1] hover:bg-[#EEF2FF]',
    text: 'bg-transparent text-[#6366F1] hover:bg-[#EEF2FF]',
    danger: 'bg-[#EF4444] hover:bg-[#DC2626] text-white shadow-sm',
    success: 'bg-[#10B981] hover:bg-[#059669] text-white shadow-sm',
    dark: 'bg-[#111827] hover:bg-[#1F2937] text-white shadow-sm',
    light: 'bg-[#F3F4F6] hover:bg-[#E5E7EB] text-[#374151] shadow-sm',
  };
  
  // Size classes
  const sizeClasses = {
    small: 'text-xs py-1.5 px-3',
    medium: 'text-sm py-2 px-4',
    large: 'text-base py-2.5 px-5',
    xlarge: 'text-lg py-3 px-6',
  };
  
  // Disabled state
  const disabledClasses = disabled 
    ? 'opacity-60 cursor-not-allowed' 
    : 'cursor-pointer';
  
  // Full width
  const widthClasses = fullWidth ? 'w-full' : '';
  
  // Rounded corners
  const roundedClasses = rounded ? 'rounded-full' : 'rounded-md';
  
  // Combine all classes
  const buttonClasses = [
    baseClasses,
    variantClasses[variant],
    sizeClasses[size],
    disabledClasses,
    widthClasses,
    roundedClasses,
    className
  ].join(' ');
  
  // Handle button click
  const handleClick = (e) => {
    if (!disabled && onClick) {
      onClick(e);
    }
  };
  
  return (
    <button
      type={type}
      className={buttonClasses}
      onClick={handleClick}
      disabled={disabled}
      {...props}
    >
      {startIcon && <span className="mr-2">{startIcon}</span>}
      {children}
      {endIcon && <span className="ml-2">{endIcon}</span>}
    </button>
  );
};

Button.propTypes = {
  children: PropTypes.node.isRequired,
  variant: PropTypes.oneOf(['primary', 'secondary', 'outline', 'text', 'danger', 'success', 'dark', 'light']),
  size: PropTypes.oneOf(['small', 'medium', 'large', 'xlarge']),
  fullWidth: PropTypes.bool,
  disabled: PropTypes.bool,
  rounded: PropTypes.bool,
  startIcon: PropTypes.node,
  endIcon: PropTypes.node,
  onClick: PropTypes.func,
  type: PropTypes.oneOf(['button', 'submit', 'reset']),
  className: PropTypes.string,
};

export default Button;