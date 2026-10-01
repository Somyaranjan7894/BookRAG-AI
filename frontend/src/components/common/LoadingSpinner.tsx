import React from 'react';
import { Loader2 } from 'lucide-react';

export interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  className?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  label,
  className = '',
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-10 h-10',
  };

  return (
    <div
      role="status"
      aria-live="polite"
      className={`inline-flex items-center justify-center gap-2.5 ${className}`}
    >
      <Loader2
        className={`animate-spin text-indigo-600 ${sizeClasses[size]}`}
        aria-hidden="true"
      />
      {label ? (
        <span className="text-sm font-medium text-slate-600">{label}</span>
      ) : (
        <span className="sr-only">Loading...</span>
      )}
    </div>
  );
};
