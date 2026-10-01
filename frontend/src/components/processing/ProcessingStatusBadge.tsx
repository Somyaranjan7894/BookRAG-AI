import React from 'react';
import { CheckCircle2, Clock, Loader2, AlertCircle } from 'lucide-react';
import { DocumentStatus } from '@/types/document';
import { Badge, BadgeVariant } from '@/components/common/Badge';

export interface ProcessingStatusBadgeProps {
  status: DocumentStatus | string;
  className?: string;
  showIcon?: boolean;
}

export const ProcessingStatusBadge: React.FC<ProcessingStatusBadgeProps> = ({
  status,
  className = '',
  showIcon = true,
}) => {
  const normStatus = (status || '').toLowerCase();

  let variant: BadgeVariant = 'default';
  let label = status;
  let icon = <Clock className="w-3.5 h-3.5 text-slate-500" />;

  switch (normStatus) {
    case 'processed':
    case 'completed':
      variant = 'success';
      label = 'Ready';
      icon = <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />;
      break;
    case 'processing':
    case 'parsing':
    case 'chunking':
    case 'embedding':
    case 'indexing':
      variant = 'info';
      label = 'Processing';
      icon = <Loader2 className="w-3.5 h-3.5 text-blue-600 animate-spin" />;
      break;
    case 'queued':
      variant = 'warning';
      label = 'Queued';
      icon = <Clock className="w-3.5 h-3.5 text-amber-600" />;
      break;
    case 'failed':
      variant = 'error';
      label = 'Failed';
      icon = <AlertCircle className="w-3.5 h-3.5 text-rose-600" />;
      break;
    default:
      variant = 'default';
      label = status;
  }

  return (
    <Badge variant={variant} className={`gap-1.5 font-medium ${className}`}>
      {showIcon && icon}
      <span>{label}</span>
    </Badge>
  );
};
