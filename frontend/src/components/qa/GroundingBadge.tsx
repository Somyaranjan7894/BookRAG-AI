import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle } from 'lucide-react';
import { Badge, BadgeVariant } from '@/components/common/Badge';

export interface GroundingBadgeProps {
  grounded: boolean;
  score: number;
  status: string;
  className?: string;
}

export const GroundingBadge: React.FC<GroundingBadgeProps> = ({
  grounded,
  score,
  status,
  className = '',
}) => {
  const percentage = Math.round(score * 100);

  let variant: BadgeVariant = 'default';
  let icon = <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />;

  const normStatus = (status || '').toLowerCase();

  if (grounded || normStatus === 'grounded' || normStatus === 'entailed') {
    variant = 'success';
    icon = <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />;
  } else if (normStatus === 'partial' || normStatus === 'partially_grounded') {
    variant = 'warning';
    icon = <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />;
  } else {
    variant = 'error';
    icon = <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />;
  }

  return (
    <div className={`inline-flex items-center gap-2 ${className}`}>
      <Badge variant={variant} className="gap-1.5 py-1 px-3 text-xs font-semibold shadow-xs">
        {icon}
        <span className="capitalize">{status || (grounded ? 'Grounded' : 'Ungrounded')}</span>
        <span className="opacity-75 font-mono">({percentage}%)</span>
      </Badge>
    </div>
  );
};
