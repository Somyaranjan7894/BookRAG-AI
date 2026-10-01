import React from 'react';
import { AlertCircle, ArrowLeft } from 'lucide-react';
import { Link } from '@/router';

export const NotFound: React.FC = () => {
  return (
    <div className="py-20 text-center space-y-4">
      <div className="w-16 h-16 mx-auto rounded-2xl bg-rose-50 text-rose-600 flex items-center justify-center">
        <AlertCircle className="w-8 h-8" />
      </div>
      <h1 className="text-2xl font-bold text-slate-900">404 — Page Not Found</h1>
      <p className="text-sm text-slate-500 max-w-sm mx-auto">
        The page or book you are looking for does not exist or has been removed.
      </p>
      <div className="pt-2">
        <Link
          to="/"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold bg-indigo-600 text-white hover:bg-indigo-700 transition"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Dashboard
        </Link>
      </div>
    </div>
  );
};
