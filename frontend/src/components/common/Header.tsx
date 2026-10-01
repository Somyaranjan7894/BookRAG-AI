import React from 'react';
import { BookOpen, Sparkles, Library } from 'lucide-react';
import { Link, useRouter } from '@/router';

export const Header: React.FC = () => {
  const { currentPath } = useRouter();

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-200 bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/60">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3 group focus:outline-none focus:ring-2 focus:ring-indigo-500 rounded-lg p-1">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-700 to-purple-600 flex items-center justify-center text-white shadow-md shadow-indigo-100 group-hover:scale-105 transition-transform duration-200">
            <BookOpen className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-lg text-slate-900 tracking-tight">BookRAG AI</span>
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200/50">
                PROD
              </span>
            </div>
            <p className="text-xs text-slate-500 font-medium">Grounded Knowledge Retrieval</p>
          </div>
        </Link>

        <nav className="flex items-center gap-3">
          <Link
            to="/"
            className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-colors ${
              currentPath === '/'
                ? 'bg-slate-100 text-slate-900'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <Library className="w-4 h-4" />
            Book Library
          </Link>
          <div className="h-4 w-px bg-slate-200 mx-1 hidden sm:block" />
          <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-50 border border-slate-200/80 text-xs text-slate-600">
            <Sparkles className="w-3.5 h-3.5 text-amber-500" />
            <span>NLI Grounded QA</span>
          </div>
        </nav>
      </div>
    </header>
  );
};
