import React from 'react';
import { Router, useRouter } from './router';
import { Header } from './components/common/Header';
import { Dashboard } from './pages/Dashboard';
import { BookDetail } from './pages/BookDetail';
import { NotFound } from './pages/NotFound';

const AppContent: React.FC = () => {
  const { currentPath } = useRouter();

  let pageContent: React.ReactNode;

  if (currentPath === '/' || currentPath === '') {
    pageContent = <Dashboard />;
  } else if (currentPath.startsWith('/books/')) {
    pageContent = <BookDetail />;
  } else {
    pageContent = <NotFound />;
  }

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 font-sans antialiased selection:bg-indigo-500 selection:text-white">
      <Header />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10">
        {pageContent}
      </main>

      <footer className="border-t border-slate-200 bg-white py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p>© 2026 BookRAG AI • Grounded RAG with NLI Claim Verification & pgvector</p>
          <div className="flex items-center gap-4 text-slate-400">
            <span>FastAPI Backend</span>
            <span>•</span>
            <span>PostgreSQL</span>
            <span>•</span>
            <span>Celery Workers</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <Router>
      <AppContent />
    </Router>
  );
};

export default App;
