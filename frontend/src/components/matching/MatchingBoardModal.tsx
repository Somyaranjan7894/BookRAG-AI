import React, { useEffect } from 'react';
import { X } from 'lucide-react';
import { MatchingBoard, MatchingBoardProps } from './MatchingBoard';

export interface MatchingBoardModalProps extends MatchingBoardProps {
  isOpen: boolean;
  onClose: () => void;
}

export const MatchingBoardModal: React.FC<MatchingBoardModalProps> = ({
  isOpen,
  onClose,
  ...boardProps
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.body.style.overflow = 'hidden';
      window.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      document.body.style.overflow = '';
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 sm:p-6 animate-fadeIn"
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-matching-board-title"
    >
      <div
        className="fixed inset-0"
        onClick={onClose}
        aria-hidden="true"
      />

      <div className="relative w-full max-w-5xl max-h-[90vh] overflow-y-auto rounded-3xl bg-slate-50 border border-slate-200 shadow-2xl p-4 sm:p-6 my-auto z-10">
        <button
          type="button"
          onClick={onClose}
          aria-label="Close Matching Board modal"
          className="absolute top-6 right-6 p-2 rounded-xl text-slate-400 hover:text-slate-700 bg-white hover:bg-slate-100 border border-slate-200 transition focus:outline-none focus:ring-2 focus:ring-indigo-500 shadow-2xs z-20"
        >
          <X className="w-5 h-5" aria-hidden="true" />
        </button>

        <MatchingBoard
          {...boardProps}
          onClose={onClose}
        />
      </div>
    </div>
  );
};
