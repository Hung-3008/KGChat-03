import React, { useState, KeyboardEvent } from 'react';

interface ChatInputProps {
  onSendMessage: (text: string) => void;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, disabled }) => {
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim() && !disabled) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="relative">
      <div className="bg-slate-100 dark:bg-slate-800 rounded-full flex items-center p-2 pl-6 shadow-inner border border-transparent focus-within:border-secondary/30 focus-within:bg-white dark:focus-within:bg-slate-900 transition-all duration-200">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe your symptoms..."
          disabled={disabled}
          className="flex-1 bg-transparent border-none outline-none text-slate-800 dark:text-slate-200 placeholder-slate-400 text-sm py-2"
        />
        <div className="flex items-center gap-1 pr-1">

          <button
            onClick={handleSend}
            disabled={!input.trim() || disabled}
            className={`p-3 rounded-full flex items-center justify-center transition-all duration-200 ${input.trim() && !disabled
              ? 'bg-secondary hover:bg-secondary-light text-white shadow-md transform hover:scale-105'
              : 'bg-slate-200 dark:bg-slate-700 text-slate-400 cursor-not-allowed'
              }`}
          >
            <span className="material-icons-outlined text-lg">arrow_upward</span>
          </button>
        </div>
      </div>
    </div>
  );
};
