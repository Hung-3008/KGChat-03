import React, { useState, useRef, useEffect } from 'react';

interface ChatInputProps {
  onSendMessage: (text: string) => void;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, disabled }) => {
  const [text, setText] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim() && !disabled) {
      onSendMessage(text);
      setText('');
    }
  };

  // Keep focus on input after send if not disabled
  useEffect(() => {
    if (!disabled && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled]);

  return (
    <div className="mt-4">
      <form onSubmit={handleSubmit} className="relative">
        <input
          ref={inputRef}
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={disabled}
          className="w-full pl-6 pr-14 py-4 bg-white dark:bg-slate-800 border-none shadow-sm rounded-xl focus:ring-2 focus:ring-primary focus:outline-none text-slate-700 dark:text-slate-200 placeholder-slate-400 transition-all disabled:opacity-50"
          placeholder={disabled ? "Please wait..." : "Type your message here..."}
        />
        <button
          type="submit"
          disabled={!text.trim() || disabled}
          className="absolute inset-y-2 right-2 w-12 flex items-center justify-center text-white bg-primary rounded-lg hover:bg-primary-hover disabled:bg-slate-300 dark:disabled:bg-slate-700 transition-colors"
        >
          <span className="material-icons-outlined transform -rotate-45 relative left-[-2px] top-[1px]">send</span>
        </button>
      </form>
    </div>
  );
};
