import React from 'react';
import { Message, Sender } from '../types';

interface MessageBubbleProps {
  message: Message;
  showAvatar?: boolean;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message, showAvatar = true }) => {
  const isBot = message.sender === Sender.BOT;

  return (
    <div className={`flex gap-3 ${!isBot ? 'justify-end' : ''} mb-4`}>
      {/* Bot Avatar */}
      {isBot && (
        <div className={`w-10 h-10 rounded-full bg-primary flex-shrink-0 flex items-center justify-center ${!showAvatar ? 'invisible' : ''}`}>
          <span className="material-icons-outlined text-white text-xl">smart_toy</span>
        </div>
      )}

      {/* Message Content */}
      <div className={`max-w-[85%] md:max-w-[70%]`}>
        <div
          className={`p-4 shadow-sm ${
            isBot
              ? 'bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 rounded-r-xl rounded-bl-xl'
              : 'bg-primary text-white rounded-l-xl rounded-br-xl'
          }`}
        >
          {message.isTyping ? (
            <div className="flex space-x-1 h-5 items-center">
              <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
              <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
              <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
            </div>
          ) : (
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.text}</p>
          )}
        </div>
        
        {!message.isTyping && (
          <span className={`text-xs text-slate-400 dark:text-slate-500 mt-1 block ${!isBot ? 'text-right' : 'text-left'}`}>
            {message.timestamp}
          </span>
        )}
      </div>
    </div>
  );
};
