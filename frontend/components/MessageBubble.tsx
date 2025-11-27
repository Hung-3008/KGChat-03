import React from 'react';
import { Message, Sender } from '../types';

interface MessageBubbleProps {
  message: Message;
  showAvatar?: boolean;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.sender === Sender.USER;
  const isTyping = message.isTyping;

  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-6 group`}>
      <div className={`flex max-w-[85%] md:max-w-[75%] gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>

        {/* Avatar */}
        <div className="flex-shrink-0 mt-1">
          <div className="w-8 h-8 rounded-full overflow-hidden shadow-sm border border-slate-100 dark:border-slate-700">
            <img
              src={isUser ? "/avatars/user.png" : "/avatars/medibot.png"}
              alt={isUser ? "User" : "Medibot"}
              className="w-full h-full object-cover"
            />
          </div>
        </div>

        {/* Message Content */}
        <div className="flex flex-col">
          <span className={`text-xs text-slate-400 mb-1 ${isUser ? 'text-right' : 'text-left'}`}>
            {isUser ? 'You' : 'MediBot AI'}
          </span>
          <div
            className={`relative px-5 py-3.5 rounded-2xl shadow-sm text-sm leading-relaxed ${isUser
                ? 'bg-secondary text-white rounded-tr-none'
                : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 border border-slate-100 dark:border-slate-700 rounded-tl-none'
              } ${isTyping ? 'animate-pulse' : ''}`}
          >
            {isTyping ? (
              <div className="flex space-x-1.5 items-center h-5">
                <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            ) : (
              <div className="whitespace-pre-wrap">{message.text}</div>
            )}
          </div>
          {/* Timestamp (optional, visible on hover or always) */}
          {!isTyping && message.timestamp && (
            <span className={`text-[10px] text-slate-300 dark:text-slate-600 mt-1 opacity-0 group-hover:opacity-100 transition-opacity ${isUser ? 'text-right' : 'text-left'}`}>
              {message.timestamp}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
