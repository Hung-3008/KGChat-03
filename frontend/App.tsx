import React, { useState, useRef, useEffect } from 'react';
import { MessageBubble } from './components/MessageBubble';
import { ChatInput } from './components/ChatInput';
import { Sidebar } from './components/Sidebar';
import { Message, Sender, RetrievalSteps } from './types';
import { sendMessageToBackend } from './services/api';

const INITIAL_MESSAGES: Message[] = [
  {
    id: '1',
    text: "Hello, I'm your AI Medical Assistant. How are you feeling today? Please describe your symptoms.",
    sender: Sender.BOT,
    timestamp: '10:30 AM',
  },
];

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [retrievalSteps, setRetrievalSteps] = useState<RetrievalSteps | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (text: string) => {
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text,
      sender: Sender.USER,
      timestamp,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsProcessing(true);
    setRetrievalSteps(null); // Reset steps for new query

    // Add typing placeholder
    const typingId = 'typing-' + Date.now();
    setMessages((prev) => [
      ...prev,
      {
        id: typingId,
        text: '',
        sender: Sender.BOT,
        timestamp: '',
        isTyping: true,
      },
    ]);

    try {
      let finalAnswer = "";

      // Call backend service with streaming callback
      await sendMessageToBackend(text, (steps) => {
        setRetrievalSteps(steps);
        if (steps.final_answer) {
          finalAnswer = steps.final_answer;
        }
      });

      const responseTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      // Remove typing placeholder and add real response
      setMessages((prev) => {
        const filtered = prev.filter((msg) => msg.id !== typingId);
        return [
          ...filtered,
          {
            id: Date.now().toString(),
            text: finalAnswer || "I processed your request but couldn't generate a final answer.",
            sender: Sender.BOT,
            timestamp: responseTimestamp,
          },
        ];
      });
    } catch (error) {
      console.error("Failed to get response", error);
      // Remove typing placeholder and add error
      setMessages((prev) => {
        const filtered = prev.filter((msg) => msg.id !== typingId);
        return [
          ...filtered,
          {
            id: Date.now().toString(),
            text: "I apologize, but I'm having trouble connecting to the server right now. Please try again.",
            sender: Sender.BOT,
            timestamp: timestamp
          }
        ]
      })
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background-light dark:bg-background-dark">
      <main className="flex-1 flex flex-col h-full relative">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-5 border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md z-10">
          <div>
            <h1 className="text-xl font-bold text-slate-900 dark:text-white font-display">
              Patient Consultation
            </h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Interacting with your AI Medical Assistant
            </p>
          </div>
          <button className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            <span className="material-icons-outlined text-slate-600 dark:text-slate-400">more_vert</span>
          </button>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 custom-scrollbar">
          <div className="max-w-4xl mx-auto flex flex-col justify-end min-h-full">
            {messages.map((msg, index) => {
              // Check if previous message was from same sender to group visually (optional, but good for UI)
              // For now, adhering strictly to design where every bot message has an avatar
              return <MessageBubble key={msg.id} message={msg} />;
            })}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="px-4 md:px-8 pb-6 md:pb-8 pt-2 bg-gradient-to-t from-background-light via-background-light to-transparent dark:from-background-dark dark:via-background-dark">
          <div className="max-w-4xl mx-auto">
            <ChatInput onSendMessage={handleSendMessage} disabled={isProcessing} />
          </div>
        </div>
      </main>

      {/* Sidebar */}
      <Sidebar steps={retrievalSteps} />
    </div>
  );
};

export default App;
