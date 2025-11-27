import React, { useState, useRef, useEffect } from 'react';
import { MessageBubble } from './components/MessageBubble';
import { ChatInput } from './components/ChatInput';
import { Sidebar } from './components/Sidebar';
import { NavigationSidebar } from './components/NavigationSidebar';
import { Message, Sender, RetrievalSteps } from './types';
import { sendMessageToBackend } from './services/api';

const INITIAL_MESSAGES: Message[] = [
  {
    id: '1',
    text: "Hello! I'm Medibot, your personal health assistant. How can I help you today? You can describe your symptoms, ask about a condition, or track your vitals.",
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
    <div className="flex h-screen overflow-hidden bg-background-light dark:bg-background-dark font-body">
      {/* Left Navigation Sidebar */}
      <NavigationSidebar />

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col h-full relative bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm">
        {/* Header - Simplified */}
        <header className="flex items-center justify-between px-8 py-4 border-b border-slate-100 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md z-10">
          <div>
            <h2 className="text-lg font-bold text-slate-800 dark:text-white">Conversation</h2>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            <span className="text-xs font-medium text-slate-500">Online</span>
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 custom-scrollbar">
          <div className="max-w-3xl mx-auto flex flex-col justify-end min-h-full space-y-6">
            {messages.map((msg, index) => {
              return <MessageBubble key={msg.id} message={msg} />;
            })}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="px-4 md:px-8 pb-6 pt-2">
          <div className="max-w-3xl mx-auto">
            <ChatInput onSendMessage={handleSendMessage} disabled={isProcessing} />
          </div>
        </div>
      </main>

      {/* Right Retrieval Sidebar */}
      <Sidebar steps={retrievalSteps} />
    </div>
  );
};

export default App;
