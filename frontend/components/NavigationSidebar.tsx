import React from 'react';
import { ThemeToggle } from './ThemeToggle';

export const NavigationSidebar: React.FC = () => {
    return (
        <aside className="w-64 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col h-full">
            {/* Header / Logo */}
            <div className="p-6 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-white shadow-lg shadow-primary/30">
                        <span className="material-icons-outlined text-xl">medical_services</span>
                    </div>
                    <div>
                        <h1 className="font-bold text-slate-900 dark:text-white text-lg leading-tight">Medibot</h1>
                        <p className="text-xs text-slate-500 font-medium">Your Health Assistant</p>
                    </div>
                </div>
                <ThemeToggle />
            </div>

            {/* Navigation Items */}
            <nav className="flex-1 px-4 py-4 space-y-1">
                <NavItem icon="chat_bubble_outline" label="Chat" active />
            </nav>

            {/* User Profile (Bottom) */}
            <div className="p-4 border-t border-slate-200 dark:border-slate-800 flex items-center gap-3 mt-auto">
                <img
                    src="/avatars/user.png"
                    alt="User"
                    className="w-8 h-8 rounded-full object-cover border border-slate-200 dark:border-slate-700"
                />
                <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-slate-900 dark:text-white truncate">Hung Nguyen</p>
                    <p className="text-xs text-slate-500 truncate">Patient</p>
                </div>
                <button className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300">
                    <span className="material-icons-outlined text-lg">logout</span>
                </button>
            </div>
        </aside>
    );
};

const NavItem: React.FC<{ icon: string; label: string; active?: boolean }> = ({ icon, label, active }) => (
    <button
        className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${active
            ? 'bg-primary/10 text-primary'
            : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
    >
        <span className={`material-icons-outlined ${active ? 'text-primary' : ''}`}>{icon}</span>
        {label}
    </button>
);
