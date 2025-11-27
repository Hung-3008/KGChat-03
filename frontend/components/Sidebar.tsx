import React from 'react';
import { RetrievalSteps } from '../types';

interface SidebarProps {
  steps: RetrievalSteps | null;
}

export const Sidebar: React.FC<SidebarProps> = ({ steps }) => {
  if (!steps) {
    return (
      <aside className="w-[30%] hidden lg:flex flex-col bg-white/30 dark:bg-slate-900/30 p-6 border-l border-slate-200 dark:border-slate-800 backdrop-blur-md">
        <div className="flex flex-col items-center justify-center h-full text-center border-2 border-dashed border-slate-300/50 dark:border-slate-700/50 rounded-3xl bg-white/20 dark:bg-slate-800/20">
          <div className="p-8 max-w-xs">
            <div className="w-16 h-16 bg-primary/10 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <span className="material-icons-outlined text-4xl text-primary">analytics</span>
            </div>
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-200 mb-3">
              Retrieval Pipeline
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              Real-time visualization of the AI's reasoning process will appear here.
            </p>
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside className="w-[30%] hidden lg:flex flex-col bg-white/30 dark:bg-slate-900/30 p-6 border-l border-slate-200 dark:border-slate-800 backdrop-blur-md overflow-y-auto custom-scrollbar">
      <div className="flex items-center gap-2 mb-6">
        <span className="material-icons-outlined text-primary">hub</span>
        <h3 className="text-lg font-bold text-slate-800 dark:text-slate-200">Pipeline Steps</h3>
      </div>

      <div className="space-y-4 relative">
        {/* Connecting Line */}
        <div className="absolute left-[19px] top-4 bottom-4 w-0.5 bg-slate-200 dark:bg-slate-700 -z-10"></div>

        {/* Step 1: Question */}
        <StepCard title="Question" content={steps.question} icon="help_outline" active />

        {/* Step 2: Keywords */}
        <StepCard
          title="Keywords"
          content={
            <div className="flex flex-wrap gap-2">
              {steps.keywords.map((k, i) => (
                <span key={i} className="px-2.5 py-1 bg-primary/10 text-primary border border-primary/20 rounded-lg text-xs font-medium">
                  {k}
                </span>
              ))}
            </div>
          }
          icon="manage_search"
          active={steps.keywords.length > 0}
        />

        {/* Step 3: Graph Retrieval */}
        <StepCard
          title="Graph Retrieval"
          content={
            <div className="text-sm text-slate-600 dark:text-slate-400">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-semibold uppercase text-slate-400">Found Nodes</span>
                <span className="px-1.5 py-0.5 bg-slate-100 dark:bg-slate-800 rounded text-[10px] font-bold">{steps.qdrant_nodes.length}</span>
              </div>
              <ul className="space-y-1">
                {steps.qdrant_nodes.slice(0, 5).map((node, i) => (
                  <li key={i} className="flex items-center gap-2 text-xs">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent-coral"></span>
                    <span className="truncate">{node}</span>
                  </li>
                ))}
                {steps.qdrant_nodes.length > 5 && <li className="text-xs text-slate-400 pl-3.5">+{steps.qdrant_nodes.length - 5} more</li>}
              </ul>
            </div>
          }
          icon="hub"
          active={steps.qdrant_nodes.length > 0}
        />

        {/* Step 4: Gather Information (Neo4j) */}
        <StepCard
          title="Knowledge Graph"
          content={
            <div className="text-sm text-slate-600 dark:text-slate-400 max-h-40 overflow-y-auto custom-scrollbar">
              {steps.graph_data.length > 0 ? (
                <ul className="space-y-2">
                  {steps.graph_data.map((edge, i) => (
                    <li key={i} className="bg-white/50 dark:bg-slate-800/50 p-2 rounded-lg border border-slate-100 dark:border-slate-700 text-xs">
                      <span className="font-semibold text-slate-700 dark:text-slate-300">{edge.source}</span>
                      <span className="text-slate-400 mx-1">→</span>
                      <span className="text-slate-500 italic">{edge.relation}</span>
                      <span className="text-slate-400 mx-1">→</span>
                      <span className="font-semibold text-slate-700 dark:text-slate-300">{edge.target}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <span className="text-xs text-slate-400 italic">No relationships found.</span>
              )}
            </div>
          }
          icon="share"
          active={steps.graph_data.length > 0}
        />

        {/* Step 5: Google Grounding */}
        <StepCard
          title="Grounding"
          content={
            <div className="text-sm text-slate-600 dark:text-slate-400 max-h-32 overflow-y-auto custom-scrollbar whitespace-pre-wrap bg-slate-50 dark:bg-slate-800/50 p-3 rounded-lg border border-slate-100 dark:border-slate-700 text-xs leading-relaxed">
              {steps.google_grounding || "No grounding info."}
            </div>
          }
          icon="public"
          active={!!steps.google_grounding}
        />
      </div>
    </aside>
  );
};

const StepCard: React.FC<{ title: string; content: React.ReactNode; icon: string; active?: boolean }> = ({ title, content, icon, active }) => (
  <div className={`relative pl-12 transition-all duration-300 ${active ? 'opacity-100' : 'opacity-50 grayscale'}`}>
    {/* Node Icon */}
    <div className={`absolute left-0 top-0 w-10 h-10 rounded-full flex items-center justify-center border-4 border-background-light dark:border-background-dark z-10 ${active ? 'bg-primary text-white shadow-lg shadow-primary/20' : 'bg-slate-200 dark:bg-slate-700 text-slate-400'}`}>
      <span className="material-icons-outlined text-lg">{icon}</span>
    </div>

    <div className="bg-white dark:bg-slate-800 rounded-2xl p-4 shadow-sm border border-slate-200 dark:border-slate-700">
      <h4 className="font-bold text-slate-800 dark:text-slate-200 text-sm mb-3">{title}</h4>
      <div>{content}</div>
    </div>
  </div>
);
