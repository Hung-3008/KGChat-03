import React from 'react';
import { RetrievalSteps } from '../types';

interface SidebarProps {
  steps: RetrievalSteps | null;
}

export const Sidebar: React.FC<SidebarProps> = ({ steps }) => {
  if (!steps) {
    return (
      <aside className="w-[35%] hidden lg:flex flex-col bg-white/50 dark:bg-slate-900/50 p-6 border-l border-slate-200 dark:border-slate-700 backdrop-blur-sm">
        <div className="flex flex-col items-center justify-center h-full text-center border-2 border-dashed border-slate-300 dark:border-slate-700 rounded-2xl bg-slate-50/50 dark:bg-slate-800/30">
          <div className="p-8 max-w-xs">
            <div className="w-16 h-16 bg-slate-200 dark:bg-slate-700 rounded-xl flex items-center justify-center mx-auto mb-6">
              <span className="material-icons-outlined text-4xl text-slate-400 dark:text-slate-500">analytics</span>
            </div>
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-200 mb-3">
              Backend Graph Display
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              Relevant data visualizations and patient health trends will appear here dynamically during the consultation.
            </p>
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside className="w-[35%] hidden lg:flex flex-col bg-white/50 dark:bg-slate-900/50 p-6 border-l border-slate-200 dark:border-slate-700 backdrop-blur-sm overflow-y-auto custom-scrollbar">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-200 mb-6">Retrieval Pipeline</h3>

      <div className="space-y-6">
        {/* Step 1: Question */}
        <StepCard title="1. Question" content={steps.question} icon="help_outline" />

        {/* Step 2: Keywords */}
        <StepCard
          title="2. Keywords Extraction"
          content={
            <div className="flex flex-wrap gap-2">
              {steps.keywords.map((k, i) => (
                <span key={i} className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded text-xs">
                  {k}
                </span>
              ))}
            </div>
          }
          icon="manage_search"
        />

        {/* Step 3: Graph Retrieval */}
        <StepCard
          title="3. Graph Retrieval (Qdrant)"
          content={
            <div className="text-sm text-slate-600 dark:text-slate-400">
              Found {steps.qdrant_nodes.length} nodes:
              <ul className="list-disc list-inside mt-1">
                {steps.qdrant_nodes.slice(0, 5).map((node, i) => (
                  <li key={i} className="truncate">{node}</li>
                ))}
                {steps.qdrant_nodes.length > 5 && <li>...and {steps.qdrant_nodes.length - 5} more</li>}
              </ul>
            </div>
          }
          icon="hub"
        />

        {/* Step 4: Gather Information (Neo4j) */}
        <StepCard
          title="4. Gather Information (Neo4j)"
          content={
            <div className="text-sm text-slate-600 dark:text-slate-400 max-h-40 overflow-y-auto custom-scrollbar">
              {steps.graph_data.length > 0 ? (
                <ul className="space-y-2">
                  {steps.graph_data.map((edge, i) => (
                    <li key={i} className="bg-slate-100 dark:bg-slate-800 p-2 rounded text-xs">
                      <span className="font-semibold">{edge.source}</span> --[{edge.relation}]--&gt; <span className="font-semibold">{edge.target}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                "No graph relationships found."
              )}
            </div>
          }
          icon="share"
        />

        {/* Step 5: Google Grounding */}
        <StepCard
          title="5. Google Grounding"
          content={
            <div className="text-sm text-slate-600 dark:text-slate-400 max-h-32 overflow-y-auto custom-scrollbar whitespace-pre-wrap">
              {steps.google_grounding || "No grounding info."}
            </div>
          }
          icon="public"
        />
      </div>
    </aside>
  );
};

const StepCard: React.FC<{ title: string; content: React.ReactNode; icon: string }> = ({ title, content, icon }) => (
  <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-sm border border-slate-200 dark:border-slate-700">
    <div className="flex items-center gap-2 mb-3">
      <span className="material-icons-outlined text-blue-500">{icon}</span>
      <h4 className="font-semibold text-slate-800 dark:text-slate-200">{title}</h4>
    </div>
    <div>{content}</div>
  </div>
);
