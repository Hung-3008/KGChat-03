export enum Sender {
  USER = 'user',
  BOT = 'bot',
}

export interface Message {
  id: string;
  text: string;
  sender: Sender;
  timestamp: string;
  isTyping?: boolean;
}

export interface ChartDataPoint {
  name: string;
  value: number;
}

export interface GraphNode {
  source: string;
  source_type: string;
  relation: string;
  target: string;
  target_type: string;
}

export interface RetrievalSteps {
  question: string;
  keywords: string[];
  qdrant_nodes: string[];
  graph_data: GraphNode[];
  google_grounding: string;
  final_answer: string;
}
