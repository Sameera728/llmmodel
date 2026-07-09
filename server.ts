export interface Model {
  id: string;
  name: string;
  size: string;
  context: string;
  loaded: boolean;
  description?: string;
}

export interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  isStreaming?: boolean;
  modelUsed?: string;
  tokenCount?: number;
  latencyMs?: number;
}

export interface FileNode {
  name: string;
  path: string;
  type: "file" | "directory";
  size?: number;
  children?: FileNode[];
}

export interface IngestedDocument {
  id: string;
  name: string;
  size: number;
  chunks: number;
  timestamp: string;
}

export interface TrainingJob {
  id: string;
  status: "idle" | "running" | "completed";
  epochs: number;
  batchSize: number;
  learningRate: number;
  datasetName: string;
  progress: number;
  lossHistory: number[];
  steps: number;
  totalSteps: number;
  elapsed: number;
}
