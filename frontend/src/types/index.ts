/**
 * Type definitions for the Feature Manifold Interface.
 */

export type EdgeType = 'cosine' | 'jaccard' | 'coactivation';

export interface GraphData {
  n_latents: number;
  positions: [number, number][];
  edge_types: string[];
}

export interface Edge {
  source: number;
  target: number;
  weight: number;
}

export interface EdgeData {
  edge_type: string;
  threshold: number;
  n_edges: number;
  edges: Edge[];
}

export interface LatentData {
  latent_id: number;
  n_tokens: number;
  token_indices: number[];
  activations: number[];
}

export interface CloudInfo {
  n_points: number;
  cluster_latents: number[];
  session_id: string;
}

export interface PCAResult {
  n_points: number;
  n_components: number;
  points: number[][];
  explained_variance_ratio: number[];
  subsampled: boolean;
  cluster_latents: number[];
}
