/**
 * API client for the Feature Manifold Interface backend.
 */

import type {
  GraphData,
  EdgeData,
  EdgeType,
  LatentData,
  CloudInfo,
  PCAResult
} from '../types';

const API_BASE = '/api';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API error: ${response.status} - ${error}`);
  }

  return response.json();
}

// Graph endpoints

export async function getGraph(): Promise<GraphData> {
  return fetchJson<GraphData>(`${API_BASE}/graph`);
}

export async function getEdges(
  type: EdgeType = 'cosine',
  threshold: number = 0
): Promise<EdgeData> {
  const params = new URLSearchParams({
    type,
    threshold: threshold.toString(),
  });
  return fetchJson<EdgeData>(`${API_BASE}/edges?${params}`);
}

export async function getLatent(latentId: number): Promise<LatentData> {
  return fetchJson<LatentData>(`${API_BASE}/latent/${latentId}`);
}

// Cluster endpoints

export async function setCluster(
  latentIds: number[],
  sessionId: string = 'default'
): Promise<CloudInfo> {
  return fetchJson<CloudInfo>(`${API_BASE}/cluster/set`, {
    method: 'POST',
    body: JSON.stringify({ latent_ids: latentIds, session_id: sessionId }),
  });
}

export async function addToCluster(
  latentId: number,
  sessionId: string = 'default'
): Promise<CloudInfo> {
  return fetchJson<CloudInfo>(`${API_BASE}/cluster/add`, {
    method: 'POST',
    body: JSON.stringify({ latent_id: latentId, session_id: sessionId }),
  });
}

export async function removeFromCluster(
  latentId: number,
  sessionId: string = 'default'
): Promise<CloudInfo> {
  return fetchJson<CloudInfo>(`${API_BASE}/cluster/remove`, {
    method: 'POST',
    body: JSON.stringify({ latent_id: latentId, session_id: sessionId }),
  });
}

export async function clearCluster(sessionId: string = 'default'): Promise<CloudInfo> {
  return fetchJson<CloudInfo>(`${API_BASE}/cluster/clear?session_id=${sessionId}`, {
    method: 'POST',
  });
}

export async function getClusterInfo(sessionId: string = 'default'): Promise<CloudInfo> {
  return fetchJson<CloudInfo>(`${API_BASE}/cluster/info?session_id=${sessionId}`);
}

export async function computePCA(
  latentIds: number[],
  nComponents: number = 16,
  maxPoints: number = 100000,
  sessionId: string = 'default'
): Promise<PCAResult> {
  return fetchJson<PCAResult>(`${API_BASE}/cluster/pca`, {
    method: 'POST',
    body: JSON.stringify({
      latent_ids: latentIds,
      n_components: nComponents,
      max_points: maxPoints,
      session_id: sessionId,
    }),
  });
}
