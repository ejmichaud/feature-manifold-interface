/**
 * Hook for managing the current cluster of latents.
 */

import { useState, useCallback } from 'react';
import * as api from '../api/client';
import type { CloudInfo, PCAResult } from '../types';

interface UseClusterReturn {
  cluster: number[];
  cloudInfo: CloudInfo | null;
  pcaResult: PCAResult | null;
  isLoading: boolean;
  error: string | null;
  addLatent: (latentId: number) => Promise<void>;
  removeLatent: (latentId: number) => Promise<void>;
  toggleLatent: (latentId: number) => Promise<void>;
  clearCluster: () => Promise<void>;
  refreshPCA: () => Promise<void>;
}

export function useCluster(sessionId: string = 'default'): UseClusterReturn {
  const [cluster, setCluster] = useState<number[]>([]);
  const [cloudInfo, setCloudInfo] = useState<CloudInfo | null>(null);
  const [pcaResult, setPcaResult] = useState<PCAResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updatePCA = useCallback(async (latentIds: number[]) => {
    if (latentIds.length === 0) {
      setPcaResult(null);
      return;
    }
    try {
      const result = await api.computePCA(latentIds, 16, 100000, sessionId);
      setPcaResult(result);
    } catch (e) {
      console.error('PCA computation failed:', e);
    }
  }, [sessionId]);

  const addLatent = useCallback(async (latentId: number) => {
    if (cluster.includes(latentId)) return;

    setIsLoading(true);
    setError(null);

    try {
      const info = await api.addToCluster(latentId, sessionId);
      setCloudInfo(info);
      setCluster(info.cluster_latents);
      await updatePCA(info.cluster_latents);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to add latent');
    } finally {
      setIsLoading(false);
    }
  }, [cluster, sessionId, updatePCA]);

  const removeLatent = useCallback(async (latentId: number) => {
    if (!cluster.includes(latentId)) return;

    setIsLoading(true);
    setError(null);

    try {
      const info = await api.removeFromCluster(latentId, sessionId);
      setCloudInfo(info);
      setCluster(info.cluster_latents);
      await updatePCA(info.cluster_latents);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to remove latent');
    } finally {
      setIsLoading(false);
    }
  }, [cluster, sessionId, updatePCA]);

  const toggleLatent = useCallback(async (latentId: number) => {
    if (cluster.includes(latentId)) {
      await removeLatent(latentId);
    } else {
      await addLatent(latentId);
    }
  }, [cluster, addLatent, removeLatent]);

  const clearClusterFn = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const info = await api.clearCluster(sessionId);
      setCloudInfo(info);
      setCluster([]);
      setPcaResult(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to clear cluster');
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  const refreshPCA = useCallback(async () => {
    if (cluster.length === 0) return;
    setIsLoading(true);
    try {
      await updatePCA(cluster);
    } finally {
      setIsLoading(false);
    }
  }, [cluster, updatePCA]);

  return {
    cluster,
    cloudInfo,
    pcaResult,
    isLoading,
    error,
    addLatent,
    removeLatent,
    toggleLatent,
    clearCluster: clearClusterFn,
    refreshPCA,
  };
}
