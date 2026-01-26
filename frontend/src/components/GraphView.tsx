/**
 * Graph visualization component using sigma.js.
 *
 * Displays SAE latents as nodes with edges based on similarity.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import Graph from 'graphology';
import Sigma from 'sigma';
import * as api from '../api/client';
import type { EdgeType, GraphData, EdgeData } from '../types';

interface GraphViewProps {
  cluster: number[];
  onNodeClick: (latentId: number) => void;
  edgeType: EdgeType;
  edgeThreshold: number;
}

export function GraphView({
  cluster,
  onNodeClick,
  edgeType,
  edgeThreshold,
}: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const graphRef = useRef<Graph | null>(null);

  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [edgeData, setEdgeData] = useState<EdgeData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load graph data
  useEffect(() => {
    async function loadGraph() {
      try {
        setIsLoading(true);
        const data = await api.getGraph();
        setGraphData(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load graph');
      } finally {
        setIsLoading(false);
      }
    }
    loadGraph();
  }, []);

  // Load edges when type or threshold changes
  useEffect(() => {
    async function loadEdges() {
      if (!graphData) return;
      try {
        const data = await api.getEdges(edgeType, edgeThreshold);
        setEdgeData(data);
      } catch (e) {
        console.error('Failed to load edges:', e);
      }
    }
    loadEdges();
  }, [graphData, edgeType, edgeThreshold]);

  // Initialize sigma when data is ready
  useEffect(() => {
    if (!containerRef.current || !graphData || !edgeData) return;

    // Create graph
    const graph = new Graph();

    // Add nodes
    graphData.positions.forEach((pos, i) => {
      graph.addNode(i, {
        x: pos[0] * 1000,  // Scale for visibility
        y: pos[1] * 1000,
        size: 3,
        label: `Latent ${i}`,
        color: cluster.includes(i) ? '#ff6b6b' : '#4a90d9',
      });
    });

    // Add edges
    edgeData.edges.forEach((edge, i) => {
      graph.addEdge(edge.source, edge.target, {
        weight: edge.weight,
        size: Math.max(0.5, edge.weight * 2),
        color: '#cccccc',
      });
    });

    graphRef.current = graph;

    // Initialize Sigma
    const sigma = new Sigma(graph, containerRef.current, {
      renderLabels: false,
      labelRenderedSizeThreshold: 20,
      minCameraRatio: 0.1,
      maxCameraRatio: 10,
    });

    // Handle node clicks
    sigma.on('clickNode', ({ node }) => {
      const latentId = parseInt(node, 10);
      onNodeClick(latentId);
    });

    sigmaRef.current = sigma;

    return () => {
      sigma.kill();
      sigmaRef.current = null;
      graphRef.current = null;
    };
  }, [graphData, edgeData]);

  // Update node colors when cluster changes
  useEffect(() => {
    const graph = graphRef.current;
    if (!graph) return;

    graph.forEachNode((node) => {
      const latentId = parseInt(node, 10);
      graph.setNodeAttribute(
        node,
        'color',
        cluster.includes(latentId) ? '#ff6b6b' : '#4a90d9'
      );
    });

    sigmaRef.current?.refresh();
  }, [cluster]);

  if (isLoading) {
    return (
      <div className="graph-view loading">
        <div className="loading-spinner">Loading graph...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="graph-view error">
        <div className="error-message">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="graph-view">
      <div className="graph-info">
        {graphData && (
          <span>
            {graphData.n_latents.toLocaleString()} nodes, {edgeData?.n_edges.toLocaleString() ?? 0} edges
          </span>
        )}
      </div>
      <div ref={containerRef} className="graph-container" />
    </div>
  );
}
