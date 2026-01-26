/**
 * Main application component for Feature Manifold Interface.
 *
 * Layout:
 * - Left: Graph visualization of latents
 * - Right: Point cloud PCA visualizations
 * - Bottom: Cluster panel showing selected latents
 */

import { useState } from 'react';
import { GraphView } from './components/GraphView';
import { PointCloudView } from './components/PointCloudView';
import { ClusterPanel } from './components/ClusterPanel';
import { useCluster } from './hooks/useCluster';
import type { EdgeType } from './types';
import './App.css';

function App() {
  const {
    cluster,
    pcaResult,
    isLoading,
    error,
    toggleLatent,
    removeLatent,
    clearCluster,
  } = useCluster();

  const [edgeType, setEdgeType] = useState<EdgeType>('cosine');
  const [edgeThreshold, setEdgeThreshold] = useState(0.3);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Feature Manifold Interface</h1>
        <div className="header-controls">
          <label>
            Edge Type:
            <select
              value={edgeType}
              onChange={(e) => setEdgeType(e.target.value as EdgeType)}
            >
              <option value="cosine">Cosine Similarity</option>
              <option value="jaccard">Jaccard Similarity</option>
              <option value="coactivation">Co-activation</option>
            </select>
          </label>
          <label>
            Threshold:
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={edgeThreshold}
              onChange={(e) => setEdgeThreshold(parseFloat(e.target.value))}
            />
            <span>{edgeThreshold.toFixed(2)}</span>
          </label>
        </div>
      </header>

      <main className="app-main">
        <div className="left-panel">
          <GraphView
            cluster={cluster}
            onNodeClick={toggleLatent}
            edgeType={edgeType}
            edgeThreshold={edgeThreshold}
          />
        </div>

        <div className="right-panel">
          <PointCloudView
            pcaResult={pcaResult}
            isLoading={isLoading}
          />
        </div>
      </main>

      <footer className="app-footer">
        <ClusterPanel
          cluster={cluster}
          onRemove={removeLatent}
          onClear={clearCluster}
          isLoading={isLoading}
        />
        {error && <div className="error-banner">{error}</div>}
      </footer>
    </div>
  );
}

export default App;
