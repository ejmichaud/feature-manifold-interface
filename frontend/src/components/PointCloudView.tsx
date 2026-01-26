/**
 * Point cloud visualization component using Plotly.
 *
 * Displays PCA projections of the activation point cloud:
 * - 2 interactive 3D plots (PCs 1-3 and PCs 4-6)
 * - 8 2D plots showing PC pairs (1-2, 3-4, ..., 15-16)
 */

import Plot from 'react-plotly.js';
import type { PCAResult } from '../types';

interface PointCloudViewProps {
  pcaResult: PCAResult | null;
  isLoading: boolean;
}

export function PointCloudView({ pcaResult, isLoading }: PointCloudViewProps) {
  if (isLoading) {
    return (
      <div className="point-cloud-view loading">
        <div className="loading-spinner">Computing PCA...</div>
      </div>
    );
  }

  if (!pcaResult || pcaResult.n_points === 0) {
    return (
      <div className="point-cloud-view empty">
        <div className="empty-message">
          Select latents from the graph to visualize their activation geometry.
        </div>
      </div>
    );
  }

  const { points, explained_variance_ratio, n_points, subsampled } = pcaResult;

  // Extract PC values (points is n_points x n_components)
  const getPC = (pcIndex: number): number[] =>
    points.map(p => p[pcIndex] ?? 0);

  // Create 3D plot data
  const create3DTrace = (pcStart: number) => ({
    x: getPC(pcStart),
    y: getPC(pcStart + 1),
    z: getPC(pcStart + 2),
    mode: 'markers' as const,
    type: 'scatter3d' as const,
    marker: {
      size: 2,
      color: '#4a90d9',
      opacity: 0.6,
    },
    hoverinfo: 'skip' as const,
  });

  // Create 2D plot data
  const create2DTrace = (pcX: number, pcY: number) => ({
    x: getPC(pcX),
    y: getPC(pcY),
    mode: 'markers' as const,
    type: 'scattergl' as const,
    marker: {
      size: 3,
      color: '#4a90d9',
      opacity: 0.5,
    },
    hoverinfo: 'skip' as const,
  });

  // Layout for 3D plots
  const layout3D = (title: string, pcStart: number) => ({
    title: {
      text: title,
      font: { size: 12 },
    },
    autosize: true,
    margin: { l: 0, r: 0, t: 30, b: 0 },
    scene: {
      xaxis: {
        title: `PC${pcStart + 1} (${(explained_variance_ratio[pcStart] * 100).toFixed(1)}%)`,
        titlefont: { size: 10 },
      },
      yaxis: {
        title: `PC${pcStart + 2} (${(explained_variance_ratio[pcStart + 1] * 100).toFixed(1)}%)`,
        titlefont: { size: 10 },
      },
      zaxis: {
        title: `PC${pcStart + 3} (${(explained_variance_ratio[pcStart + 2] * 100).toFixed(1)}%)`,
        titlefont: { size: 10 },
      },
    },
    showlegend: false,
  });

  // Layout for 2D plots
  const layout2D = (pcX: number, pcY: number) => ({
    autosize: true,
    margin: { l: 40, r: 10, t: 10, b: 40 },
    xaxis: {
      title: `PC${pcX + 1}`,
      titlefont: { size: 10 },
    },
    yaxis: {
      title: `PC${pcY + 1}`,
      titlefont: { size: 10 },
    },
    showlegend: false,
  });

  const plotConfig = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'] as const,
  };

  // PC pairs for 2D plots: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
  const pcPairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]];

  return (
    <div className="point-cloud-view">
      <div className="point-cloud-header">
        <span>{n_points.toLocaleString()} points</span>
        {subsampled && <span className="subsampled-badge">(subsampled)</span>}
      </div>

      <div className="plots-container">
        {/* 3D Plots */}
        <div className="plots-3d">
          <div className="plot-3d">
            <Plot
              data={[create3DTrace(0)]}
              layout={layout3D('PCs 1-3', 0) as object}
              config={plotConfig}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler
            />
          </div>
          <div className="plot-3d">
            <Plot
              data={[create3DTrace(3)]}
              layout={layout3D('PCs 4-6', 3) as object}
              config={plotConfig}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler
            />
          </div>
        </div>

        {/* 2D Plots */}
        <div className="plots-2d">
          {pcPairs.map(([pcX, pcY]) => (
            <div key={`pc-${pcX}-${pcY}`} className="plot-2d">
              <Plot
                data={[create2DTrace(pcX, pcY)]}
                layout={layout2D(pcX, pcY) as object}
                config={plotConfig}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
