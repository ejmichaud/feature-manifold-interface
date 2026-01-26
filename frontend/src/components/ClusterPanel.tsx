/**
 * Cluster panel component.
 *
 * Displays the current cluster of selected latents as chips/tags
 * with the ability to remove individual latents or clear all.
 */

interface ClusterPanelProps {
  cluster: number[];
  onRemove: (latentId: number) => void;
  onClear: () => void;
  isLoading: boolean;
}

export function ClusterPanel({
  cluster,
  onRemove,
  onClear,
  isLoading,
}: ClusterPanelProps) {
  return (
    <div className="cluster-panel">
      <div className="cluster-header">
        <span className="cluster-title">
          Cluster ({cluster.length} latent{cluster.length !== 1 ? 's' : ''})
        </span>
        {cluster.length > 0 && (
          <button
            className="clear-button"
            onClick={onClear}
            disabled={isLoading}
          >
            Clear All
          </button>
        )}
      </div>

      <div className="cluster-chips">
        {cluster.length === 0 ? (
          <span className="empty-hint">
            Click nodes in the graph to add to cluster
          </span>
        ) : (
          cluster.map((latentId) => (
            <div key={latentId} className="latent-chip">
              <span className="chip-label">Latent {latentId}</span>
              <button
                className="chip-remove"
                onClick={() => onRemove(latentId)}
                disabled={isLoading}
                title="Remove from cluster"
              >
                &times;
              </button>
            </div>
          ))
        )}
      </div>

      {isLoading && (
        <div className="cluster-loading">
          <span className="loading-indicator">Updating...</span>
        </div>
      )}
    </div>
  );
}
