
import numpy as np
import papy


def associate_binary(detections, trackers):
    if len(trackers) == 0:
        return np.empty((0), dtype=int), np.empty((0), dtype=int), np.empty((0, 0)), \
                np.arange(len(detections)), np.empty((0), dtype=int)

    dist_matrix = np.linalg.norm(detections[:, None, :, 0] - trackers[None, :, :], axis=2)
    matched_dets, matched_trks, weights, unmatched_detections, unmatched_trackers = \
            binary_assignment(dist_matrix)

    return matched_dets, matched_trks, weights, unmatched_detections, unmatched_trackers


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def binary_assignment(dist_matrix):
    
    matched_indices = linear_assignment(dist_matrix)

    unmatched_detections, unmatched_trackers = [], []
    for d in range(dist_matrix.shape[0]):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    for t in range(dist_matrix.shape[1]):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with large distance
    matches_bin = []
    for m in matched_indices:
        matches_bin.append(m.reshape(1, 2))

    if len(matches_bin) == 0:
        matches_bin = np.empty((0,2), dtype=int)
    else:
        matches_bin = np.concatenate(matches_bin, axis=0)

    weights = np.zeros(dist_matrix.shape)
    weights[matches_bin[:, 0], matches_bin[:, 1]] = 1.0
    
    unmatched_detections = np.unique(np.array(unmatched_detections, dtype=int))
    unmatched_trackers = np.unique(np.array(unmatched_trackers, dtype=int))

    matched_dets = np.array([d for d in range(dist_matrix.shape[0]) if d not in unmatched_detections])
    matched_trks = np.array([t for t in range(dist_matrix.shape[1]) if t not in unmatched_trackers])

    if len(matched_dets) > 0 and len(matched_trks) > 0:
        weights = weights[np.ix_(matched_dets, matched_trks)]
    else:
        weights = np.zeros((len(matched_dets), len(matched_trks)))

    return matched_dets, matched_trks, weights, unmatched_detections, unmatched_trackers


if __name__ == "__main__":
    print("Data association module.")
