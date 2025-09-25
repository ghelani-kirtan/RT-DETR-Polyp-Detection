import lap
import numpy as np


def linear_assignment(cost_matrix, thresh):
    cost_matrix = np.atleast_2d(cost_matrix)
    if (
        cost_matrix.size == 0
        or cost_matrix.ndim != 2
        or any(d == 0 for d in cost_matrix.shape)
    ):
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1]),
        )

    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches = np.array([[ix, mx] for ix, mx in enumerate(x) if mx >= 0])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    return matches, unmatched_a, unmatched_b


def iou_distance(atracks, btracks):
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64),
    )
    cost_matrix = 1 - _ious
    return cost_matrix


def ious(atlbrs, btlbrs):
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64),
    )

    return ious


def bbox_ious(boxes1, boxes2, eps=1e-9):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]))

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., np.newaxis, :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., np.newaxis, 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area[..., np.newaxis] + boxes2_area - inter_area

    ious = 1.0 * inter_area / np.maximum(union_area, eps)

    #! FIXED: This is the definitive fix for the IndexError.
    # If the union_area is zero (due to a zero-area bbox), the IoU calculation
    # results in NaN. This line ensures that any such invalid IoU values are
    # set to 0, preventing the assignment algorithm from failing.
    ious[union_area <= 0] = 0.0

    return ious


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
