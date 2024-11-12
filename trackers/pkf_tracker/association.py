import os
import numpy as np
import papy


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))

def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)                                              
    return(o)  

def giou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)  
    iou = wh / union

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1 
    hc = yyc2 - yyc1 
    assert((wc > 0).all() and (hc > 0).all())
    area_enclose = wc * hc 
    giou = iou - (area_enclose - union) / area_enclose
    giou = (giou + 1.)/2.0 # resize from (-1,1) to (0,1)
    return giou

def diou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh) 
    iou = wh / union
    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    diou = iou - inner_diag / outer_diag

    return (diou + 1) / 2.0 # resize from (-1,1) to (0,1)

def ciou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh) 
    iou = wh / union

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    # prevent dividing over zero. add one pixel shift
    h2 = h2 + 1.
    h1 = h1 + 1.
    arctan = np.arctan(w2/h2) - np.arctan(w1/h1)
    v = (4 / (np.pi ** 2)) * (arctan ** 2)
    S = 1 - iou 
    alpha = v / (S+v)
    ciou = iou - inner_diag / outer_diag - alpha * v
    
    return (ciou + 1) / 2.0 # resize from (-1,1) to (0,1)

def ct_dist(bboxes1, bboxes2):
    """
        Measure the center distance between two sets of bounding boxes,
        this is a coarse implementation, we don't recommend using it only
        for association, which can be unstable and sensitive to frame rate
        and object speed.
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    ct_dist = np.sqrt(ct_dist2)

    # The linear rescaling is a naive version and needs more study
    ct_dist = ct_dist / ct_dist.max()
    return ct_dist.max() - ct_dist # resize to (0,1)


def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:,0] + dets[:,2])/2.0, (dets[:,1]+dets[:,3])/2.0
    CX2, CY2 = (tracks[:,0] + tracks[:,2]) /2.0, (tracks[:,1]+tracks[:,3])/2.0
    dx = CX1 - CX2 
    dy = CY1 - CY2 
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm 
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def ambig_check(score_matrix, matches_bin, unmatched_trackers, ambig_thre=0.9, use_ocr=False):
    '''
        Get the ambiguous detections and trackers as described in the paper
    '''
    ambig_dets, ambig_trks = [], []
    # for d in matches_bin[:,0]:
    for k in range(len(matches_bin)):
        d, t = matches_bin[k, :]
        if score_matrix[d, :].sum() == 0:
            continue
        
        if len(matches_bin[:,1]) == 1:
            continue

        det_score_matched = score_matrix[d, matches_bin[:,1]]

        sort_indices = np.argsort(det_score_matched)[::-1]
        det_score_sorted = det_score_matched[sort_indices]
        
        if (det_score_sorted[0] * ambig_thre) < det_score_sorted[1]:
            ambig_dets.append(d)
            ambig_trks.append(matches_bin[:, 1][sort_indices[0]])

            # ambig_trks.append(matches_bin[:, 1][sort_indices[1]])
            for i in range(1, len(sort_indices)):
                if not (det_score_sorted[i-1] * ambig_thre) < det_score_sorted[i]:
                    break
                ambig_trks.append(matches_bin[:, 1][sort_indices[i]])

    # add qualified unmatched trackers to ambig_trks
    unmatched_trks2del = []
    
    if use_ocr:
        score_thre = 0.6
        for t in unmatched_trackers:
            scores_trk = score_matrix[:, t]

            if scores_trk.max() < score_thre:
                continue

            ambig_trks.append(t)
            unmatched_trks2del.append(t)

    unmatched_trks2del = np.unique(np.array(unmatched_trks2del, dtype=int))
                
    # add matched dets/trks if its corresponding trk/det is ambiguous
    ambig_dets2add, ambig_trks2add = [], []
    for i in range(len(matches_bin)):
        d, t = matches_bin[i, :]

        if score_matrix[d, t] > 0.95 and not use_ocr:
            continue

        if t in ambig_trks:
            ambig_dets2add.append(d)
        if d in ambig_dets:
            ambig_trks2add.append(t)

    ambig_dets.extend(ambig_dets2add)
    ambig_trks.extend(ambig_trks2add)

    ambig_trks = np.unique(np.array(ambig_trks, dtype=int))
    ambig_dets = np.unique(np.array(ambig_dets, dtype=int))
    
    return ambig_dets, ambig_trks, unmatched_trks2del

def prob_matrix_v0(score_matrix, ambig_dets, ambig_trks):
    ambig_score_matrix = score_matrix[ambig_dets, :][:, ambig_trks]
    cost_matrix = 1. / (ambig_score_matrix + 1e-16)
    ambig_prob_matrix = np.exp(-2 * cost_matrix)
    ambig_prob_matrix = ambig_prob_matrix / ambig_prob_matrix.sum(axis=1, keepdims=True)

    return ambig_prob_matrix


def prob_assignment(iou_matrix, score_matrix, iou_thresh=0.1, 
                    ambig_thresh = 0.9, use_ocr=False, binary=False, weight_thre=0.3):
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_thresh).astype(int)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-score_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))
        
        matched_dets = np.empty(shape=(0), dtype=int)
        matched_trks = np.empty(shape=(0), dtype=int)
        weights = np.empty(shape=score_matrix.shape)
        unmatched_detections = np.arange(score_matrix.shape[0])
        unmatched_trackers = np.arange(score_matrix.shape[1])

        return matched_dets, matched_trks, weights, unmatched_detections, unmatched_trackers

    unmatched_detections, unmatched_trackers = [], []
    for d in range(score_matrix.shape[0]):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)

    for t in range(score_matrix.shape[1]):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    # filter out matched with low score
    matches_bin = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_thresh):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches_bin.append(m.reshape(1,2))
    
    if len(matches_bin) == 0:
        matches_bin = np.empty((0,2), dtype=int)
    else:
        matches_bin = np.concatenate(matches_bin, axis=0)
    
    if not binary:
        ambig_dets, ambig_trks, unmatched_trks2del = \
            ambig_check(score_matrix, matches_bin, unmatched_trackers, ambig_thresh, use_ocr)
        
        for t in unmatched_trks2del:
            unmatched_trackers.remove(t)

        ############## construct the probability matrix with the score (IoU) matrix ##############
        ambig_prob_matrix = prob_matrix_v0(score_matrix, ambig_dets, ambig_trks)
        
        ambig_weights = papy.compute_weights(ambig_prob_matrix)
        
    weights = np.zeros_like(score_matrix)
    weights[matches_bin[:,0], matches_bin[:,1]] = 1
    if not binary and ambig_weights is not None:
        weights[np.ix_(ambig_dets, ambig_trks)] = ambig_weights

    for t in range(weights.shape[1]):
        if weights[:, t].max() < weight_thre:
            unmatched_trackers.append(t)

    unmatched_detections = np.unique(np.array(unmatched_detections, dtype=int))
    unmatched_trackers = np.unique(np.array(unmatched_trackers, dtype=int))

    matched_dets = np.array([d for d in range(score_matrix.shape[0]) if d not in unmatched_detections])
    matched_trks = np.array([t for t in range(score_matrix.shape[1]) if t not in unmatched_trackers])
    # matches = np.stack([matched_dets, matched_trks], axis=1)
    if len(matched_dets) > 0:
        weights = weights[np.ix_(matched_dets, matched_trks)]
    else:
        weights = np.zeros((len(matched_dets), len(matched_trks)))

    return matched_dets, matched_trks, weights, unmatched_detections, unmatched_trackers


def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight, 
              ambig_thresh, use_ocm, use_ocr, binary=False, weight_thre=0.3, img_info=None):
    if(len(trackers)==0):
        return np.empty((0),dtype=int), np.empty((0),dtype=int), \
            np.empty((0, 0),dtype=float), np.arange(len(detections)), np.empty((0),dtype=int)

    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi - np.abs(diff_angle)) / np.pi # arccos return range [0, pi], the value can be non-negative

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:,4]<0)] = 0
    
    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    
    if use_ocm:
        score_matrix = iou_matrix + angle_diff_cost
    else:
        score_matrix = iou_matrix
    
    matched_dets, matched_trks, weights, unmatched_dets, unmatched_trks = \
        prob_assignment(iou_matrix, score_matrix, iou_threshold, 
                        ambig_thresh, use_ocr, binary, weight_thre)
    
    return matched_dets, matched_trks, weights, unmatched_dets, unmatched_trks


def ambig_check_public(det_cates, trk_cates, score_matrix, matches_bin, unmatched_trackers, ambig_thre=0.9):
    det_cates_unique = np.unique(det_cates).astype(int)

    ambig_dets_all, ambig_trks_all = {}, {}
    for det_cate in det_cates_unique:
        
        ambig_dets, ambig_trks = [], []
        for d in matches_bin[:,0]:
            
            if det_cates[d] != det_cate:
                continue
            if score_matrix[d, :].sum() == 0:
                continue
            if len(matches_bin[:,1]) == 1:
                continue
            
            trk_matched_cate = []
            for t in matches_bin[:,1]:
                if trk_cates[t] == det_cate:
                    trk_matched_cate.append(t)

            if len(trk_matched_cate) < 2:
                continue

            det_score_matched = score_matrix[d, trk_matched_cate]
            sort_indices = np.argsort(det_score_matched)[::-1]
            det_score_sorted = det_score_matched[sort_indices]

            ambig_thre = 0.9
            if (det_score_sorted[0] * ambig_thre) < det_score_sorted[1]:
                ambig_dets.append(d)
                ambig_trks.append(trk_matched_cate[sort_indices[0]])

                for i in range(1, len(sort_indices)):
                    if not (det_score_sorted[i-1] * ambig_thre) < det_score_sorted[i]:
                        break
                    ambig_trks.append(trk_matched_cate[sort_indices[i]])

        # add qualified unmatched trackers to ambig_trks
        unmatched_trks2del = []
        # score_thre = 0.6
        score_thre = 1e16
        for t in unmatched_trackers:
            if trk_cates[t] != det_cate:
                continue
            
            scores_trk = score_matrix[:, t]
            if scores_trk.max() < score_thre:
                continue
            
            ambig_trks.append(t)
            unmatched_trks2del.append(t)

        # add matched dets/trks if its corresponding trk/det is ambiguous
        ambig_dets2add, ambig_trks2add = [], []
        for i in range(len(matches_bin)):
            d, t = matches_bin[i, :]
            if det_cates[d] != det_cate:
                continue

            if t in ambig_trks:
                ambig_dets2add.append(d)
            if d in ambig_dets:
                ambig_trks2add.append(t)
        
        ambig_dets.extend(ambig_dets2add)
        ambig_trks.extend(ambig_trks2add)

        ambig_dets = np.unique(np.array(ambig_dets, dtype=int))
        ambig_trks = np.unique(np.array(ambig_trks, dtype=int))

        ambig_dets_all[det_cate] = ambig_dets
        ambig_trks_all[det_cate] = ambig_trks
    
    unmatched_trks2del = np.unique(np.array(unmatched_trks2del, dtype=int))

    return ambig_dets_all, ambig_trks_all, unmatched_trks2del

def prob_assignment_public(det_cates, trk_cates, iou_matrix, score_matrix, cate_matrix, 
                           iou_thresh=0.1, binary=False):
    
    assert iou_matrix.shape == score_matrix.shape, \
        'iou_matrix and score_matrix should have the same shape'
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_thresh).astype(int)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-score_matrix-cate_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))
    
    unmatched_detections, unmatched_trackers = [], []
    for d in range(score_matrix.shape[0]):
        if(d not in matched_indices[:,0]):
            # print('d:', d, 'score_matrix:', score_matrix.shape)
            unmatched_detections.append(d)

    for t in range(score_matrix.shape[1]):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    # filter out matched with low score
    matches_bin = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_thresh):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches_bin.append(m.reshape(1,2))
    
    if len(matches_bin) == 0:
        matches_bin = np.empty((0,2), dtype=int)
    else:
        matches_bin = np.concatenate(matches_bin, axis=0)
    
    if not binary:
        ambig_dets_all, ambig_trks_all, unmatched_trks2del = \
            ambig_check_public(det_cates, trk_cates, score_matrix, matches_bin, unmatched_trackers)

        for t in unmatched_trks2del:
            unmatched_trackers.remove(t)

    weights = np.zeros_like(score_matrix)
    weights[matches_bin[:,0], matches_bin[:,1]] = 1
    if not binary:
        for cate in ambig_dets_all.keys():
            ambig_dets = ambig_dets_all[cate]
            ambig_trks = ambig_trks_all[cate]
            
            ambig_prob_matrix = prob_matrix_v0(score_matrix, ambig_dets, ambig_trks)
            # ambig_prob_matrix = prob_matrix_square(score_matrix, ambig_dets, ambig_trks)

            if papy.compute_permanent(ambig_prob_matrix) == 0:
                ambig_weights = ambig_prob_matrix
            else:
                ambig_weights = papy.compute_weights(ambig_prob_matrix)
            
            weights[np.ix_(ambig_dets, ambig_trks)] = ambig_weights

    for t in range(weights.shape[1]):
        if weights[:, t].max() < 0.3:
            unmatched_trackers.append(t)
    
    unmatched_detections = np.unique(np.array(unmatched_detections, dtype=int))
    unmatched_trackers = np.unique(np.array(unmatched_trackers, dtype=int))

    matched_dets = np.array([d for d in range(score_matrix.shape[0]) if d not in unmatched_detections])
    matched_trks = np.array([t for t in range(score_matrix.shape[1]) if t not in unmatched_trackers])
    
    if len(matched_dets) > 0:
        weights = weights[np.ix_(matched_dets, matched_trks)]
    else:
        weights = np.zeros((len(matched_dets), len(matched_trks)))

    return matched_dets, matched_trks, weights, unmatched_detections, unmatched_trackers


def associate_public(detections, trackers, det_cates, iou_threshold, 
        velocities, previous_obs, vdc_weight, binary=False):
    
    if (len(trackers)==0 or len(detections)==0):
        return np.empty((0),dtype=int), np.empty((0),dtype=int), np.empty((0, 0),dtype=int), \
            np.arange(len(detections)), np.arange(len(trackers))
    # if (len(detections)==0):
    #     return np.empty((0),dtype=int), np.empty((0),dtype=int), np.empty((0, 0),dtype=int), \
    #         np.empty((0),dtype=int), np.arange(len(trackers))
    """
        Cost from the velocity direction consistency
    """
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi - np.abs(diff_angle)) / np.pi # arccos return range [0, pi], the value can be non-negative

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:,4]<0)] = 0
    
    scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    
    """
        Cost from IoU
    """
    iou_matrix = iou_batch(detections, trackers)

    """
        With multiple categories, generate the cost for catgory mismatch
    """
    num_dets = detections.shape[0]
    num_trk = trackers.shape[0]
    cate_matrix = np.zeros((num_dets, num_trk))
    for i in range(num_dets):
            for j in range(num_trk):
                if det_cates[i] != trackers[j, 4]:
                        cate_matrix[i][j] = -1e16

    score_matrix = iou_matrix + angle_diff_cost
    # score_matrix = iou_matrix
    trk_cates = trackers[:, 4]
    matched_dets, matched_trks, weights, unmatched_dets, unmatched_trks = \
        prob_assignment_public(det_cates, trk_cates, iou_matrix, score_matrix, cate_matrix, 
                               iou_threshold, binary)
    
    return matched_dets, matched_trks, weights, unmatched_dets, unmatched_trks

