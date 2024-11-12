import pickle
import numpy as np
from .association import *


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

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]

def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


class PAKalmanBoxTracker(object):
   
    count = 0
    def __init__(self, bbox, delta_t=3):
        """
        Initialises a tracker using initial bounding box.

        """
        from .pakalmanfilter import PAKalmanFilter
        self.kf = PAKalmanFilter(dim_x=7, dim_z=4)

        # initialize the motion and measurement matrices
        self.kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]])

        # initialize the covariance matrices
        self.kf.V[2:,2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.W[-1,-1] *= 0.01
        self.kf.W[4:,4:] *= 0.01

        # initialize the state vector
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0

        self.id = PAKalmanBoxTracker.count
        PAKalmanBoxTracker.count += 1

        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bboxes, weights, dominant_det=None):
        """
        Updates the state vector with observed bbox.
        """
        if bboxes is not None:
            
            if dominant_det is None: 
                dominant_idx = np.argmax(weights)
                bbox = bboxes[dominant_idx]
            else:
                bbox = dominant_det

            if self.last_observation.sum() >= 0: # there is a previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)    

            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            obs_update = []
            for i in range(len(bboxes)):
                obs_update.append(convert_bbox_to_z(bboxes[i]))

            self.time_since_update = 0
            self.kf.update(obs_update, weights)
            self.hits += 1
            self.hit_streak += 1
        else:
            self.kf.update(None, None)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if(self.time_since_update > 0):
            self.hit_streak = 0            
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x), self.kf.x[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist}


class PKFTracker(object):
    def __init__(self, det_thresh, max_age=30, min_hits=3, 
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, 
        use_ocr=False, use_ocm=False, ambig_thresh=0.9, update_weight_thresh=0.3):
        """
        Sets key parameters for PKFTracker
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_ocr = use_ocr
        self.use_ocm = use_ocm
        # print('use ocr:', use_ocr, 'use ocm:', use_ocm)
        self.ambig_thresh = ambig_thresh
        self.update_weight_thresh = update_weight_thresh

        PAKalmanBoxTracker.count = 0

    def update(self, output_results, img_info, img_size):
        """
        Params:
          output_results - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))
        
        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]
        
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        
        remain_inds = scores > self.det_thresh
        dets_first = dets[remain_inds]
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos, score = self.trackers[t].predict()
            trk[:] = [pos[0, 0], pos[0, 1], pos[0, 2], pos[0, 3], score]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            # trk_innov_covs.pop(t)
        
        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
            First round of association
        """
        binary = False
        weight_thre = self.update_weight_thresh
        matched_dets, matched_trks, weights, unmatched_dets, unmatched_trks = associate(
            dets_first, trks, self.iou_threshold, velocities, k_observations, self.inertia, 
            self.ambig_thresh, self.use_ocm, self.use_ocr, binary, weight_thre, img_info)
        
        ######### save the matches and weights for visualizations #########
        # frame_id, video_id = img_info[2].item(), img_info[3].item()
        # img_file_name = img_info[4]
        # video_name = img_file_name[0].split('/')[0]
        # # save_root = 'YOLOX_outputs/dancetrack/pkf/pkf_val_assoc'
        # save_root = 'YOLOX_outputs/mot/MOT20-train/pkf/assoc'
        # assoc_data = {'frame_id': frame_id, 'video_id': video_id, 'dets': dets_first, 'trks': trks, 
        #               'matched_dets': matched_dets, 'matched_trks': matched_trks, 'weights': weights, 
        #               'unmatched_dets': unmatched_dets, 'unmatched_trks': unmatched_trks}
        # # print('frame_id:', frame_id, 'video_id:', video_id)
        # # save_folder = os.path.join(save_root, '%03d' % video_id)
        # save_folder = os.path.join(save_root, video_name)
        # os.makedirs(save_folder, exist_ok=True)
        # save_path = os.path.join(save_folder, f'frame_{frame_id}.pkl')
        # with open(save_path, 'wb') as f:
        #     pickle.dump(assoc_data, f)

        for j in range(len(matched_trks)):
            weights_trk = weights[:, j]
            valid_indices = [i for i in range(len(weights_trk)) \
                             if (weights_trk[i] > weight_thre and np.isfinite(weights_trk[i]))]
            
            if len(valid_indices) == 0:
                continue
            
            valid_det_indices = matched_dets[valid_indices]
            dets_update = dets_first[valid_det_indices]
            weights_update = weights_trk[valid_indices]

            self.trackers[matched_trks[j]].update(dets_update, weights_update)
        
        """
            Second round of associaton by OCR
        """
        if self.use_ocr and unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)

            if iou_left.max() > self.iou_threshold:
                matched_dets_2, matched_trks_2, weights_2, _, _ = \
                    prob_assignment(iou_left, iou_left, self.iou_threshold, 
                                    self.ambig_thresh, self.use_ocr, binary)
                
                unmatched_dets_to_del = []
                unmatched_trks_to_del = []
                for j in range(len(matched_trks_2)):
                    weights_trk = weights_2[:, j]
                    valid_indices = [i for i in range(len(weights_trk)) \
                                     if (weights_trk[i] > 0.3 and np.isfinite(weights_trk[i]))]

                    if len(valid_indices) == 0:
                        continue
                    
                    valid_det_indices = matched_dets_2[valid_indices]
                    dets_update = left_dets[valid_det_indices]
                    weights_update = weights_trk[valid_indices]

                    self.trackers[unmatched_trks[matched_trks_2[j]]].update(dets_update, weights_update)
                    
                    dominant_idx = np.argmax(weights_update)

                    # delete matched detections in OCR
                    if weights_update[dominant_idx] > 0.9:
                        unmatched_dets_to_del.append(unmatched_dets[valid_det_indices[dominant_idx]])

                    unmatched_trks_to_del.append(matched_trks_2[j])

                unmatched_dets = np.setdiff1d(unmatched_dets, unmatched_dets_to_del)
                unmatched_trks = np.setdiff1d(unmatched_trks, unmatched_trks_to_del)

        for m in unmatched_trks:
            self.trackers[m].update(None, None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            
            trk = PAKalmanBoxTracker(dets[i, :], delta_t=self.delta_t)
            self.trackers.append(trk)
        
        # results to return
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            
            if (trk.time_since_update < 1) \
                and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
            
        if len(ret) > 0:
            return np.concatenate(ret)
        
        return np.empty((0, 5))

    def update_public(self, dets, cates, scores):
        self.frame_count += 1
        # print('num_dets:', dets.shape[0])
        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh
        
        cates = cates[remain_inds]
        dets_first = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []

        for t, trk in enumerate(trks):
            pos, score = self.trackers[t].predict()
            # print('pos:', pos)
            cat = self.trackers[t].cate
            trk[:] = [pos[0, 0], pos[0, 1], pos[0, 2], pos[0, 3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched_dets, matched_trks, weights, unmatched_dets, unmatched_trks = associate_public(
            dets_first, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
        
        weight_thre = 0.3
        for j in range(len(matched_trks)):
            weights_trk = weights[:, j]
            valid_indices = [i for i in range(len(weights_trk)) \
                             if (weights_trk[i] > weight_thre and np.isfinite(weights_trk[i]))]
            
            if len(valid_indices) == 0:
                continue
            
            valid_det_indices = matched_dets[valid_indices]
            dets_update = dets_first[valid_det_indices]
            weights_update = weights_trk[valid_indices]
            
            self.trackers[matched_trks[j]].update(dets_update, weights_update)

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            # left_dets = dets_second
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)

            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                            """
                                For some datasets, such as KITTI, there are different categories,
                                we have to avoid associate them together.
                            """
                            cate_matrix[i][j] = -1e16
            
            if iou_left.max() > self.iou_threshold - 0.1:
                
                matched_dets_2, matched_trks_2, weights_2, _, _ = \
                    prob_assignment_public(det_cates_left, trk_cates_left, iou_left, iou_left, 
                                           cate_matrix, self.iou_threshold)
                
                unmatched_dets_to_del = []
                unmatched_trks_to_del = []
                for j in range(len(matched_trks_2)):
                    weights_trk = weights_2[:, j]
                    valid_indices = [i for i in range(len(weights_trk)) \
                                     if (weights_trk[i] > 0.3 and np.isfinite(weights_trk[i]))]

                    if len(valid_indices) == 0:
                        continue
                    
                    valid_det_indices = matched_dets_2[valid_indices]
                    dets_update = left_dets[valid_det_indices]
                    weights_update = weights_trk[valid_indices]
                    
                    self.trackers[unmatched_trks[matched_trks_2[j]]].update(dets_update, weights_update)
                    
                    dominant_idx = np.argmax(weights_update)
                    unmatched_dets_to_del.append(valid_det_indices[dominant_idx])
                    unmatched_trks_to_del.append(matched_trks_2[j])

                unmatched_dets = np.setdiff1d(unmatched_dets, unmatched_dets_to_del)
                unmatched_trks = np.setdiff1d(unmatched_trks, unmatched_trks_to_del)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = PAKalmanBoxTracker(dets[i, :], delta_t=self.delta_t)
            trk.cate = cates[i]
            self.trackers.append(trk)
        
        i = len(self.trackers)
        ret = []
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age):
                  self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 7))
