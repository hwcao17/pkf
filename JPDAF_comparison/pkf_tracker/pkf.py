
import numpy as np
from .association import associate_binary
import time

class PAKalmanBoxTracker(object):
    
    count = 0

    def __init__(self, det, meas_noise=0.75):
        from .pakalmanfilter import PAKalmanFilter

        self.kf = PAKalmanFilter(dim_x=4, dim_z=2)

        # initialize the motion and measurement matrices
        self.kf.F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        # initialize the covariance matrices
        self.kf.P = np.diag([1.5, 0.5, 1.5, 0.5])
        self.kf.V = np.eye(2) * meas_noise
        self.kf.W = np.array([[1/6, 1/4, 0., 0.],
                              [1/4, 1/2, 0., 0.],
                              [0., 0., 1/6, 1/4],
                              [0., 0., 1/4, 1/2]]) * 0.01

        # initialize the state
        self.kf.x = np.array([[det[0]], [0], [det[1]], [0]])

        self.id = PAKalmanBoxTracker.count
        PAKalmanBoxTracker.count += 1
    
    def update(self, dets, weights):
        if dets is not None:
            self.kf.update(dets, weights)
        else:
            self.kf.update(None, None)

    def predict(self):
        self.kf.predict()
        return self.kf.x.T[0]
    
    def get_state(self):
        return self.kf.x.T[0]


class JPDAFTracker(object):
    count = 0

    def __init__(self, det, meas_noise=0.75):
        from .jpdaf import JPDAF

        self.kf = JPDAF(dim_x=4, dim_z=2)

        # initialize the motion and measurement matrices
        self.kf.F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        # initialize the covariance matrices
        self.kf.P = np.diag([1.5, 0.5, 1.5, 0.5])
        self.kf.V = np.eye(2) * meas_noise
        self.kf.W = np.array([[1/6, 1/4, 0., 0.],
                              [1/4, 1/2, 0., 0.],
                              [0., 0., 1/6, 1/4],
                              [0., 0., 1/4, 1/2]]) * 0.01

        # initialize the state
        self.kf.x = np.array([[det[0]], [0], [det[1]], [0]])

        self.id = JPDAFTracker.count
        JPDAFTracker.count += 1

    def update(self, dets, weights):
        if dets is not None:
            self.kf.update(dets, weights)
        else:
            self.kf.update(None, None)

    def predict(self):
        self.kf.predict()
        return self.kf.x.T[0]
    
    def get_state(self):
        return self.kf.x.T[0]
    

class PKFTracker(object):
    def __init__(self, meas_noise=0.75):
        self.meas_noise = meas_noise

        self.trackers = []
        self.trackers_JPDAF = []
        PAKalmanBoxTracker.count = 0
        self.timestep = 0

    def get_current_tracks(self):
        ret = []
        i = len(self.trackers)
        for trk in self.trackers:
            
            if np.any(np.isnan(trk.get_state())):
                continue
            ret.append(np.concatenate((trk.get_state(), np.array([trk.id + 1]))).reshape(1, -1))

            i -= 1
        
        if len(ret) > 0:
            return np.concatenate(ret)
        else:
            return np.empty((0, 5))
    
    def get_current_tracks_JPDAF(self):
        ret = []
        i = len(self.trackers_JPDAF)
        for trk in self.trackers_JPDAF:
            
            if np.any(np.isnan(trk.get_state())):
                continue
            ret.append(np.concatenate((trk.get_state(), np.array([trk.id + 1]))).reshape(1, -1))

            i -= 1
        
        if len(ret) > 0:
            return np.concatenate(ret)
        else:
            return np.empty((0, 5))
    
    def update(self, dets, associations, binary=False):
        
        ############### prediction ###############
        # get predicted states from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            if self.timestep > 0:
                state = self.trackers[t].predict()
            else:
                state = self.trackers[t].get_state()
            trk[:] = state
            if np.any(np.isnan(state)):
                to_del.append(t)
        
        self.timestep += 1

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in to_del:
            del self.trackers[t]

        if dets is None and associations is None:
            print('No detections and associations')
            return self.get_current_tracks()

        if associations is None:
            # print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')        
            ############### data association ###############
            # print('trks:', trks[:, ::2])
            matched_dets, matched_trks, weights, _, _ = associate_binary(dets, trks[:, ::2])

            # update trackers with matched detections
            updata_time = 0
            for j in range(len(matched_trks)):
                
                weights_trk = weights[:, j]
                # valid_indices = [i for i in range(len(weights_trk)) \
                #                 if weights_trk[i] > self.update_weight_thresh and np.isfinite(weights_trk[i])]
                
                valid_indices = [i for i in range(len(weights_trk)) \
                                if weights_trk[i] > 0.9 and np.isfinite(weights_trk[i])]
                
                if len(valid_indices) == 0:
                    continue
                
                valid_det_indices = matched_dets[valid_indices]
                dets_update = dets[valid_det_indices]
                weights_update = weights_trk[valid_indices]

                # if weights_update.max() < 0.9:
                #     print('use probablistic association')
                
                # weights_update = weights_update / np.sum(weights_update)
                tic = time.time()
                if len(dets_update> 0):
                    self.trackers[matched_trks[j]].update(dets_update, weights_update)
                toc = time.time()
                updata_time += toc - tic

                # weights_trk = weights[:, j]
                # self.trackers[matched_trks[j]].update(dets, weights_trk)

            ############### results to be returned ###############
            return self.get_current_tracks(), updata_time
        
        else:
            updata_time = 0
            for i in range(len(associations)):
                dets_raw = associations[i]['measurements']
                weights_raw = associations[i]['probabilities']    
                
                assert len(dets_raw) == len(weights_raw), \
                    'dets_raw and weights_raw must have the same length'
                dets_update = []
                weights_update = []

                for j in range(len(dets_raw)):
                    if dets_raw[j] is None:
                        continue
                    else:
                        dets_update.append(np.array(dets_raw[j]))
                        weights_update.append(float(weights_raw[j]))
                
                tic = time.time()
                if len(dets_update) > 0: 
                    dets_update = np.array(dets_update)
                    self.trackers[i].update(dets_update, weights_update)
                toc = time.time()
                updata_time += toc - tic

        return self.get_current_tracks(), updata_time

    def update_JPDAF(self, dets, associations):
        ############### prediction ###############
        # get predicted states from existing trackers
        trks = np.zeros((len(self.trackers_JPDAF), 4))
        to_del = []
        for t, trk in enumerate(trks):
            if self.timestep > 0:
                state = self.trackers_JPDAF[t].predict()
            else:
                state = self.trackers_JPDAF[t].get_state()
            trk[:] = state
            if np.any(np.isnan(state)):
                to_del.append(t)
        
        self.timestep += 1

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in to_del:
            del self.trackers_JPDAF[t]

        if dets is None and associations is None:
            print('No detections and associations')
            return self.get_current_tracks()
        
        for i in range(len(associations)):
            dets_raw = associations[i]['measurements']
            weights_raw = associations[i]['probabilities']    
            
            assert len(dets_raw) == len(weights_raw), \
                'dets_raw and weights_raw must have the same length'
            dets_update = []
            weights_update = []

            for j in range(len(dets_raw)):
                if dets_raw[j] is None:
                    continue
                else:
                    dets_update.append(np.array(dets_raw[j]))
                    weights_update.append(float(weights_raw[j]))
            
            if len(dets_update) > 0: 
                dets_update = np.array(dets_update)
                self.trackers_JPDAF[i].update(dets_update, weights_update)

        return self.get_current_tracks_JPDAF()

    def add_tracker(self, det):
        self.trackers.append(PAKalmanBoxTracker(det, self.meas_noise))

    def add_tracker_JPDAF(self, det):
        self.trackers_JPDAF.append(JPDAFTracker(det, self.meas_noise))
