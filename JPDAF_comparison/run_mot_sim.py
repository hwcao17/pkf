
import os
import numpy as np
from pkf_tracker.pkf import PKFTracker
import matplotlib.pyplot as plt
import pickle

# stonesoup tools
from datetime import datetime, timedelta
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.update import GaussianStateUpdate

import time
from tqdm import tqdm
import argparse


class MOTEvaluator:

    def __init__(self, args):
        self.args = args
        self.n_obj = args.n_obj
        self.noise_scale = args.noise_scale
        self.data_path = os.path.join(args.dataset_root, 
                                      "multi-object-data-%d" % self.n_obj)
    
    def generate_measurements(self, gt_trajs):

        meas_cov = np.eye(2) * self.noise_scale
        prob_detect = 0.9

        nstep = gt_trajs.shape[1]
        all_measurements, true_measurements, all_clutters = [], [], []
        
        for k in range(nstep):
            measurement_set, true_measurement_set, clutter_set = [], [], []
            
            for i in range(self.n_obj):
                # generate actual detection
                if np.random.rand() <= prob_detect:
                    gt_pos = gt_trajs[i, k, :2, 3]
                    meas = gt_pos + np.random.multivariate_normal([0, 0], meas_cov)

                    measurement_set.append(meas)
                    true_measurement_set.append(meas)

                # generate clutter
                for _ in range(np.random.randint(2)):
                    x = np.random.uniform(gt_pos[0] - 5, gt_pos[0] + 5)
                    y = np.random.uniform(gt_pos[1] - 5, gt_pos[1] + 5)

                    measurement_set.append(np.array([x, y]))
                    clutter_set.append(np.array([x, y]))

            all_measurements.append(np.array(measurement_set))
            true_measurements.append(np.array(true_measurement_set))
            all_clutters.append(np.array(clutter_set))
        
        return all_measurements, true_measurements, all_clutters

    def load_measurements(self, data_fname):
        
        with open(data_fname, 'rb') as f:
            data = pickle.load(f)
        
        start_time = data['start_time']
        all_measurements_raw = data['measurements_raw']
        all_measurements = data['measurements']
        # print('all_measurements:', len(all_measurements), type(all_measurements[0]))
        for i in range(len(all_measurements)):
            all_measurements[i] = np.array(all_measurements[i])
        
        return start_time, all_measurements_raw, all_measurements


    def evaluate(self):
        if self.args.jpdaf:
            print('Evaluating JPDAF...')
        else:
            print('Evaluating PKF...')

        np.random.seed(1991)

        ########## load ground truth ##########
        gt_trajs_path = os.path.join(self.data_path, 'gt_trajs.npy')
        gt_trajs = np.load(gt_trajs_path)
        assert gt_trajs.shape[0] == self.n_obj, 'gt_trajs.shape[0] != n_obj'

        gt_states = np.zeros([gt_trajs.shape[0], gt_trajs.shape[1], 4])

        for i in range(self.n_obj):
            for k in range(gt_trajs.shape[1]):
                gt_pos = gt_trajs[i, k, :2, 3]
                next_k = min(k + 1, gt_trajs.shape[1] - 1)
                next_pos = gt_trajs[i, next_k, :2, 3]
                gt_velo = next_pos - gt_pos
                gt_states[i, k, 0], gt_states[i, k, 2] = gt_pos[0], gt_pos[1]
                gt_states[i, k, 1], gt_states[i, k, 3] = gt_velo[0], gt_velo[1]

        gt_velos_path = os.path.join(self.data_path, 'gt_velos.npy')
        gt_velos = np.load(gt_velos_path)
        assert gt_velos.shape[0] == self.n_obj, 'gt_velos.shape[0] != n_obj'

        nstep = gt_trajs.shape[1]
        assert gt_trajs.shape[1] == gt_velos.shape[1], 'gt_trajs.shape[1] != gt_velos.shape[1]'

        ########## generate measurements ##########
        if args.n_obj == 10:
            data_fname = os.path.join(self.data_path, 'data_noise_%.2f.pkl' % self.noise_scale)
        else:
            data_fname = os.path.join(self.data_path, 'data.pkl')
        print('data_fname:', data_fname)
        if os.path.exists(data_fname):
            start_time, all_measurements_ss, all_measurements = \
                self.load_measurements(data_fname)
        else:
            start_time = datetime.now()
            all_measurements, true_measurements, all_clutters = self.generate_measurements(gt_trajs)
        
        ############## plot true detections and clutter ##############
        # plt.figure()
        # for k in range(nstep):
        #     for meas in true_measurements[k]:
        #         plt.scatter(meas[0], meas[1], c='b', marker='o')
        #     for clutter in all_clutters[k]:
        #         plt.scatter(clutter[0], clutter[1], c='r', marker='x')
        # for i in range(self.n_obj):
        #     plt.plot(gt_trajs[i, :, 0, 3], gt_trajs[i, :, 1, 3], 'k--')
        # plt.xlim(-20, 20)
        # plt.ylim(-20, 20)
        # plt.show()

        ########## run PKF or JPDAF ##########
        pkf = PKFTracker(self.noise_scale)
        for i in range(self.n_obj):
            # print('gt_trajs[i, 0, :2, 3]:', gt_trajs[i, 0, :2, 3])
            if self.args.jpdaf and not self.args.binary:
                pkf.add_tracker_JPDAF(gt_trajs[i, 0, :2, 3])
            else:
                pkf.add_tracker(gt_trajs[i, 0, :2, 3])    

        if not self.args.binary:
            ########### keep the tracks with stonesoup format ############
            
            # create tracks
            tracks_ss = []
            for i in range(self.n_obj):
                p = gt_trajs[i, 0, :2, 3]
                init_state_cov = np.diag([1.5, 0.5, 1.5, 0.5])
                # print('p:', p)
                prior = GaussianState([[p[0]], [0], [p[1]], [0]], init_state_cov, timestamp=start_time)
                tracks_ss.append(Track([prior]))
            

            # measurement and transition models
            measurement_model = LinearGaussian(
                ndim_state=4,
                mapping=(0, 2),
                noise_covar=np.eye(2) * self.noise_scale
                )
            # print('measurement_model:', measurement_model.matrix())
            transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                                      ConstantVelocity(0.005)])
            predictor = KalmanPredictor(transition_model)
            updater = KalmanUpdater(measurement_model)

            # create data associator
            hypothesiser = PDAHypothesiser(predictor=predictor,
                                        updater=updater,
                                        clutter_spatial_density=0.125,
                                        prob_detect=0.9)
            data_associator = JPDA(hypothesiser=hypothesiser)

        results = []
        update_times = []
        for k in tqdm(range(nstep)):

            ####### association saved from jpdaf sim #######
            # associations_k = []
            # for i in range(self.n_obj):
            #     associations_k.append(associations[i][k])
            # online_targets = pkf.update(all_measurements[k], associations_k)
            ################################################
            
            if self.args.binary:
                tic = time.time()
                online_targets, update_time = pkf.update(all_measurements[k], None, self.args.binary) 
                toc = time.time()

                update_times.append(update_time)
            else:
                ############ association stonesoup #############
                measurements_ss = all_measurements_ss[k]

                hypotheses = data_associator.associate(tracks_ss, measurements_ss, 
                                                    timestamp=start_time+timedelta(seconds=k))
                associations_ss = []
                for i in range(len(tracks_ss)):
                    track_hypotheses = hypotheses[tracks_ss[i]]
                    associations_ss_i = {'measurements': [], 'predictions': [], 'probabilities': []}
                    for h in track_hypotheses:
                        if h.measurement:
                            associations_ss_i['measurements'].append(h.measurement.state_vector.reshape(-1))
                            associations_ss_i['predictions'].append(h.measurement_prediction.state_vector.reshape(-1))
                            associations_ss_i['probabilities'].append(h.probability)
                        else:
                            associations_ss_i['measurements'].append(None)
                            associations_ss_i['predictions'].append(None)
                            associations_ss_i['probabilities'].append(h.probability)
                    associations_ss.append(associations_ss_i)

                if self.args.jpdaf:
                    tic = time.time()
                    online_targets = pkf.update_JPDAF(all_measurements[k], associations_ss) 
                    toc = time.time()
                    update_times.append(toc - tic)
                else:
                    tic = time.time()
                    online_targets, _ = pkf.update(all_measurements[k], associations_ss) 
                    toc = time.time()
                    update_times.append(toc - tic)

            results.append(online_targets)

            if not self.args.binary:
                ######### store the posterior states in stone soup #########
                if self.args.jpdaf:
                    for i in range(len(pkf.trackers_JPDAF)):
                        post_mean, post_covar = pkf.trackers_JPDAF[i].kf.x, pkf.trackers_JPDAF[i].kf.P
                        post_state = GaussianStateUpdate(post_mean, post_covar, 
                                                        hypotheses[tracks_ss[i]],
                                                        timestamp=start_time+timedelta(seconds=k))
                        tracks_ss[i].append(post_state)
                else:  
                    for i in range(len(pkf.trackers)):
                        post_mean, post_covar = pkf.trackers[i].kf.x, pkf.trackers[i].kf.P
                        post_state = GaussianStateUpdate(post_mean, post_covar, 
                                                        hypotheses[tracks_ss[i]],
                                                        timestamp=start_time+timedelta(seconds=k))
                        tracks_ss[i].append(post_state)

        # compute the distance between track and gt at each time step
        all_distances = [[] for _ in range(self.n_obj)]
        for i in range(self.n_obj):
            for k in range(nstep):
                gt_state = gt_states[i, k]
                track_state = results[k][i][:4]
                dist = np.linalg.norm(gt_state - track_state)
                    
                all_distances[i].append(dist)
        
        all_distances = np.array(all_distances)
        avg_distances = np.mean(all_distances, axis=1)
        print('avg_distances:', avg_distances)
        print('overall avg distance:', np.mean(avg_distances))

        avg_update_time = np.mean(update_times)
        print('average update time:', avg_update_time)
        fps = 1 / avg_update_time
        print('fps:', fps)

        ########## plot results ##########
        # estim_trajs = []
        # for i in range(self.n_obj):
        #     obj_i_traj = []
        #     for k in range(nstep):
        #         obj_i_traj.append(results[k][i])
        #     estim_trajs.append(obj_i_traj)

        # plt.figure()
        # for i in range(self.n_obj):
        #     obj_i_traj = np.array(estim_trajs[i])
        #     plt.plot(obj_i_traj[:, 0], obj_i_traj[:, 2], '--', label='robot %d' % i)
        
        # plt.legend()
        # plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("PKF simulation parameters")
    parser.add_argument("--dataset_root", type=str, default="./data")
    parser.add_argument("--n_obj", type=int, default=3)

    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--jpdaf", action="store_true")
    parser.add_argument("--noise_scale", type=float, default=0.75)
    args = parser.parse_args()


    evaluator = MOTEvaluator(args)
    evaluator.evaluate()

