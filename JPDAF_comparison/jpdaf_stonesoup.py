import os
import numpy as np

from datetime import datetime, timedelta
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from stonesoup.plotter import Plotter
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import TrueDetection, Clutter

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater

from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.array import StateVectors
from stonesoup.functions import gm_reduce_single
from stonesoup.types.update import GaussianStateUpdate

import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import time
import argparse


def load_trajs(data_root, n_obj):
    gt_trajs_path = os.path.join(data_root, 'multi-object-data-%d' % (n_obj), 'gt_trajs.npy')
    gt_trajs = np.load(gt_trajs_path)
    assert gt_trajs.shape[0] == n_obj, 'gt_trajs.shape[0] != n_obj'

    gt_velos_path = os.path.join(data_root, 'multi-object-data-%d' % (n_obj), 'gt_velos.npy')
    gt_velos = np.load(gt_velos_path)
    assert gt_velos.shape[0] == n_obj, 'gt_velos.shape[0] != n_obj'
    assert gt_trajs.shape[1] == gt_velos.shape[1], 'gt_trajs.shape[1] != gt_velos.shape[1]'

    return gt_trajs, gt_velos


def generate_measurements(gt_trajs, gt_velos, start_time, measurement_model, prob_detect=0.9):

    ########### generate ground truth trajectories ###########
    nstep = gt_trajs.shape[1]

    truths = []
    for i in range(n_obj):
        p, v = gt_trajs[i, 0, :2, 3], gt_velos[i, 0, :2]
        truth = GroundTruthPath([GroundTruthState([[p[0]], [v[0]], [p[1]], [v[1]]], timestamp=start_time)])

        for k in range(1, nstep):
            p, v = gt_trajs[i, k, :2, 3], gt_velos[i, k, :2]
            # p = gt_trajs[i, k, :2, 3]
            next_k = min(k+1, nstep-1)
            v = gt_trajs[i, next_k, :2, 3] - p
            truth.append(GroundTruthState([[p[0]], [v[0]], [p[1]], [v[1]]],
                timestamp=start_time+timedelta(seconds=k)))
        truths.append(truth)

    ########### generate measurements ###########
    all_measurements = []

    for k in range(nstep):
        measurement_set = set()
        for truth in truths:
            # Generate actual detection from the state with a 10% chance that no detection is received.
            if np.random.rand() <= prob_detect:
                measurement = measurement_model.function(truth[k], noise=True)
                measurement_set.add(TrueDetection(state_vector=measurement,
                                                groundtruth_path=truth,
                                                timestamp=truth[k].timestamp,
                                                measurement_model=measurement_model))

            # Generate clutter at this time-step
            truth_x = truth[k].state_vector[0]
            truth_y = truth[k].state_vector[2]
            if n_obj <= 5:
                avg_n_clutters = 2
                c_range = 10
                for _ in range(np.random.randint(avg_n_clutters)):
                    x = np.random.uniform(truth_x - c_range, truth_x + c_range)
                    y = np.random.uniform(truth_y - c_range, truth_y + c_range)
                    measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=truth[k].timestamp,
                                                measurement_model=measurement_model))
            else:
                avg_n_clutters = 2
                c_range = 10
                for _ in range(np.random.randint(avg_n_clutters)):
                    x = np.random.uniform(truth_x - c_range, truth_x + c_range)
                    y = np.random.uniform(truth_y - c_range, truth_y + c_range)
                    measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=truth[k].timestamp,
                                                measurement_model=measurement_model))
            
        all_measurements.append(measurement_set)

    return all_measurements, truths


def load_measurements(measurement_fname):
    with open(measurement_fname, 'rb') as f:
        data = pickle.load(f)
    
    start_time = data['start_time']
    truths = data['truths']
    all_measurements = data['measurements_raw']

    return start_time, truths, all_measurements


if __name__ == "__main__":
    
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--n_obj', type=int, default=3)
    parser.add_argument('--noise_scale', type=float, default=0.75)
    parser.add_argument('--visualize', action='store_true', help='visualize the tracking results')
    args = parser.parse_args()

    data_root = args.data_root
    n_obj = args.n_obj
    noise_scale = args.noise_scale

    np.random.seed(1991)
    visu_root = './jpdaf_visu'
    visu_path = os.path.join(visu_root, 'multi-object-data-%d' % (n_obj))
    os.makedirs(visu_path, exist_ok=True)

    data_saveroot = './data'
    data_savepath = os.path.join(data_saveroot, 'multi-object-data-%d' % (n_obj))
    os.makedirs(data_savepath, exist_ok=True)

    load_saved_measurements = True
    if n_obj <= 5:
        measurement_fname = os.path.join(data_root, 'multi-object-data-%d' % (n_obj), 'data.pkl')
    else:
        measurement_fname = os.path.join(data_root, 'multi-object-data-%d' % (n_obj), 'data_noise_%.2f.pkl' % noise_scale)
    print('load saved measurements from:', measurement_fname)

    ########### get measurements ###########
    # Load ground truth trajectories
    gt_trajs, gt_velos = load_trajs(data_root, n_obj)
    init_state_cov = np.diag([1.5, 0.5, 1.5, 0.5])
    
    if n_obj <= 5:
        measurement_model = LinearGaussian(
            ndim_state=4,
            mapping=(0, 2),
            noise_covar=np.array([[0.75, 0],
                                [0, 0.75]])
            )
        prob_detect = 0.9  # 90% chance of detection.
    else:
        measurement_model = LinearGaussian(
            ndim_state=4,
            mapping=(0, 2),
            noise_covar=np.eye(2) * noise_scale
            )
        prob_detect = 0.95  # 90% chance of detection for ten objects.
    
    if load_saved_measurements:
        gt_trajs, gt_velos = load_trajs(data_root, n_obj)

        start_time, truths, all_measurements = load_measurements(measurement_fname)
    else:
        start_time = datetime.now()

        all_measurements, truths = generate_measurements(gt_trajs, gt_velos, start_time, 
                                                         measurement_model, prob_detect)

    if args.visualize:
        # Plot ground truth.
        plotter = Plotter()
        plotter.ax.set_ylim(-20, 20)
        plotter.plot_ground_truths(truths, [0, 2])
        plotter.fig.savefig(os.path.join(visu_path, 'ground_truths.png'))

        # Plot true detections and clutter.
        plotter.plot_measurements(all_measurements, [0, 2], color='g')
        plotter.fig.savefig(os.path.join(visu_path, 'measurements.png'))
    

    ########### JPDAF ###########
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])

    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    hypothesiser = PDAHypothesiser(predictor=predictor,
                                   updater=updater,
                                   clutter_spatial_density=0.125,
                                   prob_detect=prob_detect)

    data_associator = JPDA(hypothesiser=hypothesiser)

    # Create prior Gaussian state for tracks
    # tracks = set()
    tracks = []
    for i in range(n_obj):
        p, v = gt_trajs[i, 0, :2, 3], gt_velos[i, 0, :2]
        # prior = GaussianState([[p[0]], [v[0]], [p[1]], [v[1]]], init_state_cov, timestamp=start_time)
        prior = GaussianState([[p[0]], [0], [p[1]], [0]], init_state_cov, timestamp=start_time)
        track = Track([prior], id=f'object-{i}')
        # tracks.add(track)
        tracks.append(track)

    # all_measurements_np = [] # save measurements with numpy
    all_tracks_np = [[] for _ in range(n_obj)] # save tracks with numpy
    all_associations = [[] for _ in range(n_obj)] # save associations
    
    all_associations_raw = [] # save associations
    update_times = [[] for _ in range(n_obj)]
    all_measurements_np = []

    print('start tracking')
    for n, measurements in tqdm(enumerate(all_measurements)):
        
        hypotheses = data_associator.associate(tracks, measurements,
                                                start_time + timedelta(seconds=n))
        all_associations_raw.append(hypotheses)

        measurement_np = []
        for m in measurements:
            # print('measurement:', m.state_vector.shape)
            measurement_np.append(m.state_vector.reshape(-1))
        all_measurements_np.append(np.array(measurement_np))

        # Loop through each track, performing the association step with weights adjusted according to 
        # JPDA.
        for i in range(len(tracks)):
            track = tracks[i]
            for track_key in hypotheses.keys():
                if track_key.id == track.id:
                    break
            track_hypotheses = hypotheses[track_key]
            
            ############## Save all associations for analysis ##############
            track_associations = {'measurements': [], 'predictions': [], 'probabilities': []}
            total_prob = 0
            for h in track_hypotheses:
                total_prob += h.probability

                if h.measurement:
                    track_associations['measurements'].append(h.measurement.state_vector.reshape(-1))
                else:
                    track_associations['measurements'].append(None)
                    
                    if h.measurement_prediction:
                        print('measurement prediction is None while measurement is not None')

                if h.measurement_prediction:
                    track_associations['predictions'].append(h.measurement_prediction.state_vector.reshape(-1)) 
                else:
                    track_associations['predictions'].append(None)
                    if h.measurement.state_vector is not None:
                        print('measurement is not None while prediction is None')
                
                track_associations['probabilities'].append(h.probability)
            
            if total_prob < 0.99:
                print('total_prob less than 1:', total_prob)
            if total_prob > 1.01:
                print('total_prob greater than 1:', total_prob)

            all_associations[i].append(track_associations)
            ############### End of saving associations ###############

            tic = time.time()            
            posterior_states = []
            posterior_state_weights = []

            ############### Consider the none hypothesis ###############
            # for hypothesis in track_hypotheses:
            #     if not hypothesis:
            #         posterior_states.append(hypothesis.prediction)
            #     else:
            #         posterior_state = updater.update(hypothesis)

            #         posterior_states.append(posterior_state)
            #     posterior_state_weights.append(hypothesis.probability)


            ############ Do not consider the none hypothesis ############
            det_cnt = 0
            for hypothesis in track_hypotheses:
                if hypothesis:
                    det_cnt += 1
                    posterior_states.append(updater.update(hypothesis))
                    posterior_state_weights.append(hypothesis.probability)
            if det_cnt == 0:
                posterior_states.append(track_hypotheses[0].prediction)
                posterior_state_weights.append(1.0)

            ###########################################################

            means = StateVectors([state.state_vector for state in posterior_states])
            covars = np.stack([state.covar for state in posterior_states], axis=2)
            weights = np.asarray(posterior_state_weights)

            # Reduce mixture of states to one posterior estimate Gaussian.
            post_mean, post_covar = gm_reduce_single(means, covars, weights)

            # Add a Gaussian state approximation to the track.
            track.append(GaussianStateUpdate(
                post_mean, post_covar,
                track_hypotheses,
                track_hypotheses[0].measurement.timestamp))

            toc = time.time()
            update_times[i].append(toc - tic)
        
        for i in range(n_obj):
            all_tracks_np[i].append(tracks[i].states[-1].state_vector)

    if args.visualize:
        plotter.plot_tracks(tracks, [0, 2], uncertainty=True)
        plotter.fig.savefig(os.path.join(visu_path, 'tracks.png'))

    # compute the distance between track and gt at each time step
    all_distances = [[] for _ in range(n_obj)]
    for i in range(n_obj):
        tracks_i, truths_i = tracks[i], truths[i]
        for track, truth in zip(tracks_i, truths_i):
            all_distances[i].append(np.linalg.norm(track.state_vector[::2] - truth.state_vector[::2]))
    
    all_distances = np.array(all_distances)
    avg_distances = np.mean(all_distances, axis=1)
    print('avg_distances:', avg_distances)
    print('overall avg distance:', np.mean(avg_distances))

    update_times = np.array(update_times)
    avg_update_times = np.sum(update_times, axis=0)
    # print('avg_update_times:', avg_update_times)
    print('overall avg update time:', avg_update_times.mean())
    print('fps:', 1/avg_update_times.mean())

