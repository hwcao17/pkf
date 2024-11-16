import os
import random
import numpy as np
import os.path as osp
from argparse import Namespace
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import TrueDetection, Clutter
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
import pickle
import argparse

# add arguments
parser = argparse.ArgumentParser(description='Generate data for multi-object tracking')
parser.add_argument('--n_obj', type=int, default=3, help='Number of curves to generate')
parser.add_argument('--noise_scale', type=float, default=0.75, help='Noise scale for the measurement model')
args = parser.parse_args()


def generate_config():
    C = Namespace

    C.seed = 0
    C.generate_3d = False
    C.rotate_xy = False
    C.num_objects = args.n_obj

    C.MM = 420 # Number of time steps for the eight-shaped path (preferably above 40 steps)
    C.half_path = 10 # in meters
    # covariance for relative poses' positions and rotations respectively:
    C.sigma_pose_perturbation = 0.015
    C.sigma_rotation_perturbation = 0.01
    # Covariance of the prior features positions for initialization if the triangulation is not used (in meters)
    C.prior_sigma = 0.7

    # Path where generated data is to be saved.
    C.save_path = osp.join((osp.dirname(__file__) or "."), 'data', 'multi-object-data-%d' % C.num_objects)
    # print('file dir:', osp.dirname(__file__))
    print('save path:', C.save_path)

    if not osp.exists(C.save_path):
        os.makedirs(C.save_path)

    return C

CONFIG = generate_config()

def batch_matmul(A, B):
    assert A.shape[-1] == B.shape[-2]
    return (A[..., :, :, None] * B[..., None, :, :]).sum(axis=-2)


class Lissajous:
    """
    Plots Lissajous curve

    x = A sin(at + δ)
    y = B sin(bt)

    Optional arguments
    """
    def __init__(s,
                 A=CONFIG.half_path,
                 B=CONFIG.half_path,
                 a=1,
                 b=2,
                 δ=np.pi/2):
        s.A = A
        s.B = B
        s.a = a
        s.b = b
        s.δ = δ

    def generate_ground_truth_trajectory(s, object_num):
        A = s.A
        B = s.B
        a = s.a
        b = s.b
        δ = s.δ

        t = np.linspace(-np.pi, np.pi, CONFIG.MM)
        x = A * np.sin(a*t + δ)
        y = B * np.sin(b*t)
        z = np.zeros_like(t)

        θ = np.arctan2(np.roll(y, -1) - y, np.roll(x, -1) -x).reshape(-1, 1, 1)
        n = θ.shape[0]
        θ = θ - np.pi/2 # Hack to make robots align with the trajectory
        zs = np.zeros((n, 1, 1))
        os = np.ones((n, 1, 1))
        Rθ = np.concatenate([
            np.concatenate([np.cos(θ), -np.sin(θ), zs], axis=-1),
            np.concatenate([np.sin(θ),  np.cos(θ), zs], axis=-1),
            np.concatenate([       zs,         zs, os], axis=-1),
            ], axis=-2)
        #assert is_valid_so3(Rθ)
        poses_gt = np.zeros((t.shape[0], 4, 4))
        poses_gt[:, :3, :3] = Rθ
        poses_gt[:, 0, 3] = x
        poses_gt[:, 1, 3] = y
        poses_gt[:, 2, 3] = z
        poses_gt[:, 3, 3] = 1

        # velocities
        vx = A * a * np.cos(a*t + δ)
        vy = B * b * np.cos(b*t)
        vz = np.zeros_like(t)
        velocities_gt = np.stack([vx, vy, vz], axis=-1)

        curve_num = object_num
        robot_num_on_curve = object_num
        
        if CONFIG.generate_3d:
            
            curve_theta = 2*np.pi*curve_num/CONFIG.num_objects
            curve_rot = np.array([
                [np.cos(curve_theta), - np.sin(curve_theta), 0],
                [np.sin(curve_theta),   np.cos(curve_theta), 0],
                [                  0,                     0, 1]])

            if CONFIG.rotate_xy:
                roll = random.uniform(-np.pi/2, np.pi/2)
                curve_rot_roll = np.array([[1, 0, 0],
                                        [0, np.cos(roll), -np.sin(roll)],
                                        [0, np.sin(roll), np.cos(roll)]])
                pitch = random.uniform(-np.pi/2, np.pi/2)
                curve_rot_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                                            [0, 1, 0],
                                            [-np.sin(pitch), 0, np.cos(pitch)]])

                curve_rot = curve_rot_roll @ curve_rot_pitch @ curve_rot

            curve_shift = curve_rot @ np.array([A*0.5, 0, 0])
            curve_trans = np.vstack([np.hstack([curve_rot,  curve_shift.reshape(-1, 1)]),
                                    np.array([[0, 0, 0, 1]])])
        else:
            curve_theta = 2*np.pi*curve_num/CONFIG.num_objects
            curve_rot = np.array([
                [np.cos(curve_theta), - np.sin(curve_theta), 0],
                [np.sin(curve_theta),   np.cos(curve_theta), 0],
                [                  0,                     0, 1]])
            
            curve_shift = curve_rot @ np.array([A*0.5, 0, 0])
            curve_trans = np.vstack([np.hstack([curve_rot,  curve_shift.reshape(-1, 1)]),
                                    np.array([[0, 0, 0, 1]])])            
        
        poses_gt_curve = batch_matmul(curve_trans, poses_gt)
        velocities_gt_curve = (curve_trans[:3, :3] @ velocities_gt.T).T + curve_trans[None, :3, 3]
        
        robot_shift = robot_num_on_curve * len(t)
        poses_gt_robot = np.roll(poses_gt_curve, robot_shift, axis=0)
        velocities_gt_robot = np.roll(velocities_gt_curve, robot_shift, axis=0)
        
        return poses_gt_robot, velocities_gt_robot

    def generate_ground_truth_trajectories(self):

        poses_gt, velocities_gt = list(), list()
        for r in range(CONFIG.num_objects):
            pose_gt_per_robot, velo_gt_per_robot = self.generate_ground_truth_trajectory(r)
            poses_gt.append(pose_gt_per_robot[None, ...])
            velocities_gt.append(velo_gt_per_robot[None, ...])

        return np.concatenate(poses_gt, axis=0), np.concatenate(velocities_gt, axis=0)


def plot_gt_trajs(gt_trajs):
    n_obj, nstep = gt_trajs.shape[0], gt_trajs.shape[1]

    for i in range(n_obj):
        x = gt_trajs[i, :, 0, 3]
        y = gt_trajs[i, :, 1, 3]
        plt.plot(x, y)

    plt.show()

def generate_measurements(gt_trajs, gt_velos, start_time, measurement_model, prob_detect=0.9):

    ########### generate ground truth trajectories ###########
    n_obj = gt_trajs.shape[0]
    nstep = gt_trajs.shape[1]

    truths = []
    for i in range(n_obj):
        p, v = gt_trajs[i, 0, :2, 3], gt_velos[i, 0, :2]
        truth = GroundTruthPath([GroundTruthState([[p[0]], [v[0]], [p[1]], [v[1]]], timestamp=start_time)])

        for k in range(1, nstep):
            p, v = gt_trajs[i, k, :2, 3], gt_velos[i, k, :2]
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


if __name__ == "__main__":

    np.random.seed(1991)

    gt_trajs, gt_velos = Lissajous().generate_ground_truth_trajectories()
    print('gt_trajs:', gt_trajs.shape, 'gt_velos:', gt_velos.shape)
    plot_gt_trajs(gt_trajs)

    # save the ground truth trajectories
    os.path.makedirs(CONFIG.save_path, exist_ok=True)
    np.save(osp.join(CONFIG.save_path, 'gt_trajs.npy'), gt_trajs)
    np.save(osp.join(CONFIG.save_path, 'gt_velos.npy'), gt_velos)
    
    # generate measurements
    n_obj = CONFIG.num_objects
    noise_scale = args.noise_scale

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
        prob_detect = 0.95  # 95% chance of detection for 10 objects.

    start_time = datetime.now()

    all_measurements, truths = generate_measurements(gt_trajs, gt_velos, start_time, 
                                                     measurement_model, prob_detect)
    
    all_measurements_np = []

    for k in range(len(all_measurements)):
        measurements_np = []
        for m in all_measurements[k]:
            measurements_np.append(m.state_vector)
        all_measurements_np.append(np.array(measurements_np))

    data_to_save = {'start_time': start_time, 
                    'truths': truths,
                    'measurements_raw': all_measurements,
                    'measurements': all_measurements_np}

    if n_obj <= 5:
        data_fname = osp.join(CONFIG.save_path, 'data.pkl')
    else:
        data_fname = osp.join(CONFIG.save_path, 'data_noise_%.2f.pkl' % noise_scale)
    
    with open(data_fname, 'wb') as f:
        pickle.dump(data_to_save, f)
    

