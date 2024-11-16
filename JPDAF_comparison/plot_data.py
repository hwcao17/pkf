import os
import pickle
import numpy as np
from stonesoup.types.detection import TrueDetection, Clutter
import matplotlib.pyplot as plt


def load_measurements(data_fname):
        
    with open(data_fname, 'rb') as f:
        data = pickle.load(f)
    
    start_time = data['start_time']
    all_measurements_raw = data['measurements_raw']
    all_measurements = data['measurements']
    for i in range(len(all_measurements)):
        all_measurements[i] = np.array(all_measurements[i])
    

    return start_time, all_measurements_raw, all_measurements


if __name__ == "__main__":

    data_root = './data'
    
    n_obj = 3
    data_path = os.path.join(data_root, 'multi-object-data-%d' % n_obj)

    # load gt trajectories
    gt_trajs_fname = os.path.join(data_path, 'gt_trajs.npy')
    gt_trajs = np.load(gt_trajs_fname, allow_pickle=True)
    gt_trajs = gt_trajs[:, :, :3, 3]
    print('gt_trajs:', gt_trajs.shape)
    
    # load detections and clutters
    data_fname = os.path.join(data_path, 'data.pkl')
    start_time, all_measurements_ss, all_measurements = load_measurements(data_fname)
    
    measurements, clutters = [], []
    for measurement_set in all_measurements_ss:
        for meas in measurement_set:

            if isinstance(meas, TrueDetection):
                measurements.append(meas.state_vector)
            elif isinstance(meas, Clutter):
                clutters.append(meas.state_vector)
    
    measurements = np.array(measurements)
    clutters = np.array(clutters)

    print('num of measurements:', len(measurements))
    print('num of clutters:', len(clutters))

    # plot gt trajectories
    plt.figure()
    obj_colors = ['r', 'b', 'm']
    for i in range(n_obj):
        plt.plot(gt_trajs[i, :, 0], gt_trajs[i, :, 1], '--', c=obj_colors[i], linewidth=6, \
                 label='Object-%d' % i)

    # plot measurements
    plt.scatter(measurements[:, 0], measurements[:, 1], s=128, c='g', marker='o', \
                    label='Measurement')

    # plot clutters
    plt.scatter(clutters[:, 0], clutters[:, 1], s=128, c=[[0.9, 0.9, 0]], \
                marker='x', label='Clutter')

    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('X', fontsize=20)
    plt.ylabel('Y', fontsize=20)
    # plt.savefig('jpdaf_data.png')
    plt.show()
