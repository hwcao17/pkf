import numpy as np
from copy import deepcopy


class PAKalmanFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = np.zeros((dim_x, 1))        # state
        self.P = np.eye(dim_x)               # state covariance
        
        self.F = np.eye(dim_x)               # state transition matrix
        self.G = None                        # control transition matrix
        self.W = np.eye(dim_x)               # motion uncertainty

        self.H = np.zeros((dim_z, dim_x))    # measurement matrix
        self.V = np.eye(dim_z)               # measurement uncertainty

        self.history_obs = []
        self.observed = False
        self.freezed = False
        self.attr_saved = None

        self.inv = np.linalg.inv

    def predict(self, u=None, G=None, F=None, W=None):
        if G is None:
            G = self.G
        if F is None:
            F = self.F
        if W is None:
            W = self.W
        elif np.isscalar(W):
            W = np.eye(self.dim_x) * W

        # x = Fx + Gu
        if G is not None and u is not None:
            self.x = F @ self.x + G @ u
        else:
            self.x = F @ self.x

        # P = FPF' + W
        self.P = F @ self.P @ F.T + W

    def update(self, z, w, V=None, H=None):
    
        if z is None:
            self.history_obs.append(None)
            return
        
        dominant_idx = np.argmax(w)
        dominant_z = z[dominant_idx]
        
        # append the observation
        self.history_obs.append(dominant_z)

        # z: list of measurements; w: list of weights
        assert len(z) == len(w), 'z and w must have the same length'

        m = len(z) # number of measurements

        z_exp = np.array(z).reshape(-1, 1)
        
        # create expanded V matrices
        if V is None:
            V_exp = np.kron(np.eye(m), self.V)
            
        elif np.isscalar(V):
            V_exp = np.kron(np.eye(m), np.eye(self.dim_z) * V)

        for i in range(m):
            V_exp[i*self.dim_z:(i+1)*self.dim_z, i*self.dim_z:(i+1)*self.dim_z] = self.V / w[i]
        
        if np.linalg.det(V_exp) == 0:
            print('singular V_exp\n', 'w:\n', w)

        # create expanded H matrices
        if H is None:
            H_exp = np.kron(np.ones((m, 1)), self.H)
        else:
            H_exp = np.kron(np.ones((m, 1)), H)
        
        # Kalman gain
        K = self.P @ H_exp.T @ self.inv(H_exp @ self.P @ H_exp.T + V_exp)

        # measurement residual
        y = z_exp - H_exp @ self.x
        
        # update
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ H_exp) @ self.P

