import numpy as np


class PMHT(object):

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

        # self.history_obs = []
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
            return
        # z: list of measurements; w: list of weights
        assert len(z) == len(w), 'z and w must have the same length'

        # m = len(z) # number of measurements

        z_tilde = np.sum(z * w[:, None], axis=0)
        z_tilde /= np.sum(w) + 1e-16  # avoid division by zero
        z_tilde = z_tilde.reshape(-1, 1)

        if V is None:
            V = self.V / (np.sum(w) + 1e-16)  # avoid division by zero
        
        if H is None:
            H = self.H
        
        K = self.P @ H.T @ self.inv(H @ self.P @ H.T + V)

        self.x = self.x + K @ (z_tilde - H @ self.x)
        self.P = self.P - K @ H @ self.P

