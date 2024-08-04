import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, dt=1.0):
        # Define state vector [x, y, vx, vy]
        self.x = np.zeros((4, 1), dtype=np.float32)
        
        # Time interval
        self.dt = dt

        # State transition matrix (will be updated in predict)
        self.F = np.eye(4, dtype=np.float32)
        
        # Process noise covariance matrix
        self.Q = np.eye(4, dtype=np.float32) * 0.03

        # Measurement matrix (linearized version)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * 0.3

        # Error covariance matrix
        self.P = np.eye(4, dtype=np.float32)

    def f(self, x):
        # Nonlinear state transition function
        dt = self.dt
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        return np.dot(F, x)

    def h(self, x):
        # Measurement function (assumes direct measurement of position)
        return np.dot(self.H, x)

    def jacobian_F(self, x):
        # Jacobian of the state transition function
        dt = self.dt
        return np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)

    def jacobian_H(self, x):
        # Jacobian of the measurement function (constant in this case)
        return self.H

    def predict(self):
        # Predict the next state
        self.F = self.jacobian_F(self.x)
        self.x = self.f(self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2]

    def update(self, z):
        # Calculate the Kalman gain
        H = self.jacobian_H(self.x)
        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # Update the state vector
        y = z - self.h(self.x)  # Measurement residual
        self.x = self.x + np.dot(K, y)

        # Update the error covariance matrix
        I = np.eye(4, dtype=np.float32)
        self.P = np.dot((I - np.dot(K, H)), self.P)

        return self.x[:2]
