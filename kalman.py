import numpy as np
import cv2
import numbers
from pdb import set_trace
from numpy.core.numeric import True_


class KalmanFilterAttitude(cv2.KalmanFilter):
    # KalmanFilterPose extends cv2.KalmanFilter
    # by accounting for discontinuity in angles
    # i.e. for attitude

    # TODO: Make a different class called KalmanFilterAttitude 
    # that extends cv2.KalmanFilter that operates on attitude data alone
    # or have KalmanFilterPose class accept an input for which indicies
    # are the angle observations
    
    def correct(self, measurement):
        
        # Make sure the measurement is an n x 1 array
        ndims = len(measurement.shape)
        if ndims < 2:
            measurement = np.expand_dims(measurement, axis=1)

        # Measurement matrix
        Hk = self.measurementMatrix

        # Measurement noise covariance
        Rk = self.measurementNoiseCov

        # Transpose of measurement matrix
        HkT = np.transpose(Hk)

        # Pre-fit error covariance
        Pk_km1 = self.errorCovPre

        # Predicted state
        xk_km1 = self.statePre

        # Predicted measurement
        yk = measurement - np.matmul(Hk, xk_km1)

        # Length of the difference between 
        # measured and predicted Rodrigues vectors
        # Here we're assuming that a difference
        # greater than pi probably means 
        # the rotation axis flipped between observations,
        # i.e., the rotation angle wrapped around 
        if np.linalg.norm(yk) > np.pi:

            # Flip sign of state prediction to
            # make it consistent with the measurement
            xk_km1 = -1 * xk_km1

            # Recalculate measurement residual
            yk = measurement - np.matmul(Hk, xk_km1)

        # Pre-fit residual covariance
        Sk = np.matmul(H, np.matmul(Pk_km1, HkT)) + Rk

        # Inverse
        SkInv = np.linalg.inv(Sk)
        
        # Optimal Kalman gain
        Kk = np.matmul(Pk_km1, np.matmul(HkT, SkInv))

        # Updated state estimate
        xk_k = xk_km1 + np.matmul(Kk, yk)

        # Update error covariance
        I = np.eye(Pk_km1.shape[0])
        Pk_k = np.matmul((I - np.matmul(Kk, Hk)), Pk_km1)

        # Update the object's parameters
        self.statePost = xk_k
        self.errorCovPost = Pk_k
        self.gain = Kk

def kalmanFilter6DOFConstantVelocity(dt = 1/30, processNoise = 5E-3,  measurementNoise = 1E-2):

    # State transition matrix (2D constant velocity)
    F = np.array(
        [
        [ 1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0],
        [ 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0],
        [ 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0],
        [ 0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0],
        [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt],
        [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ]
    , dtype=np.float64)

    # Indices of state transition matrix containing dt
    dt_idx = np.where(F == dt)

    # Observation matrix
    H = np.array(
        [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        ], dtype = np.float64)

    # Dimensionality of state space and measurements
    dim_meas = H.shape[0]
    dim_state = H.shape[1]

    # Instantiate the kalman filter object
    kf = KalmanFilterPose(dim_state, dim_meas, 0, type=cv2.CV_64F)

    # Covariance of process noise
    Q = processNoise * np.eye(dim_state, dtype=np.float64)

    # Covariance of measurement noise
    if isinstance(measurementNoise, numbers.Number):
        R = measurementNoise * np.eye(dim_meas, dtype=np.float64)
    elif isinstance(measurementNoise, (list, np.ndarray)):
        R = np.diag(measurementNoise)
    else:
        raise ValueError("Error in kalmanFilter6DOFConstantVelocityMultiCam(): measurementNoise must be a scalar, list, or numpy array, but supplied type is %s" % str(type(measurementNoise)))

    # kf.statePre: xk_km1
    # kf.errorCovPre: Pk_km1
    # kf.statePost: xk_k
    # kf.errorCovPost: Pk_k

    kf.transitionMatrix = F
    kf.measurementMatrix = H
    kf.processNoiseCov = Q
    kf.measurementNoiseCov = R
    kf.statePost = np.zeros(dim_state, dtype=np.float64)
    kf.errorCovPost = 1E6 * np.eye(dim_state, dtype=np.float64)

    return kf, dt_idx

def kalmanFilter6DOFConstantVelocityMultiCam(dt = 1/30, processNoise = 5E-3,  measurementNoise = 1E-2):

    # State transition matrix (2D constant velocity)
    F = np.array(
        [
        [ 1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0],
        [ 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0],
        [ 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0],
        [ 0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0],
        [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt],
        [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ]
    , dtype=np.float64)

    # Indices of state transition matrix containing dt
    dt_idx = np.where(F == dt)

    # Observation matrix
    H = np.array(
        [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        ], dtype = np.float64)

    # Dimensionality of state space and measurements
    dim_meas = H.shape[0]
    dim_state = H.shape[1]

    # Instantiate the kalman filter object
    kf = KalmanFilterPoseMultiCam(dim_state, dim_meas, 0, type=cv2.CV_64F)

    # Covariance of process noise
    Q = processNoise * np.eye(dim_state, dtype=np.float64)

    # Covariance of measurement noise
    if isinstance(measurementNoise, numbers.Number):
        R = measurementNoise * np.eye(dim_meas, dtype=np.float64)
    elif isinstance(measurementNoise, (list, np.ndarray)):
        R = np.diag(measurementNoise)
    else:
        raise ValueError("Error in kalmanFilter6DOFConstantVelocityMultiCam(): measurementNoise must be a scalar, list, or numpy array, but supplied type is %s" % str(type(measurementNoise)))

    # kf.statePre: xk_km1
    # kf.errorCovPre: Pk_km1
    # kf.statePost: xk_k
    # kf.errorCovPost: Pk_k

    kf.transitionMatrix = F
    kf.measurementMatrix = H
    kf.processNoiseCov = Q
    kf.measurementNoiseCov = R
    kf.statePost = np.zeros(dim_state, dtype=np.float64)
    kf.errorCovPost = 1E6 * np.eye(dim_state, dtype=np.float64)

    return kf, dt_idx

class KalmanFilterPose(cv2.KalmanFilter):
    # KalmanFilterPose extends cv2.KalmanFilter
    # by accounting for discontinuity in angles
    # i.e. for attitude

    # TODO: Make a different class called KalmanFilterAttitude 
    # that extends cv2.KalmanFilter that operates on attitude data alone
    # or have KalmanFilterPose class accept an input for which indicies
    # are the angle observations
    
    # Correct measurement
    def correct(self, measurement):
        
        # Make sure the measurement is an n x 1 array
        ndims = len(measurement.shape)
        if ndims < 2:
            measurement = np.expand_dims(measurement, axis=1)

        # Measurement matrix
        Hk = self.measurementMatrix.copy()

        # Measurement noise covariance
        Rk = self.measurementNoiseCov

        # Pre-fit error covariance
        Pk_km1 = self.errorCovPre

        # Predicted state
        xk_km1 = self.statePre.copy()

        # Predicted measurement
        meas_pre = np.matmul(Hk, xk_km1)

        # Rotation
        rvec_pred = meas_pre[3:6]
        rvec_meas = measurement[3:6]
        
        # Length of the difference between 
        # measured and predicted Rodrigues vectors
        # Here we're assuming that a difference
        # greater than pi probably means 
        # the rotation axis flipped between observations,
        # i.e., the rotation angle wrapped around 
        norm_d = np.linalg.norm(rvec_meas - rvec_pred)
        if norm_d > np.pi:
            measurement[3:6] *= -1

        # Pre-fit measurement residual
        yk = measurement - np.matmul(Hk, xk_km1)
        
        # Pre-fit residual covariance
        Sk = np.matmul(Hk, np.matmul(Pk_km1, Hk.T)) + Rk

        # Inverse
        SkInv = np.linalg.inv(Sk)
        
        # Optimal Kalman gain
        Kk = np.matmul(Pk_km1, np.matmul(Hk.T, SkInv))

        # Updated state estimate
        xk_k = xk_km1 + np.matmul(Kk, yk)
        if norm_d > np.pi:
            xk_k[3:6] *= -1

        # Update error covariance
        I = np.eye(Pk_km1.shape[0])
        Pk_k = np.matmul((I - np.matmul(Kk, Hk)), Pk_km1)

        # Update the object's parameters
        self.statePost = xk_k
        self.errorCovPost = Pk_k
        self.gain = Kk

class KalmanFilterPoseMultiCam(cv2.KalmanFilter):
    # KalmanFilterPose extends cv2.KalmanFilter
    # by accounting for discontinuity in angles
    # i.e. for attitude
    
    def correct(self, measurement):
        
        degrees_of_freedom = 6

        # Make sure the measurement is an n x 1 array
        ndims = len(measurement.shape)
        if ndims < 2:
            measurement = np.expand_dims(measurement, axis=1)

        # Measurement matrix
        Hk = self.measurementMatrix

        # Measurement noise covariance
        Rk = self.measurementNoiseCov

        # Pre-fit error covariance
        Pk_km1 = self.errorCovPre

        # Predicted state
        xk_km1 = self.statePre

        # Predicted measurement
        meas_pred = np.matmul(Hk, xk_km1)

        # Number of measurements
        num_meas = int(Hk.shape[0] / degrees_of_freedom)

        numFlip = 0
        for i in range(num_meas):
            startRow = i * degrees_of_freedom + 3 # each 6dof measurement is [x,y,z,rx,ry,rz]
            endRow = startRow + 3
            rvec_pred = meas_pred[startRow:endRow]
            rvec_meas = measurement[startRow:endRow]
            norm_diff = np.linalg.norm(rvec_meas - rvec_pred)
            if norm_diff > np.pi:
                measurement[startRow:endRow] *= -1
                numFlip += 1

        # Pre-fit measurement residual
        yk = measurement - np.matmul(Hk, xk_km1)
        
        # Pre-fit residual covariance
        Sk = np.matmul(Hk, np.matmul(Pk_km1, Hk.T)) + Rk

        # Inverse
        SkInv = np.linalg.inv(Sk)
        
        # Optimal Kalman gain
        Kk = np.matmul(Pk_km1, np.matmul(Hk.T, SkInv))

        # Updated state estimate
        xk_k = xk_km1 + np.matmul(Kk, yk)

        # Determine whether to flip the rotation part of xk_k
        numNoFlip = num_meas - numFlip
        if numFlip >= numNoFlip:
            xk_k[3:6] *= -1

        # Update error covariance
        I = np.eye(Pk_km1.shape[0])
        Pk_k = np.matmul((I - np.matmul(Kk, Hk)), Pk_km1)

        # Update the object's parameters
        self.statePost = xk_k
        self.errorCovPost = Pk_k
        self.gain = Kk
