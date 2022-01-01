import numpy as np
import cv2

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

class KalmanFilterPose(cv2.KalmanFilter):
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
            xk_km1[3:6] = -1 * xk_km1[3:6]
            xk_km1[9:]  = -1 * xk_km1[9:]

        # Pre-fit measurement residual
        yk = measurement - np.matmul(Hk, xk_km1)
        
        # Pre-fit residual covariance
        Sk = np.matmul(Hk, np.matmul(Pk_km1, HkT)) + Rk

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
