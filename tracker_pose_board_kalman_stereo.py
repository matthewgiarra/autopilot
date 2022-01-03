import kalman
import drawing
import aruco
import depthai as dai
import cv2
import numpy as np
import sys
import time
from pdb import set_trace

np.set_printoptions(precision=2)
np.set_printoptions(floatmode = "fixed")

# Output video path
out_video_path = None

# Draw tracking? 
draw_tracking_status = False

# Draw corner numbers?
draw_corner_nums = False

# Draw edges of each tag?
draw_aruco_edges = True

# Draw the ID of each tag?
draw_aruco_ids = False

# Draw the cube pose?
draw_aruco_axes = True

# Which camera's coordinate system to report state in
master_camera = "mono_right"

# Which camera to use ("mono_left," "mono_right," or "color")
camera_type = "mono_right"

# For plotting
tracker_color = (0,255,255) # Line / text color
tracker_thickness = 4 # Line thickness
aruco_edge_color = (0,0,255) # Line / text color
aruco_edge_thickness = 4 # Line thickness

# Aruco stuff
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_CONTOUR

# Aruco tag size in meters
# Small cube uses 40 mm tags
# Big cube uses 80 mm tags
aruco_tag_size_meters = 0.040 # 2 inch cube
# aruco_tag_size_meters = 0.080 # 4 inch cube

# Length of each side of the aruco cube in meters.
# 2 inch = 0.0508 mm 
# 4 inch = 0.1016 mm
aruco_cube_size_meters = 0.0508 # 2 inch cube
# aruco_cube_size_meters = 0.1016 # 4 inch cube

# Create aruco board object (for the 4" cube)
# Order of tags in board_id should be: [front, right, back, left, top, bottom]
board_ids = np.array([0, 1, 2, 3, 5, 4]) # 2 inch cube. There is actually no tag 5; need it to specify cube 
# board_ids = np.array([5, 6, 7, 8, 9, 10]) # 4 inch cube

# Make the aruco cube
board = aruco.create_aruco_cube(board_ids = board_ids, aruco_dict = arucoDict, cube_width_m = aruco_cube_size_meters, tag_width_m = aruco_tag_size_meters)

# DepthAI Pipeline
pipeline = dai.Pipeline()

# Number of cameras
num_cams = 2
cam_fps = 120

# The reference coordinate system is cameraBoardSockets[0]
cameraBoardSockets = [dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT]

# Initialize lists of cameras and output links
cameras = []
xOutImages = []

# Set up pipeline
for i in range(num_cams):

    xOutImage = pipeline.create(dai.node.XLinkOut)
    xOutImage.input.setBlocking(False)
    xOutImage.input.setQueueSize(1)
    xOutImage.setStreamName("imageOut" + str(i))

    cam = pipeline.create(dai.node.MonoCamera)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    cam.setFps(cam_fps)
    cam.setBoardSocket(cameraBoardSockets[i])
    cam.out.link(xOutImage.input)

    cameras.append(cam)
    xOutImages.append(xOutImage)

# # # # # Kalman Filter set up # # # # # #

# Dimensionality of state space and measurements
dim_state = 12
dim_meas = 6
dt = 1 / cam.getFps()

# Instantiate the kalman filter object
kf = kalman.KalmanFilterPose(dim_state, dim_meas, 0, type=cv2.CV_64F)

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

# Covariance of process noise
Q = 5E-3 * np.eye(dim_state, dtype=np.float64)

# Covariance of measurement noise
R = 1E-2 * np.eye(dim_meas, dtype=np.float64)

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

# # # # # End kalman filter set up # # # #

# Connect and start pipeline
with dai.Device(pipeline) as device:

    # Get camera calibration info
    calibData = device.readCalibration()

    # Get camera extrinsics relative to first camera in list
    camera_extrinsics = []
    for i, cam in enumerate(cameras):
        if i > 0:
            extrinsicMatrix = np.array(calibData.getCameraExtrinsics(cam.getBoardSocket(), cameras[0].getBoardSocket()))
        else:
            extrinsicMatrix = np.eye(4)
        camera_extrinsics.append(extrinsicMatrix)

    # 3x3 matrix of [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # Get camera intrinsic matrices and distortion parameters
    camera_matrices = []
    camera_distortions = []
    for cam in cameras:
        resolution = cam.getResolutionSize()
        camera_matrix = np.array(calibData.getCameraIntrinsics(cam.getBoardSocket(), resizeWidth=resolution[0], resizeHeight=resolution[1]))
        camera_distortion = np.array(calibData.getDistortionCoefficients(cam.getBoardSocket()))[:-2]
        
        camera_matrices.append(camera_matrix)
        camera_distortions.append(camera_distortion)

    # Listen for data on the device xLinkOut queue
    image_out_queues = []
    for xOutImage in xOutImages:
        qImageOut = device.getOutputQueue(name = xOutImage.getStreamName(), maxSize=1, blocking=False)
        image_out_queues.append(qImageOut)

    prev_frame_time = 0
    new_frame_time = 0
    rmat_kalman_prev = np.eye(3)
    rvec_kalman = np.zeros(3)
    debug = False

    # Main processing loop
    while True:

        # Update kalman state predictions (calculate xk_km1)
        F[dt_idx] = dt
        kf.transitionMatrix = F
        kf.predict()

        # Update time step
        new_frame_time = time.time()
        dt = new_frame_time - prev_frame_time
        prev_frame_time = new_frame_time

        frames = []
        detectedCornersAllCams = []
        detectedIdsAllCams = []
        for i, queue in enumerate(image_out_queues):
            imgRaw = queue.get()
            frame = imgRaw.getCvFrame()

            # Detect corners and refine detections
            (detectedCorners, detectedIds, rejectedCorners) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
            detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = cv2.aruco.refineDetectedMarkers(frame, board, 
                detectedCorners, detectedIds, rejectedCorners, 
                cameraMatrix = camera_matrices[i], distCoeffs = camera_distortions[i], parameters=arucoParams)
            
            # Append items to lists
            frames.append(frame)
            detectedCornersAllCams.append(detectedCorners)
            detectedIdsAllCams.append(detectedIds)

        # Initialize translation vector
        tvec = np.array([0,0,0])

        ############# Temporarily grab parameters
        detectedCorners = detectedCornersAllCams[0]
        detectedIds = detectedIdsAllCams[0]
        camera_matrix = camera_matrices[0]
        camera_distortion = camera_distortions[0]
        frame = frames[0]

        # Make sure it's color
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Estimate the board pose if any tags were detected
        if len(detectedCorners) > 0:
            [ret, rvec, tvec, valid] = aruco.estimatePoseBoardAndValidate(detectedCorners, detectedIds, board, camera_matrix, camera_distortion, np.zeros(3), np.zeros(3))
            measurement = np.concatenate([tvec, rvec], axis=0).astype(np.float64)

            # Correct state prediction (calculate xk_k)
            # Only do this if the measurement is "good," i.e. z coordinate > 0
            if valid is True:
                kf.correct(measurement)

            # Draw IDs?
            if draw_aruco_ids is True:
                frame = drawing.draw_ids(frame, detectedCorners, detectedIds)
           
            if draw_aruco_edges is True:
                frame = drawing.draw_edges(frame, detectedCorners)

            if draw_corner_nums is True:
                frame = drawing.draw_corners(frame, detectedCorners)

            if draw_tracking_status is True:
                frame = drawing.draw_tracking(frame, detectedCorners)

        # Update the state of the cube
        tvec_kalman = kf.statePost[0:3]
        rvec_kalman = kf.statePost[3:6]
        
        rmat_kalman, _ = cv2.Rodrigues(rvec_kalman)
        residual_mat = np.matmul(np.transpose(rmat_kalman), rmat_kalman_prev)
        residual_vec, _ = cv2.Rodrigues(residual_mat)
        rnorm_kalman = np.linalg.norm(residual_vec) 

        # Draw FPS on frame
        frame = drawing.draw_fps(frame, 1/dt)

        # Copy the frame (we'll use the copied frame to visualize the Kalman-estimated board state)
        frame_kalman = frame.copy()

        # Draw the axes
        if draw_aruco_axes is True:
            frame_kalman = drawing.draw_pose(frame_kalman, camera_matrix, camera_distortion, rvec_kalman, tvec_kalman, aruco_tag_size_meters / 2)
            if len(detectedCorners) > 0:
                frame = drawing.draw_pose(frame, camera_matrix, camera_distortion, rvec, tvec, aruco_tag_size_meters / 2)

        # Display the image
        cv2.imshow("Image", frame)
        cv2.imshow("Kalman", frame_kalman)
        
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            debug = not debug
            if debug is True:
                print("Debugging ON")
            else:
                print("Debugging OFF")

        print("FPS: %0.2f" % (1/dt))

        # Update previous rotation matrix
        rmat_kalman_prev = rmat_kalman
        rvec_kalman_prev = rvec_kalman

        if (tvec[-1] < 0) and (debug is True):
            print("Warning: tvec[-1] < 0")
            set_trace()
        if (rnorm_kalman > 2) and (debug is True):
            print("Warning: norm(rvec_kalman) > threshold")
            set_trace()



    



