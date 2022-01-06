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

# # Aruco tag size in meters
# # Small cube uses 40 mm tags
# # Big cube uses 80 mm tags
# aruco_tag_size_meters = 0.040 # 2 inch cube
aruco_tag_size_meters = 0.080 # 4 inch cube

# # Length of each side of the aruco cube in meters.
# # 2 inch = 0.0508 m 
# # 4 inch = 0.1016 m
# aruco_cube_size_meters = 0.0508 # 2 inch cube
aruco_cube_size_meters = 0.1016 # 4 inch cube

# Create aruco board object (for the 4" cube)
# Order of tags in board_id should be: [front, right, back, left, top, bottom]
# board_ids = np.array([0, 1, 2, 3, 5, 4]) # 2 inch cube. There is actually no tag 5; need it to specify cube 
board_ids = np.array([5, 6, 7, 8, 9, 10]) # 4 inch cube

# Make the aruco cube
board = aruco.create_aruco_cube(board_ids = board_ids, aruco_dict = arucoDict, cube_width_m = aruco_cube_size_meters, tag_width_m = aruco_tag_size_meters)

# DepthAI Pipeline
pipeline = dai.Pipeline()

# Number of cameras
cam_fps = 120
video_fps = 30

# The reference coordinate system is cameraBoardSockets[0]
cameraBoardSockets = [dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT]

# Initialize lists of cameras and output links
cameras = []
xOutImages = []

# Set up pipeline
for i in range(len(cameraBoardSockets)):

    xOutImage = pipeline.create(dai.node.XLinkOut)
    xOutImage.input.setBlocking(False)
    xOutImage.input.setQueueSize(1)
    xOutImage.setStreamName("imageOut" + str(i))

    cam = pipeline.create(dai.node.MonoCamera)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    cam.setFps(cam_fps)
    cam.setBoardSocket(cameraBoardSockets[i])
    cam.out.link(xOutImage.input)

    cameras.append(cam)
    xOutImages.append(xOutImage)

# Connect and start pipeline
with dai.Device(pipeline) as device:

    # Get camera calibration info
    calibData = device.readCalibration()

    # Get camera extrinsics relative to first camera in list
    camera_extrinsics = []
    for i, cam in enumerate(cameras):
        if i > 0:
            extrinsicMatrix = np.array(calibData.getCameraExtrinsics(cameras[0].getBoardSocket(), cam.getBoardSocket()))

            # calibData outputs the translation part of the extrinsic matrix in centimeters.
            # In this house we work in meters!
            extrinsicMatrix[0:3, 3] /= 100
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

    # Open a video writer object
    if out_video_path is not None:
        video_resolution = (2 * resolution[0], resolution[1])
        outVideo = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, video_resolution)
    else:
        outVideo = None
    
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

    # Set up the kalman filter
    dt = 1 / cam.getFps() # Time step
    kf, dt_idx = kalman.kalmanFilter6DOFConstantVelocityMultiCam(dt = dt)
    H_ref = kf.measurementMatrix
    R_ref_diag = np.diag(kf.measurementNoiseCov)

    # Main processing loop
    while True:

        # Update kalman state predictions (calculate xk_km1)
        kf.transitionMatrix[dt_idx] = dt
        kf.predict()

        # Update time step
        new_frame_time = time.time()
        dt = new_frame_time - prev_frame_time
        prev_frame_time = new_frame_time

        # Initialize vectors for frames (images) and pose measurements
        frames = []
        measurements = []

        # Get all the frames before we do any processing so that they're closely spaced in time
        for i, queue in enumerate(image_out_queues):
            frame = queue.get().getCvFrame()
            frames.append(frame)

        # Process each frame
        for i, frame in enumerate(frames):
            cameraMatrix = camera_matrices[i]
            distCoeffs    = camera_distortions[i]

            valid, measurement, detectedCorners, detectedIds = aruco.getPoseMeasurement(frame, board, 
            cameraMatrix=cameraMatrix, distCoeffs = distCoeffs, arucoDict = arucoDict, parameters=arucoParams)

            # Transform measurement to reference camera's coordinate system
            if valid is True:
                
                # Transform the measurement into the reference coordinate system (cameras[0])
                measurement = aruco.transformMeasurement(measurement, np.linalg.inv(camera_extrinsics[i]))

                # If the two rvecs differ by >= pi then one of them has flipped.
                # The actual angle between the vectors is smaller than the norm of their differences.
                # Flip the new measurement to point in the same direction as the reference measurement
                if len(measurements) > 0:
                    measurement = aruco.alignPose(measurement, measurements[0])

                # Make the measurements array              
                measurements.append(measurement)

            # Make the image 3 channel color for plotting
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Estimate the board pose if any tags were detected
            if len(detectedCorners) > 0:
                
                # Draw IDs?
                if draw_aruco_ids is True:
                    frame = drawing.draw_ids(frame, detectedCorners, detectedIds)
            
                if draw_aruco_edges is True:
                    frame = drawing.draw_edges(frame, detectedCorners)

                if draw_corner_nums is True:
                    frame = drawing.draw_corners(frame, detectedCorners)

                if draw_tracking_status is True:
                    frame = drawing.draw_tracking(frame, detectedCorners)

            # Draw FPS on frame
            frame = drawing.draw_fps(frame, 1/dt)

            # Upate frames vector
            frames[i] = frame
        
        # Update the kalman filter's measurement matrix according to 
        # the number of valid measurements that were recorded.
        if len(measurements) > 0:

            # Set the measurement matrix and meas. noise matrix for the number of valid measurements
            H = np.concatenate([H_ref for x in measurements], axis=0)
            R = np.diag(np.concatenate([R_ref_diag for x in measurements], axis=0))    
            measurement = np.concatenate(measurements)

            # Set kalman filter parameters for this iteration
            kf.measurementMatrix = H
            kf.measurementNoiseCov = R

            # Correct state prediction (calculate xk_k)
            kf.correct(measurement)

        # Read the pose from the kf state vector
        tvec = kf.statePost[0:3]
        rvec = kf.statePost[3:6]

        # Draw the axes on the rames
        for i, frame in enumerate(frames):
            pose_mat = np.eye(4)
            pose_mat[0:3, 0:3], _ = cv2.Rodrigues(rvec)
            pose_mat[0:3, 3] = np.transpose(tvec)
            pose_mat_transformed = np.matmul(camera_extrinsics[i], pose_mat)
            pose_mat_transformed = pose_mat_transformed / pose_mat_transformed[3,3]

            rvec_transformed, _ = cv2.Rodrigues(pose_mat_transformed[0:3, 0:3])
            tvec_transformed = pose_mat_transformed[0:3, 3]
            
            if draw_aruco_axes is True:
                frame = drawing.draw_pose(frame, camera_matrices[i], camera_distortions[i], rvec_transformed, tvec_transformed, aruco_tag_size_meters / 2)

            # Display the image
            frame_title = str(cameraBoardSockets[i])
            cv2.imshow(frame_title, frame)

        # Write the video frame
        if outVideo is not None:
            frames_cat = np.concatenate(frames, axis=1)
            outVideo.write(frames_cat)

        # Print pose parameters to console
        print(np.squeeze(np.concatenate([tvec, rvec], axis=0).astype(np.float64)))
        
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            debug = not debug
            if debug is True:
                print("Debugging ON")
            else:
                print("Debugging OFF")

# Close the video
if out_video_path is not None:
    outVideo.release()

# GTFO
print("Exiting")


    



