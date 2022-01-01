import kalman
import depthai as dai
import cv2
import numpy as np
import sys
import time
from pdb import set_trace

np.set_printoptions(precision=2)
np.set_printoptions(floatmode = "fixed")

def create_aruco_cube(cube_width_m = 0.0508, tag_width_m = 0.040, board_ids = [0,1,2,3,4,5], aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)):
    # Create an "aruco cube" using the aruco board class
    # Order of tags in board_id should be: [front, right, back, left, top, bottom]
    # These face names are specified as though you are looking at the front side.
    # Axes: With the "front" side facing you,
    #   +x points right
    #   +y points up
    #   +z points at you
    # Tag corners: If you rotate the cube to view a different face using 
    # only one rotation, the tag corners sould be:
    #   [0 1 2 3] = [top left, top right, bottom right, bottom left]

    c = cube_width_m
    t = tag_width_m
    board_corners = [
        np.array([ [-t/2, t/2, c/2],  [t/2, t/2, c/2],  [t/2, -t/2, c/2],  [-t/2, -t/2, c/2]],  dtype=np.float32),
        np.array([ [c/2, t/2, t/2],   [c/2, t/2, -t/2], [c/2, -t/2, -t/2], [c/2, -t/2, t/2] ],  dtype=np.float32),
        np.array([ [t/2, t/2, -c/2],  [-t/2, t/2, -c/2],[-t/2, -t/2, -c/2],[t/2, -t/2, -c/2]],  dtype=np.float32),
        np.array([ [-c/2, t/2, -t/2], [-c/2, t/2, t/2], [-c/2, -t/2, t/2], [-c/2, -t/2, -t/2]], dtype=np.float32),
        np.array([ [-t/2, c/2, -t/2], [t/2, c/2, -t/2], [t/2, c/2, t/2],   [-t/2, c/2, t/2]],   dtype=np.float32),
        np.array([ [-t/2, -c/2, t/2], [t/2, -c/2, t/2], [t/2, -c/2, -t/2], [-t/2, -c/2, -t/2]], dtype=np.float32)
    ]
    board = cv2.aruco.Board_create(board_corners, arucoDict, board_ids)
    return board

def estimatePoseBoardAndValidate(detectedCorners, detectedIds, board, camera_matrix, camera_distortion, rvec = np.zeros(3), tvec = np.zeros(3)):
    # Assumes that a pose estimate is bad when its Z coordinate is negative,
    # then drops out tags one at a time to re-estimate the pose. 

    # Estimate the pose with all the detected corners
    [ret, rvec, tvec] = cv2.aruco.estimatePoseBoard(detectedCorners, detectedIds, 
                        board, camera_matrix, camera_distortion, rvec, tvec)
    valid = True

    # If the board position was estimated with negative Z coordinate, presume it failed.
    # Drop out markers one at a time until we get a reasonable measurement.
    if tvec[-1] < 0: # tvec[-1] is the estimated Z coordinate
        ret = 0
        valid = False
        idxes = np.arange(len(detectedCorners))
        for drop_idx in idxes:
            detectedCorners_sub = tuple([detectedCorners[i] for i in idxes if i != drop_idx])
            detectedIds_sub = np.array([detectedIds[i] for i in idxes if i != drop_idx])
            [ret_test, rvec_test, tvec_test] = cv2.aruco.estimatePoseBoard(detectedCorners_sub, detectedIds_sub, 
                board, camera_matrix, camera_distortion, np.zeros(3), np.zeros(3))
            if tvec_test[-1] > 0:
                rvec = rvec_test
                tvec = tvec_test
                ret = ret_test
                valid = True
                break

    # If the z coordinate is still negative, 
    # return the results but 
    return(ret, rvec, tvec, valid)

# Output video path
out_video_path = None

# Draw tracking? 
draw_tracking = True

# Draw corner numbers?
draw_corner_nums = False

# Draw edges of each tag?
draw_aruco_edges = True

# Draw the ID of each tag?
draw_aruco_ids = False

# Draw the cube pose?
draw_aruco_axes = True

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
aruco_tag_size_meters = 0.080 # 4 inch cube
# aruco_tag_size_meters = 0.040 # 2 inch cube

# Length of each side of the aruco cube in meters.
# 2 inch = 0.0508 mm 
# 4 inch = 0.1016 mm
aruco_cube_size_meters = 0.1016 # 4 inch cube
# aruco_cube_size_meters = 0.0508 # 2 inch cube

# Create aruco board object (for the 4" cube)
# Order of tags in board_id should be: [front, right, back, left, top, bottom]
board_ids = np.array([5, 6, 7, 8, 9, 10]) # 4 inch cube
# board_ids = np.array([0, 1, 2, 3, 5, 4]) # 2 inch cube. There is actually no tag 5; need it to specify cube 

# Make the aruco cube
board = create_aruco_cube(cube_width_m = aruco_cube_size_meters, tag_width_m = aruco_tag_size_meters, board_ids = board_ids, aruco_dict = arucoDict)

# DepthAI Pipeline
pipeline = dai.Pipeline()

# Set up the camera
if camera_type == "color":
    cam = pipeline.create(dai.node.ColorCamera)

    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1920, 1080)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(40)
    inputFrameShape = cam.getVideoSize()
    cameraBoardSocket = dai.CameraBoardSocket.RGB
else: 
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    cam.setFps(120)
    inputFrameShape = cam.getResolutionSize()
    if camera_type == "mono_left":
        cameraBoardSocket = dai.CameraBoardSocket.LEFT
    elif camera_type == "mono_right":
        cameraBoardSocket = dai.CameraBoardSocket.RIGHT
    else: 
        print("Unknown camera type: " + camera_type)
        print("Allowable options: color, mono_left, mono_right")
        sys.exit()

    # Set the mono board socket
    # Did it this way because I refer to cameraBoardSocket later.
    cam.setBoardSocket(cameraBoardSocket)
    
# This node is used to send the image from device -> host
xOutImage = pipeline.create(dai.node.XLinkOut)
xOutImage.setStreamName("imageOut")
xOutImage.input.setBlocking(False)
xOutImage.input.setQueueSize(1)

# Send the images to the output queues
if isinstance(cam, dai.node.ColorCamera):
    cam.video.link(xOutImage.input)
elif isinstance(cam, dai.node.MonoCamera):
    cam.out.link(xOutImage.input)
else:
    print("Unknown camera node type; exiting")
    sys.exit()

# Open a video writer object
if out_video_path is not None:
    outVideo = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30, inputFrameShape)
else:
    outVideo = None

# # # # # Kalman Filter set up # # # # # #

# Dimensionality of state space and measurements
dim_state = 12
dim_meas = 6
dt = 1 / cam.getFps() # Check this

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

    # 3x3 matrix of [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    camera_matrix = np.array(calibData.getCameraIntrinsics(cameraBoardSocket, resizeWidth=inputFrameShape[0], resizeHeight=inputFrameShape[1]))

    # vector of distortion coefficients (k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4)
    camera_distortion = np.array(calibData.getDistortionCoefficients(cameraBoardSocket))[:-2]

    # Listen for data on the device xLinkOut queue
    qImageOut = device.getOutputQueue(name="imageOut", maxSize=1, blocking=False)

    startTime = time.monotonic()
    prev_frame_time = 0
    new_frame_time = 0
    rmat_kalman_prev = np.eye(3)
    rvec_kalman = np.zeros(3)
    debug = False

    # Main processing loop
    while True:
        imgRaw = qImageOut.get()
        if imgRaw is None: 
            print("Empty image...")
            continue

        # Update the state transition matrix
        F[dt_idx] = dt
        kf.transitionMatrix = F

        # Calculate frames per second
        new_frame_time = time.time()
        dt = new_frame_time - prev_frame_time
        fps =  1 / dt
        
        # Update previous time for next loop
        prev_frame_time = new_frame_time

        # Get the image
        frame = imgRaw.getCvFrame()
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Detect the aruco markers
        (detectedCorners, detectedIds, rejectedCorners) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

        # Refine the detection
        detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = cv2.aruco.refineDetectedMarkers(frame, board, 
            detectedCorners, detectedIds, rejectedCorners, 
            cameraMatrix = camera_matrix, distCoeffs = camera_distortion, parameters=arucoParams)

        # Predict state (calculate xk_km1)
        kf.predict()
        tvec = np.array([0,0,0])

        # Estimate the board pose if any tags were detected
        if len(detectedCorners) > 0:
            [ret, rvec, tvec, valid] = estimatePoseBoardAndValidate(detectedCorners, detectedIds, board, camera_matrix, camera_distortion, np.zeros(3), np.zeros(3))
            
            measurement = np.concatenate([tvec, rvec], axis=0).astype(np.float64)
            print(measurement)

            # Correct state prediction (calculate xk_k)
            # Only do this if the measurement is "good," i.e. z coordinate > 0
            if valid is True:
                kf.correct(measurement)

            # Draw IDs?
            if draw_aruco_ids is True:
                frame = cv2.aruco.drawDetectedMarkers(frame, detectedCorners, detectedIds)

            # Initialize min and max coordinates for 
            # the bounding box around the entire cube
            xmin = np.inf
            xmax = 0
            ymin = np.inf
            ymax = 0
            for i, tag_corners in enumerate(detectedCorners):
                poly_pts = tag_corners[0].astype(np.int32)
                poly_pts.reshape((-1, 1, 2))
                ypts = [x[1] for x in poly_pts]
                xpts = [x[0] for x in poly_pts]
                ymin = np.min([ymin, np.min(ypts)])
                ymax = np.max([ymax, np.max(ypts)])
                xmin = np.min([xmin, np.min(xpts)])
                xmax = np.max([xmax, np.max(xpts)])

                # Draw the aruco tag border on the frame
                if draw_aruco_edges is True:
                    frame = cv2.polylines(frame, [poly_pts], True, aruco_edge_color, aruco_edge_thickness)
                # Draw corner numbers
                if draw_corner_nums is True:
                    for j, corner in enumerate(tag_corners[0]):
                        cv2.putText(frame, str(j), (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,0,0))

            # Draw the tracking info on the frame
            if draw_tracking is True:
                cv2.putText(frame, 'tracking', (int(xmin), int(ymin) + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, tracker_color)
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), tracker_color, tracker_thickness, cv2.FONT_HERSHEY_SIMPLEX)
            
        # Update the state of the cube
        tvec_kalman = kf.statePost[0:3]
        rvec_kalman = kf.statePost[3:6]

        rmat_kalman, _ = cv2.Rodrigues(rvec_kalman)
        residual_mat = np.matmul(np.transpose(rmat_kalman), rmat_kalman_prev)
        residual_vec, _ = cv2.Rodrigues(residual_mat)
        rnorm_kalman = np.linalg.norm(residual_vec) 
        
        # Draw FPS on frame
        cv2.putText(frame, "FPS: {:.2f}".format(fps),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))

        # Copy the frame (we'll use the copied frame to visualize the Kalman-estimated board state)
        frame_kalman = frame.copy()

        # Draw the axes
        if draw_aruco_axes is True:
            frame_kalman = cv2.aruco.drawAxis(frame_kalman, camera_matrix, camera_distortion, rvec_kalman, tvec_kalman, aruco_tag_size_meters / 2)
            if len(detectedCorners) > 0:
                frame = cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, aruco_tag_size_meters / 2)

        # Display the image
        if outVideo is not None:
            outVideo.write(frame)
        cv2.imshow("Image", frame)
        cv2.imshow("Kalman", frame_kalman)
        
        key=cv2.waitKey(10)
        if key == ord('q'):
            break
        elif key == ord(' '):
            debug = not debug
            if debug is True:
                print("Debugging ON")
            else:
                print("Debugging OFF")

        # Update previous rotation matrix
        rmat_kalman_prev = rmat_kalman
        rvec_kalman_prev = rvec_kalman

        if (tvec[-1] < 0) and (debug is True):
            print("Warning: tvec[-1] < 0")
            set_trace()
        if (rnorm_kalman > 2) and (debug is True):
            print("Warning: norm(rvec_kalman) > threshold")
            set_trace()

if outVideo is not None:
    outVideo.release()



    



