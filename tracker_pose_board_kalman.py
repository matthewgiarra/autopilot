import depthai as dai
import cv2
import numpy as np
import sys
from pdb import set_trace

np.set_printoptions(precision=2)
np.set_printoptions(floatmode = "fixed")

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
        rvec_pre = meas_pre[3:6]
        rvec_meas = measurement[3:6]
        
        # Length of the difference between 
        # measured and predicted Rodrigues vectors
        # Here we're assuming that a difference
        # greater than pi probably means 
        # the rotation axis flipped between observations,
        # i.e., the rotation angle wrapped around 
        norm_d = np.linalg.norm(rvec_meas - rvec_pre)
        if norm_d > np.pi:
            xk_km1[3:6] = -1 * xk_km1[3:6]
            xk_km1[9:]  = -1 * xk_km1[9:]

        # Pre-fit measurement residual
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


# Output video path
out_video_path = None

# Draw tracking? 
draw_tracking = True

# Draw corner numbers?
draw_corner_nums = False

# Draw edges of each tag?
draw_aruco_edges = False

# Draw the ID of each tag?
draw_aruco_ids = True

# Draw the cube pose?
draw_aruco_axes = True

# Which camera to use ("mono_left," "mono_right," or "color")
camera_type = "mono_left"

# For plotting
tracker_color = (0,255,255) # Line / text color
tracker_thickness = 4 # Line thickness
aruco_color = (0,0,255) # Line / text color
aruco_thickness = 4 # Line thickness

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
    cam.setPreviewSize(640, 400)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(40)
    inputFrameShape = cam.getPreviewSize()
    cameraBoardSocket = dai.CameraBoardSocket.RGB
else: 
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
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
    
# Image manipulator node for setting data type
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
manip.setMaxOutputFrameSize(3072000)

# This node is used to send the image from device -> host
xOutImage = pipeline.create(dai.node.XLinkOut)
xOutImage.setStreamName("imageOut")

# This node is used to pass the aruco bounding bbox coordinates from host -> device
xinDetections = pipeline.create(dai.node.XLinkIn)
xinDetections.setStreamName("inDetections")

# This node is used to send the tracks from device -> host
xOutTracks = pipeline.create(dai.node.XLinkOut)
xOutTracks.setStreamName("trackletsOut")

# This node is the object tracker (device)
objectTracker = pipeline.create(dai.node.ObjectTracker)
objectTracker.inputTrackerFrame.setBlocking(True)
objectTracker.inputDetectionFrame.setBlocking(True)
objectTracker.inputDetections.setBlocking(True)
objectTracker.setDetectionLabelsToTrack([1])
objectTracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.SMALLEST_ID)
objectTracker.setMaxObjectsToTrack(1)

# Send the raw image to the image manipulator node to set the correct color format etc
if isinstance(cam, dai.node.ColorCamera):
    cam.preview.link(manip.inputImage)
elif isinstance(cam, dai.node.MonoCamera):
    cam.out.link(manip.inputImage)
else:
    print("Unknown camera node type; exiting")
    sys.exit()

# Link image manipulator output to object tracker input
manip.out.link(objectTracker.inputTrackerFrame)
manip.out.link(objectTracker.inputDetectionFrame)

# Link image manipulator output to xLinkOutput's input (device -> host)
manip.out.link(xOutImage.input)

# Link detections (computed on host) to object tracker on the device
xinDetections.out.link(objectTracker.inputDetections)

# Link tracks (computed on device) to host (device -> host)
objectTracker.out.link(xOutTracks.input)

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
kf = KalmanFilterPose(dim_state, dim_meas, 0, type=cv2.CV_64F)

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
Q = 1E-2 * np.eye(dim_state, dtype=np.float64)

# Covariance of measurement noise
R = 1E-2 * np.eye(dim_meas, dtype=np.float64)
# R[3,3] = R[4,4] = R[5,5] = 1E-4

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

    # Listen for data on the device xLinkIn / xLinkOut queues
    qImageOut = device.getOutputQueue(name="imageOut", maxSize=4, blocking=False)
    qDetectionsIn = device.getInputQueue(name="inDetections", maxSize=4, blocking=False)
    qTracklets = device.getOutputQueue(name="trackletsOut", maxSize=4, blocking=False)

    # Main processing loop
    while True:
        imgRaw = qImageOut.get()
        if imgRaw is None: 
            print("Empty image...")
            continue
        
        # Get the image
        frame = imgRaw.getCvFrame()
        
        # Putting these outside the loop
        # lets us pass valid data to the detections queue
        # even if there are no detections
        decodedDetections = list()
        imgDetections = dai.ImgDetections()

        # Detect the aruco markers
        (corners_all, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

        # Calculates kf.statePre (xk_km1)
        kf.predict() 

        if len(corners_all) > 0:
            [ret, rvec, tvec] = cv2.aruco.estimatePoseBoard(corners_all, ids, board, camera_matrix, camera_distortion, np.zeros(3), np.zeros(3))
            rot_theta = np.linalg.norm(rvec)
            rot_axis = rvec / rot_theta
            measurement = np.concatenate([tvec, rvec], axis=0).astype(np.float64)
            kf.correct(measurement)

            # Draw IDs?
            if draw_aruco_ids is True:
                frame = cv2.aruco.drawDetectedMarkers(frame, corners_all, ids)

            # Initialize min and max coordinates for 
            # the bounding box around the entire cube
            xmin = np.inf
            xmax = 0
            ymin = np.inf
            ymax = 0
            for i, tag_corners in enumerate(corners_all):
                poly_pts = tag_corners[0].astype(np.int32)
                poly_pts.reshape((-1, 1, 2))
                ypts = [x[1] for x in poly_pts]
                xpts = [x[0] for x in poly_pts]
                ymin = np.min([ymin, np.min(ypts)])
                ymax = np.max([ymax, np.max(ypts)])
                xmin = np.min([xmin, np.min(xpts)])
                xmax = np.max([xmax, np.max(xpts)])
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            # Draw the tracking info on the frame
            if draw_tracking is True:
                cv2.putText(frame, 'tracking', (xmin, ymin + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, tracker_color)

            # Draw the aruco tag border on the frame
            if draw_aruco_edges is True:
                frame = cv2.polylines(frame, [poly_pts], True, aruco_color, aruco_thickness)
            # Draw corner numbers
            if draw_corner_nums is True:
                for j, corner in enumerate(tag_corners[0]):
                    cv2.putText(frame, str(j), (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,0,0))
        
            # Normalize detection coordinates to (0-1)
            imgDetection = dai.ImgDetection()
            imgDetection.xmin = xmin / inputFrameShape[0]
            imgDetection.xmax = xmax / inputFrameShape[0]
            imgDetection.ymin = ymin / inputFrameShape[1]
            imgDetection.ymax = ymax / inputFrameShape[1]
            imgDetection.confidence = 1.0 # Fake metadata
            imgDetection.label = 1 # Fake metadata

            # Append detection to list
            decodedDetections.append(imgDetection)
            imgDetections.detections = decodedDetections

        # Update the state of the cube
        tvec_kalman = kf.statePost[0:3]
        rvec_kalman = kf.statePost[3:6]
      
        # Send the aruco detections to the detections queue (host -> device)
        qDetectionsIn.send(imgDetections)

        # Read and plot the tracker output
        track = qTracklets.get()
        trackletsData = track.tracklets
        for t in trackletsData:
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

          # Copy the frame 
        frame_kalman = frame.copy()

        # Draw the axes
        if draw_aruco_axes is True:
            frame_kalman = cv2.aruco.drawAxis(frame_kalman, camera_matrix, camera_distortion, rvec_kalman, tvec_kalman, aruco_tag_size_meters / 2)

            if len(corners_all) > 0:
                frame = cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, aruco_tag_size_meters / 2)

        # Display the image
        if outVideo is not None:
            outVideo.write(frame)
        cv2.imshow("Image", frame)
        cv2.imshow("Kalman", frame_kalman)
        key=cv2.waitKey(10)
        if key == ord('q'):
            break

if outVideo is not None:
    outVideo.release()



    



