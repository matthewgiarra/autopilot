import streams
from constants import CKeys, CColors
import kalman
import drawing
import aruco
import depthai as dai
import cv2
import numpy as np
import sys
import time
from pdb import set_trace

# Function to calculate the camera intrinsic matrix for a cropped field of view
def getCroppedCameraIntrinsics(calibData, cameraId, topLeft = dai.Point2f()):

    # Image height and width
    img_width, img_height = cameraId.getResolutionSize()

    # Get the default intrinsic matrix (uncropped)
    camera_matrix = calibData.getCameraIntrinsics(cameraId.getBoardSocket(), resizeWidth=img_width, resizeHeight=img_height)

    # Update the translation components of the matrix according to the crop location
    camera_matrix[0][2] -= topLeft.x * img_width
    camera_matrix[1][2] -= topLeft.y * img_height

    return camera_matrix

# Printing options
np.set_printoptions(precision=2)
np.set_printoptions(floatmode = "fixed")

# Limit value of num to be within range (v0, v1)
def clamp(num, v0, v1):
    return max(v0, min(num, v1))

# Exposure controls
expTime = 2000
expMin = 1
expMax = 33000
expStep = 500  # us

# Sensitivity controls
sensIso = 1600
sensMin = 100
sensMax = 1600
isoStep = 50

# Step size for crop location ('W','A','S','D' controls)
cropLocationStepSize = 0.02

# Output video path
out_video_path = None

# Draw tracking? 
draw_tracking_status = False

# Draw corner numbers?
draw_corner_nums = False

# Draw edges of each tag?
draw_aruco_edges = False

# Draw the ID of each tag?
draw_aruco_ids = False

# Draw the cube pose?
draw_aruco_axes = True

# Draw xyz coordinates?
draw_xyz = True

# Which camera's coordinate system to report state in
master_camera = "mono_right"

# Which camera to use ("mono_left," "mono_right," or "color")
camera_type = "mono_right"

 # Target position in the image (homogeneous coordinates, e.g., [x,y,z,1])
xyz_target = np.array([0,0,0.5,1], dtype=np.float32)

# For plotting
tracker_color = (0,255,255) # Line / text color
tracker_thickness = 2 # Line thickness
aruco_edge_color = (0,0,255) # Line / text color
aruco_edge_thickness = 2 # Line thickness
frames_to_plot = [0,1]

# For plotting coordinates
xyz_xo = 0
xyz_dy = 35
xyz_yo = int(-3.5 * xyz_dy)
xyz_color = (0,255,255)
xyz_size = 0.8

# Aruco stuff
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_CONTOUR

# # Aruco tag size in meters
# # Small cube uses 40 mm tags
# # Big cube uses 80 mm tags
aruco_tag_size_meters = 0.040 # 2 inch cube
# aruco_tag_size_meters = 0.080 # 4 inch cube
# aruco_tag_size_meters = 0.187325

# # Length of each side of the aruco cube in meters.
# # 2 inch = 0.0508 m 
# # 4 inch = 0.1016 m
aruco_cube_size_meters = 0.0508 # 2 inch cube
# aruco_cube_size_meters = 0.1016 # 4 inch cube
# aruco_cube_size_meters = 0 

# Create aruco board object (for the 4" cube)
# Order of tags in board_id should be: [front, right, back, left, top, bottom]
board_ids = np.array([0, 1, 2, 3, 5, 4]) # 2 inch cube. There is actually no tag 5; need it to specify cube 
# board_ids = np.array([5, 6, 7, 8, 9, 10]) # 4 inch cube

# Board rotation
board_rvec = np.array([0,0,np.pi])
# board_rvec = np.array([0,0,0])

# Make the aruco cube
board = aruco.create_aruco_cube(board_ids = board_ids, 
    aruco_dict = arucoDict, 
    cube_width_m = aruco_cube_size_meters,
    tag_width_m = aruco_tag_size_meters, 
    rvec = board_rvec)

# DepthAI Pipeline
pipeline = dai.Pipeline()

# Number of cameras
cam_fps = 40
fpsStepSize = 5
video_fps = 40

# The reference coordinate system is cameraBoardSockets[0]
cameraBoardSockets = [dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT]

# Initialize lists of cameras and output links
cameras = []
xOutImages = []
imageManips = []

# Camera control node
controlIn = pipeline.create(dai.node.XLinkIn)
controlIn.setStreamName('control')

# Camera config node
configIn = pipeline.create(dai.node.XLinkIn)
configIn.setStreamName('config')
sendCamConfig = False

# Initial crop range
topLeft = dai.Point2f(0.2, 0.2)
bottomRight = dai.Point2f(0.8, 0.8)

# Set up pipeline
for i in range(len(cameraBoardSockets)):

    # Mono camera nodes
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    cam.setFps(cam_fps)
    cam.setBoardSocket(cameraBoardSockets[i])

    # Image manipulation nodes
    imageManip = pipeline.create(dai.node.ImageManip)

    # Output nodes
    xOutImage = pipeline.create(dai.node.XLinkOut)
    xOutImage.input.setBlocking(False)
    xOutImage.input.setQueueSize(1)
    xOutImage.setStreamName("imageOut" + str(i))

    # Cropping etc
    imageManip.initialConfig.setCropRect(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)
    imageManip.setMaxOutputFrameSize(cam.getResolutionHeight()*cam.getResolutionWidth()*3)

    # Link camera control / config
    controlIn.out.link(cam.inputControl)
    configIn.out.link(imageManip.inputConfig)

    # Link stuff
    cam.out.link(imageManip.inputImage)
    imageManip.out.link(xOutImage.input)

    # Append camera to list of cams
    cameras.append(cam)
    xOutImages.append(xOutImage)

# Set up the connection for streaming out data
socket = 5555
pub = streams.Publisher(socket=socket)
pub.connect()

# Connect and start pipeline
with dai.Device(pipeline) as device:

    # Control and config queues
    controlQueue = device.getInputQueue(controlIn.getStreamName())
    configQueue = device.getInputQueue(configIn.getStreamName())

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
        img_width, img_height = cam.getResolutionSize()

        # Intrinsic matrix
        camera_matrix = np.array(getCroppedCameraIntrinsics(calibData, cam, topLeft = topLeft))
        
        # Distortion coeffs
        camera_distortion = np.array(calibData.getDistortionCoefficients(cam.getBoardSocket()))[:-2]

        # Append matrices to lists
        camera_matrices.append(camera_matrix)
        camera_distortions.append(camera_distortion)

    # Open a video writer object
    if out_video_path is not None:

        # Figure out the video resolution
        videoWidthPixels  = int((bottomRight.x - topLeft.x) * cameras[0].getResolutionWidth())
        videoHeightPixels = int((bottomRight.y - topLeft.y) * cameras[0].getResolutionHeight())
        video_resolution = (videoWidthPixels, videoHeightPixels)

        # Open the videowriter
        outVideo = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, video_resolution)
    else:
        outVideo = None
        video_resolution = (0,0)
    
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
    plotFiltered = True

    # Set up the kalman filter
    dt = 1 / cam.getFps() # Time step

    # Measurement noise
    kf, dt_idx = kalman.kalmanFilter6DOFConstantVelocityMultiCam(dt = dt)
    H_ref = kf.measurementMatrix
    R_ref_diag = np.diag(kf.measurementNoiseCov)

    # Initialize camera controls
    ctrl = dai.CameraControl()
    ctrl.setManualExposure(expTime, sensIso)
    controlQueue.send(ctrl)

    # Initialize counter
    count = 0

    # Set up plotting windows
    for i, boardSock in enumerate(cameraBoardSockets):
        if i in frames_to_plot:
            # How far to shift image windows
            dx_image_windows = int((bottomRight.x - topLeft.x) * cameras[i].getResolutionWidth())
            fTitle = str(boardSock)
            cv2.namedWindow(fTitle)
            cv2.moveWindow(fTitle, i * dx_image_windows, 0)

    # Main processing loop
    while True:

        # Update time step
        new_frame_time = time.time()
        if prev_frame_time > 0:
            dt = new_frame_time - prev_frame_time
        prev_frame_time = new_frame_time
        
        # Update kalman state predictions (calculate xk_km1)
        kf.transitionMatrix[dt_idx] = dt
        kf.predict()

        # Initialize vectors for frames (images) and pose measurements
        frames = []
        measurements = []

        # Get all the frames before we do any processing so that they're closely spaced in time
        for i, queue in enumerate(image_out_queues):

            # Get the frame from the queue
            frame = queue.get().getCvFrame()
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frames.append(frame)

        # Process each frame
        for i, frame in enumerate(frames):
            cameraMatrix = camera_matrices[i]
            distCoeffs    = camera_distortions[i]

            # Measure the position and pose of the aruco board
            valid, measurement, detectedCorners, detectedIds = aruco.getPoseMeasurement(frame, board, cameraMatrix=cameraMatrix, distCoeffs = distCoeffs, arucoDict = arucoDict, parameters=arucoParams)

            # Estimate the board pose if any tags were detected
            if len(detectedCorners) > 0 and i in frames_to_plot:

                # Draw IDs?
                if draw_aruco_ids is True:
                    frame = drawing.draw_ids(frame, detectedCorners, detectedIds)
            
                if draw_aruco_edges is True:
                    frame = drawing.draw_edges(frame, detectedCorners)

                if draw_corner_nums is True:
                    frame = drawing.draw_corners(frame, detectedCorners)

                if draw_tracking_status is True:
                    frame = drawing.draw_tracking(frame, detectedCorners)

            if plotFiltered is False and i in frames_to_plot:
                cv2.putText(frame, "Kalman filter OFF",
                    (2, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))

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
            
            # Draw FPS on frame
            if i in frames_to_plot:
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

        # Error covariance trace
        errCovTrace = np.trace(kf.errorCovPost)

        # Create the data object
        dataframe = streams.DataFrame(
            tvec = tvec, rvec = rvec, 
            kfErrorCov = errCovTrace, 
            timestamp = new_frame_time,
            framenumber = count)
        pub.send_frame(data=dataframe)

        # Draw the axes on the rames
        for i, frame in enumerate(frames):

            if i in frames_to_plot:

                # Transform cube coordinates into current camera's frame of ref
                pose_mat = np.eye(4)
                pose_mat[0:3, 0:3], _ = cv2.Rodrigues(rvec)
                pose_mat[0:3, 3] = np.transpose(tvec)

                # Target point in the board coordinates
                xyz_target_board_frame_homogeneous = np.linalg.inv(pose_mat) @ xyz_target
                
                # XYZ of the target in drone-centered coordinate system, in mm 
                xyz_t_d = 1000 * xyz_target_board_frame_homogeneous / xyz_target_board_frame_homogeneous[-1]

                # Coordinates to plot
                xyz = np.squeeze(tvec)

                # Draw xyz
                if draw_xyz:
                    imagePoints, _ = cv2.projectPoints(objectPoints = tvec.astype(np.float32), rvec = np.array([0,0,0], dtype=np.float32), tvec = np.array([0,0,0], dtype=np.float32), cameraMatrix = camera_matrix, distCoeffs = camera_distortion)
                    xy = np.squeeze(imagePoints)
                    yc = int(xy[1])
                    xc = int(xy[0])

                    try:
                        cv2.putText(frame, "X: {:.0f} mm".format(xyz[0]*1000), (int(xc + xyz_xo), int(yc) + xyz_yo ), cv2.FONT_HERSHEY_TRIPLEX,         xyz_size, xyz_color)
                        cv2.putText(frame, "Y: {:.0f} mm".format(xyz[1]*1000), (int(xc + xyz_xo), int(yc) + xyz_yo + xyz_dy), cv2.FONT_HERSHEY_TRIPLEX,     xyz_size, xyz_color)
                        cv2.putText(frame, "Z: {:.0f} mm".format(xyz[2]*1000), (int(xc + xyz_xo), int(yc) + xyz_yo + 2 * xyz_dy), cv2.FONT_HERSHEY_TRIPLEX, xyz_size, xyz_color)
                    except Exception as err:
                        print(str(err))

                # Draw text saying that we're plotting filtered data
                if plotFiltered is True:
                    cv2.putText(frame, "Kalman filter ON", (2, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))

                if draw_aruco_axes is True and i == 0:
                    if plotFiltered is True:
                        frame = drawing.draw_pose(frame, camera_matrices[i], camera_distortions[i], rvec, tvec, 2*aruco_tag_size_meters)
                    else:
                        if len(measurements) > 0:
                            tvec = np.array(measurements[0][:3])
                            rvec = np.array(measurements[0][3:])
                            frame = drawing.draw_pose(frame, camera_matrices[i], camera_distortions[i], rvec, tvec, 2*aruco_tag_size_meters)

                # Display the image
                frame_title = str(cameraBoardSockets[i])
                cv2.imshow(frame_title, frame)
            else:
                frame_title = str(cameraBoardSockets[i])
                cv2.imshow(frame_title, frame)

        # Write the video frame
        if outVideo is not None:
            
            # If the size of the video and the size of the frame aren't equal, close the videoWriter and make a new one of the correct size. Any previously-written frames are lost! 
            if frames[0].shape[:2] != video_resolution[::-1]:
                outVideo.release()
                videoWidthPixels = frames[0].shape[1]
                videoHeightPixels = frames[0].shape[0]
                video_resolution = (videoWidthPixels, videoHeightPixels)
                outVideo = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, video_resolution)
            outVideo.write(frames[0]) # Write the frame to the output video

        key=cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            debug = not debug
            if debug is True:
                print("Debugging ON")
            else:
                print("Debugging OFF")
        elif key == ord('f'):
            plotFiltered = not plotFiltered
            if plotFiltered: 
                print("Kalman filter ON")
            else:
                print("Kalman filter OFF")
        elif key == ord('e'):
            print("Autoexposure enable")
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            controlQueue.send(ctrl)

        # Set exposure and ISO
        elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
            if key == ord('i'): expTime -= expStep
            if key == ord('o'): expTime += expStep
            if key == ord('k'): sensIso -= isoStep
            if key == ord('l'): sensIso += isoStep
            expTime = clamp(expTime, expMin, expMax)
            sensIso = clamp(sensIso, sensMin, sensMax)
            print("Setting manual exposure, time:", expTime, "iso:", sensIso)
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(expTime, sensIso)
            controlQueue.send(ctrl)

        # Move crop up
        elif key == ord('w'):
            if topLeft.y - cropLocationStepSize >= 0:
                topLeft.y -= cropLocationStepSize
                bottomRight.y -= cropLocationStepSize
                sendCamConfig = True

        # Move crop left
        elif key == ord('a'):
            if topLeft.x - cropLocationStepSize >= 0:
                topLeft.x -= cropLocationStepSize
                bottomRight.x -= cropLocationStepSize
                sendCamConfig = True
        
        # Move crop down
        elif key == ord('s'):
            if bottomRight.y + cropLocationStepSize <= 1:
                topLeft.y += cropLocationStepSize
                bottomRight.y += cropLocationStepSize
                sendCamConfig = True

        # Move crop right
        elif key == ord('d'):
            if bottomRight.x + cropLocationStepSize <= 1:
                topLeft.x += cropLocationStepSize
                bottomRight.x += cropLocationStepSize
                sendCamConfig = True

        # Make image bigger       
        elif key == ord('='):
            topLeft.x -= cropLocationStepSize
            topLeft.y -= cropLocationStepSize
            bottomRight.x += cropLocationStepSize
            bottomRight.y += cropLocationStepSize
            sendCamConfig = True

        # Make image smaller
        elif key == ord('-'):
            topLeft.x += cropLocationStepSize
            topLeft.y += cropLocationStepSize
            bottomRight.x -= cropLocationStepSize
            bottomRight.y -= cropLocationStepSize
            sendCamConfig = True

        # Update camera config (cropping)
        if sendCamConfig is True:

            # Clamp the image corner coordinates so the size can never go to 0
            topLeft.x = clamp(topLeft.x, 0.0, 0.9)
            topLeft.y = clamp(topLeft.y, 0.0, 0.9)
            bottomRight.x = clamp(bottomRight.x, topLeft.x + 0.1, 1.0)
            bottomRight.y = clamp(bottomRight.y, topLeft.y + 0.1, 1.0)

            # Configure the image crop
            cfg = dai.ImageManipConfig()
            cfg.setCropRect(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)
            
            # Send the image crop command to the camera
            configQueue.send(cfg)

            # Update the camera intrinsic matrices according to the new crop
            camera_matrices = []
            for cam in cameras:
                topLeftPixel = dai.Point2f(topLeft.x * img_width, topLeft.y * img_height)
                camera_matrix = np.array(getCroppedCameraIntrinsics(calibData, cam, topLeft = topLeft))
                camera_matrices.append(camera_matrix)            

            # No more cam config sending
            sendCamConfig = False
        
        # Update loop counter
        count += 1

# Close the video
if out_video_path is not None:
    outVideo.release()

# GTFO
print("Exiting")


    



