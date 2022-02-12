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

# Limit value of num to be within range (v0, v1)
def clamp(num, v0, v1):
    return max(v0, min(num, v1))

# Manual exposure/focus set step
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

np.set_printoptions(precision=2)
np.set_printoptions(floatmode = "fixed")

# Output video path
out_video_path = None
# out_video_path = "/Users/matthewgiarra/Desktop/codrone_aruco_with_hat_400p_30fps_01.avi"
# out_video_path = "/Users/matthewgiarra/Desktop/codrone_autopilot_800p_03.avi"
# out_video_path = "/Users/matthewgiarra/Desktop/codrone_autopilot_800p_04.avi"
# out_video_path = "/Users/matthewgiarra/Desktop/codrone_autopilot_400p_05.avi"
# out_video_path = "/Users/matthewgiarra/Desktop/codrone_autopilot_400p_07.avi"
# out_video_path = "/Users/matthewgiarra/Desktop/codrone_autopilot_800p_07.avi"



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

# For plotting coordinates
xyz_xo = 0
xyz_dy = 35
xyz_yo = int(-3.5 * xyz_dy)
xyz_color = (0,0,0)
xyz_size = 0.8

# Aruco stuff
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_CONTOUR
# arucoParams.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX


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
cam_fps = 60
video_fps = 60

# The reference coordinate system is cameraBoardSockets[0]
cameraBoardSockets = [dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT]

# Initialize lists of cameras and output links
cameras = []
xOutImages = []

# Camera control node
controlIn = pipeline.create(dai.node.XLinkIn)
controlIn.setStreamName('control')

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

    # Control input
    controlIn.out.link(cam.inputControl)

    # Append camera to list of cams
    cameras.append(cam)
    xOutImages.append(xOutImage)

# Set up the connection for streaming out data
socket = 5555
pub = streams.Publisher(socket=socket)
pub.connect()

# Connect and start pipeline
with dai.Device(pipeline) as device:

    # Camera controls
    controlQueue = device.getInputQueue(controlIn.getStreamName())

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
        video_resolution = (resolution[0], resolution[1])
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
    plotFiltered = True

    # Set up the kalman filter
    dt = 1 / cam.getFps() # Time step

    # Measurement noise
    kf, dt_idx = kalman.kalmanFilter6DOFConstantVelocityMultiCam(dt = dt, measurementNoise = 0.05)
    H_ref = kf.measurementMatrix
    R_ref_diag = np.diag(kf.measurementNoiseCov)

    # Initialize camera controls
    ctrl = dai.CameraControl()
    ctrl.setManualExposure(expTime, sensIso)
    controlQueue.send(ctrl)

    # Initialize counter
    count = 0

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
            valid, measurement, detectedCorners, detectedIds = aruco.getPoseMeasurement(frame, board, 
            cameraMatrix=cameraMatrix, distCoeffs = distCoeffs, arucoDict = arucoDict, parameters=arucoParams)

            # Estimate the board pose if any tags were detected
            if len(detectedCorners) > 0 and i == 0:
                
                # Draw IDs?
                if draw_aruco_ids is True:
                    frame = drawing.draw_ids(frame, detectedCorners, detectedIds)
            
                if draw_aruco_edges is True:
                    frame = drawing.draw_edges(frame, detectedCorners)

                if draw_corner_nums is True:
                    frame = drawing.draw_corners(frame, detectedCorners)

                if draw_tracking_status is True:
                    frame = drawing.draw_tracking(frame, detectedCorners)

            if plotFiltered is False and i == 0:
                cv2.putText(frame, "Kalman filter OFF",
                    (2, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))

            # if draw_aruco_axes is True and plotFiltered is False and valid is True and i == 0:
            #     tvec = np.array(measurement[0:3])
            #     rvec = np.array(measurement[3:])
            #     drawing.draw_pose(frame, camera_matrix=cameraMatrix, camera_distortion=distCoeffs,
            #     rvec=rvec,tvec=tvec, axis_size = 2*aruco_tag_size_meters)

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
            if i == 0:
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

            # print("Loop %d" % count)

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

        # print("tr(Pk_k) = %0.2e" % errCovTrace)

        # Draw the axes on the rames
        for i, frame in enumerate(frames):

            if i == 0:

                # Transform cube coordinates into current camera's frame of ref
                pose_mat = np.eye(4)
                pose_mat[0:3, 0:3], _ = cv2.Rodrigues(rvec)
                pose_mat[0:3, 3] = np.transpose(tvec)
                # pose_mat_transformed = np.matmul(camera_extrinsics[i], pose_mat)
                # pose_mat_transformed = pose_mat_transformed / pose_mat_transformed[3,3]
                # rvec_transformed, _ = cv2.Rodrigues(pose_mat_transformed[0:3, 0:3])
                # tvec_transformed = pose_mat_transformed[0:3, 3]

                # Target point in the board coordinates
                xyz_target_board_frame_homogeneous = np.linalg.inv(pose_mat) @ xyz_target
                
                # XYZ of the target in drone-centered coordinate system, in mm 
                xyz_t_d = 1000 * xyz_target_board_frame_homogeneous / xyz_target_board_frame_homogeneous[-1]

                # Print the coordinates
                # print("Frame %d: [x, y, z] = %0.2f, %0.2f, %0.2f" % (count, xyz_t_d[0], xyz_t_d[1], xyz_t_d[2]))

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
                
                # Draw the pose
                if draw_aruco_axes is True and plotFiltered is True:
                    frame = drawing.draw_pose(frame, camera_matrices[i], camera_distortions[i], rvec, tvec, 2*aruco_tag_size_meters)

                # Display the image
                frame_title = str(cameraBoardSockets[i])
                cv2.imshow(frame_title, frame)

        # Write the video frame
        if outVideo is not None:
            # frames_cat = np.concatenate(frames, axis=1)
            outVideo.write(frames[0])

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
        
        count += 1

# Close the video
if out_video_path is not None:
    outVideo.release()

# GTFO
print("Exiting")


    



