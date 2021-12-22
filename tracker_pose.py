import depthai as dai
import cv2
import numpy as np
import sys
from pdb import set_trace

# Output video path
out_video_path = "out_pose.avi"

# Draw tracking? 
draw_tracking = True

# Which camera to use ("mono_left," "mono_right," or "color")
camera_type = "mono_right"

# For plotting
tracker_color = (0,255,255) # Line / text color
tracker_thickness = 4 # Line thickness
aruco_color = (0,0,255) # Line / text color
aruco_thickness = 8 # Line thickness

# Aruco stuff
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()

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

# Connect and start pipeline
with dai.Device(pipeline) as device:

    # Get camera calibration info
    calibData = device.readCalibration()

    # 3x3 matrix of [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    camera_matrix = np.array(calibData.getCameraIntrinsics(cameraBoardSocket, resizeWidth=inputFrameShape[0], resizeHeight=inputFrameShape[1]))

    # vector of distortion coefficients (k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4)
    camera_distortion = np.array(calibData.getDistortionCoefficients(cameraBoardSocket))[:-2]

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
        
        if len(corners_all) > 0:
            rvec, tvec, obj_pts = cv2.aruco.estimatePoseSingleMarkers(corners_all, 0.040, camera_matrix, camera_distortion)
            
            # set_trace()

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

                # Draw the aruco tag border on the frame
                frame = cv2.polylines(frame, [poly_pts], True, aruco_color, aruco_thickness)
                frame = cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec[i], tvec[i], 0.02)

            
            
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

            if draw_tracking is True:
                # Draw the tracking info on the frame
                cv2.putText(frame, t.status.name, (x1, y2 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, tracker_color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), tracker_color, tracker_thickness, cv2.FONT_HERSHEY_SIMPLEX)

        # Display the image
        if outVideo is not None:
            outVideo.write(frame)
        cv2.imshow("Image", frame)
        key=cv2.waitKey(10)
        if key == ord('q'):
            break

if outVideo is not None:
    outVideo.release()



    



