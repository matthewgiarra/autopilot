import depthai as dai
import cv2
import numpy as np
import sys
from pdb import set_trace

# Bounding box
class bbox():
    def __init__(self):
        self.xmin = 0
        self.ymin = 0
        self.xmax = 0
        self.ymax = 0
        self.label = 1
        self.confidence = 0.5        

# Which camera to use ("mono_left," "mono_right," or "color")
camera_type = "color"

# For plotting
color = (255, 0, 0)

# Aruco stuff
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()

# DepthAI stuff
pipeline = dai.Pipeline()

# Set up the camera
if camera_type == "color":
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 400)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(40)
else: 
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    cam.setFps(120)
    if camera_type == "mono_left":
        cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    elif camera_type == "mono_right":
        cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    else: 
        print("Unknown camera type: " + camera_type)
        print("Allowable options: color, mono_left, mono_right")
        sys.exit()

# Image manipulator node for setting data type
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

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
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.SMALLEST_ID)

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

# Connect and start pipeline
with dai.Device(pipeline) as device:
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
        
        # Image size. Can't we do this outside the loop? 
        inputFrameShape = frame.shape

        # Detect the aruco markers
        (corners_all, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
        
        # Initialize imgDetections structure
        imgDetections = dai.ImgDetections()
        decodedDetections = list()

        # Each aruco tag is a corner
        for tag_corners in corners_all:
            # Make an empty ImgDetection object
            imgDetection = dai.ImgDetection()

            # Get the aruco corners
            poly_pts = tag_corners[0].astype(np.int32)
            poly_pts.reshape((-1, 1, 2))
            ypts = [x[1] for x in poly_pts]
            xpts = [x[0] for x in poly_pts]
            ymin = np.min(ypts)
            ymax = np.max(ypts)
            xmin = np.min(xpts)
            xmax = np.max(xpts)

            # Draw the aruco tag border on the frame
            frame = cv2.polylines(frame, [poly_pts], True, (0,0,255), 8)

            # Normalize detection coordinates to (0-1)
            imgDetection.xmin = xmin / inputFrameShape[1]
            imgDetection.xmax = xmax / inputFrameShape[1]
            imgDetection.ymin = ymin / inputFrameShape[0]
            imgDetection.ymax = ymax / inputFrameShape[0]

            # Faking metadata for now
            imgDetection.confidence = 1.0
            imgDetection.label = 1

            # Add this detection to the list of detections
            decodedDetections.append(imgDetection)
        
        # Populate imgDetections 
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
            label = t.label

            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        # Display the image
        cv2.imshow("Image", frame)
        key=cv2.waitKey(10)
        if key == ord('q'):
            break



    



