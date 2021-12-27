import depthai as dai
import cv2
import numpy as np
import sys
from pdb import set_trace

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
draw_tracking = False

# Draw corner numbers?
draw_corner_nums = True

# Draw edges of each tag?
draw_aruco_edges = False

# Draw the ID of each tag?
draw_aruco_ids = True

# Draw the cube pose?
draw_aruco_axes = True

# Which camera to use ("mono_left," "mono_right," or "color")
camera_type = "color"

# For plotting
tracker_color = (0,255,255) # Line / text color
tracker_thickness = 4 # Line thickness
aruco_color = (0,0,255) # Line / text color
aruco_thickness = 4 # Line thickness

# Aruco stuff
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()

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
        
        if len(corners_all) > 0:
            
            # Draw the axes of the aruco cube to show its pose
            if draw_aruco_axes is True:
                # Estimate the pose of the whole cube
                [ret, rvec, tvec] = cv2.aruco.estimatePoseBoard(corners_all, ids, board, camera_matrix, camera_distortion, np.zeros([1,1,3]), np.zeros([1,1,3]))
                frame = cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, aruco_tag_size_meters / 2)
                print("rvec = " + str(np.squeeze(rvec)))
                print("tvec = " + str(np.squeeze(tvec)))
                print("")

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



    



