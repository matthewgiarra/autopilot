import numpy as np
import cv2

# Draw detected tag IDs
def draw_ids(frame, detectedCorners = [], detectedIds = []):
    frame = cv2.aruco.drawDetectedMarkers(frame, detectedCorners, detectedIds)
    return frame

# Draw the detected tag edges
def draw_edges(frame, detectedCorners = [], color = (0,0,255), thickness = 4):
    if len(detectedCorners) > 0:
        for i, tag_corners in enumerate(detectedCorners):
            poly_pts = tag_corners[0].astype(np.int32)
            poly_pts.reshape((-1, 1, 2))
            frame = cv2.polylines(frame, [poly_pts], True, color, thickness)
    return frame

# Draw corner numbers
def draw_corners(frame, detectedCorners = [], size = 0.5, color = (255,0,0)):
    if len(detectedCorners) > 0:
        for i, tag_corners in enumerate(detectedCorners):
            for j, corner in enumerate(tag_corners[0]):
                cv2.putText(frame, str(j), (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_TRIPLEX, size, color)
    return frame

def draw_tracking(frame, detectedCorners = [], size = 0.5, color = (0, 255, 255), thickness = 4):
    # Initialize min and max coordinates for the bounding box
    xmin = np.inf
    xmax = 0
    ymin = np.inf
    ymax = 0

    # Figure out the extents of the bounding box
    if len(detectedCorners) > 0:
        for i, tag_corners in enumerate(detectedCorners):
            poly_pts = tag_corners[0].astype(np.int32)
            poly_pts.reshape((-1, 1, 2))
            ypts = [x[1] for x in poly_pts]
            xpts = [x[0] for x in poly_pts]
            ymin = np.min([ymin, np.min(ypts)])
            ymax = np.max([ymax, np.max(ypts)])
            xmin = np.min([xmin, np.min(xpts)])
            xmax = np.max([xmax, np.max(xpts)])

        # Draw 
        cv2.putText(frame, 'tracking', (int(xmin), int(ymin) + 20), cv2.FONT_HERSHEY_TRIPLEX, size, color)
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness, cv2.FONT_HERSHEY_SIMPLEX)
    return frame

def draw_fps(frame, fps, size = 0.4, color = (255,255,255)):
    cv2.putText(frame, "FPS: {:.2f}".format(fps),
                    (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
    return frame

def draw_pose(frame, camera_matrix, camera_distortion, rvec, tvec, axis_size):
    frame = cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, axis_size)
    return frame