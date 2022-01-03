
import numpy as np
import cv2

def create_aruco_cube(board_ids, aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50), cube_width_m = 0.0508, tag_width_m = 0.040):
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
    board = cv2.aruco.Board_create(board_corners, aruco_dict, board_ids)
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
    if tvec[-1] < 0: # tvec[-1] (last element in tvec) is the estimated Z coordinate
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
