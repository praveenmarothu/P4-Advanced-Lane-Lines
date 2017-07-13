import numpy as np
import cv2
import matplotlib.image as mpimg
import glob
import os
import pickle

class Camera(object):

    calibrated_camera = None

    def __init__(self):
        self.camera_matrix = None
        self.dist_coeff = None

    def calibrate(self):

        nx,ny=9,6
        object_points = []
        image_points =[]

        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        images = glob.glob('./camera_cal/*')
        for idx, fname in enumerate(images):
            img  = mpimg.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret == True:
                object_points.append(objp)
                image_points.append(corners)
                cv2.drawChessboardCorners(img, (nx,ny), corners,ret)
                cv2.imwrite("output_images/chessboard/out_" + os.path.basename(fname) + ".png",img)
        image_size = mpimg.imread(images[0]).shape[0:2]
        ret, self.camera_matrix, self.dist_coeff, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)


    @staticmethod
    def get_calibrated_camera():

        camera = None
        if Camera.calibrated_camera is not None:
            camera = Camera.calibrated_camera

        elif os.path.isfile("camera_calibrated.p"):
            with open('camera_calibrated.p', 'rb') as f:
                camera = pickle.load(f)
        else:
            camera = Camera()
            camera.calibrate()
            with open('camera_calibrated.p', 'wb') as f:
                pickle.dump(camera, f)

        return camera




