import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class CameraCalibration():
    """ Class that calibrate camera using chessboard images.

    Attributes:
        mtx (np.array): Camera matrix 
        dist (np.array): Distortion coefficients
    """
    def __init__(self, image_dir, nx, ny, debug=False):
        """ Init CameraCalibration.

        Parameters:
            image_dir (str): path to folder contains chessboard images
            nx (int): width of chessboard (number of squares)
            ny (int): height of chessboard (number of squares)
        """
        fnames = glob.glob(image_dir)
        self.objpoints = []
        self.imgpoints = []
        
        # Coordinates of chessboard's corners in 3D
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
        # Go through all chessboard images
        for f in fnames:
            img = cv2.imread(f)

            # Convert to grayscale image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny))
            if ret == True:
                self.imgpoints.append(corners)
                self.objpoints.append(objp)

    def undistort(self, img):
        """ Return undistort image.

        Parameters:
            img (np.array): Input image

        Returns:
            Image (np.array): Undistorted image
        """
        # Convert to grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

        return cv2.undistort(img, mtx, dist, None, mtx)
