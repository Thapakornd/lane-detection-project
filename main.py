import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import glob
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from Pipeline import *

class FindLaneLine:
    
    def __init__(self):
        self.calibration = CameraCalibration('camera_cal/*.jpg', 9,6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.pipeline = Pipeline()
        
    def forward(self, img):
        img = self.calibration.undistort(img)
        img = self.transform.plot_roi(img)
        binary_warped = self.thresholding.forward(img)
        img = self.pipeline.process_image(img, binary_warped)
        return img,binary_warped
    
if __name__ == "__main__":
    image = cv2.VideoCapture("MOV00551.avi")
    ret, frame = image.read()
    
    findLaneLine = FindLaneLine()
    
    while ret:
        ret, frame = image.read()
        frame = cv2.resize(frame, (1280,720))
        
        undistort,binary_warped = findLaneLine.forward(frame)
        
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        plt.plot(histogram)
        plt.show()
        cv2.imshow("Result", undistort)
        cv2.imshow("Binary", binary_warped)
        #cv2.imshow("Original", frame)
        
        if cv2.waitKey(60) == 27:
            break
    