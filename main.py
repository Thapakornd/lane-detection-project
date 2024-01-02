import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import glob
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from Pipeline import *
from moviepy.editor import VideoFileClip

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
    
    def process_image(self, img):
        clip = VideoFileClip(img, audio=False)
        result = clip.fl_image(self.forward)
        return result
    
if __name__ == "__main__":
    # Reading video file
    image = cv2.VideoCapture("MOV00551.avi")
    ret, frame = image.read()
    
    findLaneLine = FindLaneLine()
    
    # Result for testing video frame
    result = findLaneLine.process_image("test1.mp4")
    result.write_videofile("output_3.mp4", audio=False)
    
    # For real time
    while ret:
        ret, frame = image.read()
        frame = cv2.resize(frame, (1280,720))
        
        #undistort,binary_warped = findLaneLine.forward(frame)
        
        #histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        #plt.plot(histogram)
        #plt.show()
        #cv2.imshow("Result", undistort)
        #cv2.imshow("Binary", binary_warped)
        #cv2.imshow("Original", frame)
        
        if cv2.waitKey(60) == 27:
            break
    