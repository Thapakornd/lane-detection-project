import cv2
import numpy as np

class PerspectiveTransformation:
    
    def __init__(self):
        """Init PerspectiveTransformation."""
        self.src = np.float32([(550-50, 460),     # top-left
                               (150-40, 720),     # bottom-left
                               (1200-140, 720),    # bottom-right
                               (770+40, 460)])    # top-right
        self.dst = np.float32([(100, 0),
                               (100, 720),
                               (1100, 720),
                               (1100, 0)])
        
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)
        
    def plot_roi(self, img):
        cv2.circle(img, (int(self.src[0][0]),int(self.src[0][1])), 5, (0,0,255), -1)
        cv2.circle(img, (int(self.src[1][0]),int(self.src[1][1])), 5, (0,0,255), -1)
        cv2.circle(img, (int(self.src[2][0]),int(self.src[2][1])), 5, (0,0,255), -1)
        cv2.circle(img, (int(self.src[3][0]),int(self.src[3][1])), 5, (0,0,255), -1)
        return img
        
    def forward(self, img, img_size=(1280,720), flags=cv2.INTER_LINEAR):
        return cv2.warpPerspective(img, self.M, img_size, flags=flags)
    
    def backward(self, img, img_size=(1280,720), flags=cv2.INTER_LINEAR):
        return cv2.warpPerspective(img, self.M_inv, img_size, flags=flags)