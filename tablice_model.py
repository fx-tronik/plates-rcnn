import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import numpy as np
import cv2
from config import Config
import model as modellib

#MODEL_PATH = 'logs/plates20180423T0856/mask_rcnn_plates_0131.h5'   #najlepszy do tej pory model
#MODEL_PATH = 'logs/plates20180604T1313/mask_rcnn_plates_0001.h5'
#ROOT_DIR = os.getcwd()
#MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = "plates"
    NUM_CLASSES = 1 + 1  # background + ` shapes
    IMAGE_MIN_DIM = 1024*1
    IMAGE_MAX_DIM = 1024*4
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels
    POST_NMS_ROIS_INFERENCE = 5000
    RPN_NMS_THRESHOLD = 0.8

class lokalizacja_tablic():
    def __init__(self, model_path):
        self.inference_config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", 
                          config=self.inference_config,
                          model_dir='logs/')
        print("Wczytywanie wag z: ", model_path)
        self.model.load_weights(model_path, by_name=True)

    def detect(self, image):
        results = self.model.detect([image], verbose=0)
        res = results[0]
        masks = res['masks']
        pts = []
    
        for i in range(masks.shape[2]):
            img, contours, hierarchy = cv2.findContours(masks[:,:,i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for con in contours:
                rect = cv2.minAreaRect(np.array(con))
                if(rect[1][1] * rect[1][0] < 100):      # w*h jesli obszar mniejszy niz 100 pikseli to pominac
                    continue
                bbox = cv2.boxPoints(rect).astype(int)
                if(rect[1][1] > rect[1][0]):      # if w > h
                    bbox = np.array([bbox[1], bbox[2], bbox[3], bbox[0]])
                pts.append(bbox)
        return pts, res