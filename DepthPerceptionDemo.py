from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import ctypes
import _ctypes
import pygame
import sys
import numpy as np
import os
import time
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from config import Config
import utils
import model as modellib

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

ROOT_DIR = os.getcwd()
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class CocoConfig(Config):
    NAME = "coco"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

class DepthRuntime(object):
    def __init__(self):
        #Initialize Window & Kinect2
        print("Initializing Window & Kinect 2")
        pygame.init()
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 30)
        self._done = False
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        pygame.display.set_caption("Depth Perception Mask R-CNN Demo " + str(self._kinect.color_frame_desc.Width) + "x" + str(self._kinect.color_frame_desc.Height))
        print("Task Completed")

        #Initialize CNN & Coco Model
        print("Initializing TensorFlow & Coco Model")
        class InferenceConfig(CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
        self._model = modellib.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)
        self._model.load_weights(COCO_MODEL_PATH, by_name=True)

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def draw_depth_frame(self, frame, target_surface):
        if frame is None:
            return
        f8=np.uint8(frame.clip(1,4000)/32.)
        frame8bit=np.dstack((f8,f8,f8))
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address
        target_surface.unlock()

    def run(self):
        while not self._done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._done = True

                elif event.type == pygame.VIDEORESIZE:
                    self._screen = pygame.display.set_mode(event.dict['size'], pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

            if self._kinect.has_new_depth_frame():
                frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(frame, self._frame_surface)
                frame = None

            #Get Image as Numpy 3D Array [H,W,3]
            image = pygame.surfarray.array3d(self._frame_surface)
            image.swapaxes(0,1)

            #Object Detection Results
            r = self._model.detect([image], verbose=0)[0]
            rois = r["rois"]
            class_id = r["class_ids"]
            scores = r["scores"]
            masks = r["masks"]

            #Draw Masks Onto Image
            if rois is not None:
                for i in range(rois.shape[0]):
                    if class_names[class_id[i]] == 'person' and scores[i] >= 0.9:
                        color = [0,0,255]
                        mask = masks[:,:,i]
                        for c in range(3):
                            image[:,:,c] = np.where(mask == 1, image[:,:,c] * (1 - 0.5) + 0.5 * color[c] * 255, image[:,:,c])

            #Draw Image
            image.swapaxes(0,1)
            newFrame = pygame.surfarray.make_surface(image)
            self._screen.blit(newFrame, (0,0))

            #Draw Obejct Detected labels
            if rois is not None:
                for i in range(rois.shape[0]):
                    if class_names[class_id[i]] == 'person' and scores[i] >= 0.9:
                        box = rois[i]
                        pygame.draw.rect(self._screen, (0,0,255), (box[0], box[1], box[2]-box[0], box[3]-box[1]), 5)
                        label = self._font.render(class_names[class_id[i]] + ": " + str(scores[i]), True, pygame.Color('white'))
                        self._screen.blit(label, ((box[0] + box[2])/2, box[1]-50))

            #Draw FPS
            fps = self._font.render("FPS: " + str(int(self._clock.get_fps())), True, pygame.Color('white'))
            self._screen.blit(fps, (50, 50))

            pygame.display.update()

            pygame.display.flip()

            self._clock.tick(60)

        self._kinect.close()
        pygame.quit()


__main__ = "Depth Perception Mask R-CNN Demo"
game = DepthRuntime();
game.run();
