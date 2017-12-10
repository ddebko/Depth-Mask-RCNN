# Depth-Mask-RCNN
Using Kinect2 Depth Sensors To Train a Neural Network For Object Detection &amp; Interaction

## This is a prototype design to train a Region-Based Convolution Neural Network to detect objects and how they interact with each other based on a depth map input.

### This project utilizes [Mask_RCNN](https://github.com/matterport/Mask_RCNN) for Object Detection and Segmentation.
### This project utilizes [PyKinect2](https://github.com/Kinect/PyKinect2) to feed the video stream from the kinect and into the Mask_RNN prediciton model.
### I was encouraged to try this solution based on this [Demo](https://www.youtube.com/watch?v=OOT3UIXZztE)

# The Plan
To use Mask_RCNN prediciton model to obtain the mask region data and use this information to extract the average depth distance inside the region from the Kinect2 sensor.
By adding the average depth distance as a parameter to the prediction model, it becomes possible for the neural network to predict how objects are iteracting with each other. The improved predicion model takes into account the depth of each object and if the mask layers are overlapping.
For example, the model would be able to predict if a worker was holding a tool based on the person mask layer overlapping with the tool mask layer and if their average depth distance passes a threshold of certainty to determine if their are in contact with one another.

# The Results
I utilized a Nvidia GTX 1080 graphics card to processes the video stream. Although this is one of the best graphics card on the market, the average fps ranged between 0-2.
Based on these results, I would not encourage using a Region-Based Convolution Neural Network for realtime object detection. I talked to the creator of the Mask_RCNN project and they confirmed that this prediciton model does have an average fps of 2.
Slow speeds are also described as an issue in the science journal [Mask R-CNN](https://arxiv.org/abs/1703.06870)

# Conclusion
The focus of this project is for the object detection to be realtime. So a future implementation of the mask prediction model would need to be created using a Single Shot MultiBox Detector such as [Yolo9000](https://github.com/philipperemy/yolo-9000).
[Single Shot MultiBox Detectors](https://www.youtube.com/watch?v=02vmIjAAY8c) are known to be very fast with an average fps of 30-90. I acknowledge that Single Shot MultiBox Detectors have faults in accuracy due to occlusion and object size.
