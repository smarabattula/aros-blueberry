# aros-blueberry
ARoS Lab Blueberry Detection Project

Used SAM and RoboFlow to annotate dataset

blueberry.ipynb is used to crop the images using yellow thresholding

blueberry_masks.ipynb is used to derive segmentation masks

blueberry_yolo.ipynb has the code to train a YOLOv8 model

ColorCorrection.ipynb has code snippets used for performing color correction

sagemaker_working.ipynb is used to deploy a Sagemaker endpoint for detectron2 model


sasank_blueberry_segmentation_m1_12_7_23.ipynb file will be used in future for blueberry estimates. (Private file, people with access permissions can only view this notebook) 

# How to run?

Make sure all required dependencies in your machine are installed (Mentioned all dependency installation in notebook already)

Run the notebooks!

# Citations

Roboflow. Supervision [Computer software]. https://github.com/roboflow/supervision

Roboflow notebooks. https://github.com/roboflow/notebooks

