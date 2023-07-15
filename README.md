# Repository overview
Welcome to the repository! Here you will find an overview of the most important files and resources related to the project. Please note that this repository does not contain the full program but rather serves as a demonstration and reference for key components.

## Contents

1. [Introduction](#introduction)
2. [Important Files](#important-files)
3. [Missing Files](#missing-files)
4. [Example of usage](#example-of-usage)
5. [YOLO](#YOLO)

## Introduction 
The purpose of this repository is to provide an overview of the project and highlight the key files.

## Important files
The following are the most significant files in this repository:

- main.py: This file contains the main entry point of the program and demonstrates the overall flow of execution.
- utils_proj/dataset.py: Contains the Dataset class, the functions used for the transformations, and other useful functions related to the dataset.
- utils_proj/model.py: Contains the function used to load the two different Faster R-CNN models.
- utils_proj/testing.py: Contains the functions for test and validation.
- utils_proj/training.py: Contains the functions used for training.

## Missing Files
The files omitted from this repo are:

- The dataset folder
- The detection folder: it's a clone of the pytorch repo (https://github.com/pytorch/vision/tree/main/references/detection)
- The weights folder

## Example of usage
The main.py file supports the following operations:

- **Training**: To perform training, use the following command:
```bash
python3 main.py --mode train --data <path/to/dataset> --model <resnet50/mobilenet> --weigths <path/to/weights> --epochs <number/of/epochs>
```
- **Testing**: For testing purposes, run the following command:
```bash
python3 main.py --mode test --data <path/to/dataset> --model <resnet50/mobilenet> --weigths <path/to/weights>
```
- **Validation**: For validation purposes, run the following command:
```bash
python3 main.py --mode validation --data <path/to/dataset> --model <resnet50/mobilenet> --weigths <path/to/weights>
```
-- **Inference**: For perform inference on a single image or on a folder of images, run the following command:
```bash
python3 main.py --mode inference --model <resnet50/mobilenet> --weigths <path/to/weights> --image <path/to/tes/image or folder>
```

## YOLO 
Please note that all tests conducted on YOLO were performed on a separate file that is a clone of the official YOLO repository. As a result, the testing code is not displayed in this repository but only in the report.
The YOLO testing file includes the necessary implementation and dependencies specific to the YOLO algorithm. It is used to evaluate the performance and accuracy of the YOLO model on various datasets.
To access the official YOLO repository, you can visit: [Official YOLO Repository](https://github.com/ultralytics/yolov5)