# Hand Tracking CNN, a replacement for the MediaPipe Hand Tracking that hasn't updated to the latest Python3.12 version
Made with Python and Pytorch
This is a Convolutional Neural Network created to replicate the MediaPipe hands ML model that has deprecated. 
`dataset.py` reads the data from the folder into a subclass of the Dataset class from pytorch, overloading the \_\_len\_\_ and \_\_getitem\_\_ functions. It reads the images and files using cv2, os, and json, and converts it to properly formatted data, a \[224, 224, 3\] tensor as an input and a \[42\] tensor for an output. (x, y for 21 points on the hand)
`dataloader.py` is a custom dataloader that is based on the torch.utils.data dataloader. Instead of making it an enumeratable list, it follows the iterator design pattern, returning and moving onto the next value. 
`display.py` displays the values using cv2 to display the image and draw the points onto the image. 
`model.py` is the CNN, using two sequential layers -- a Conv2d and a Linear -- to convert the image to the corresponding points on the hand. It converts the \[224, 224, 3\] input to a \[28, 28, 128\] tensor through the Convolutional Sequence, then flattens it to a \[128 * 28 * 28, 1\] tensor and finally outputs a \[42, 1\] tensor through the Linear Sequence.
`trainHandDetection.py` is the training logic. It saves (and grabs) from a `.pth` file to load the model, if avaliable, and continues training those weights to fit the data better. Training can be toggled on and off depending on need. It also randomly grabs data from the DataLoader, due to the iterator set to random on creation. This assures that rerunning the file does not continue to train it on the first number of images and all images are trained and tested on. 

# Data
The data is taken from a Kaggle dataset called "Hand Keypoint Dataset 26K". It is by Rion Dsilva and contains 26,768 images of hands using the MediaPipe library to get the points.
The dataset can be found here: https://www.kaggle.com/datasets/riondsilva21/hand-keypoint-dataset-26k
