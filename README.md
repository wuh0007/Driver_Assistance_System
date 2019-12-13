# Driver_Assistance_System
Combine Yolov3 framework with road lane detection in order to accomplish the task of hazard recognition.

**Technique used:**    

Image Processing:

1.**color & gradients transforms**

2.**threshold generation on binary image**   

3.**perspective transformation**

4.**pixel tracking to find lane boundary**

Machine learning algorithm: 

1.**Convolutional Neural Network**

2.**Yolov3 framework**   

Combined:

1.**Determine threshold to recognize car position**

2.**print front road status into image**  

**Requirement** 

- Python 3.6  
- Keras
- Tensorflow > 1.0   
- Opencv3
- PIL

## Data

For the dataset, we collected hundreds of pictures taken by the middle camera in front of the car, and labeled them with labelling tool call 'labellmg', below are some sample labeled images.

![1](/image/1.png)

![2](/image/2.png)

![3](/image/3.png)

## 1 - Color and gradients transforms

We manually set the thresholds for both color and gradients transforms in order to extract the pixels of road lanes.

![Figure_2](/image/Figure_2.png)

![Figure_3](/image/Figure_3.png)

![Figure_7](/image/Figure_7.png)

![Figure_8](/image/Figure_8.png)

![Figure_9](/image/Figure_9.png)

![Figure_10](/image/Figure_10.png)

## 2 - Lane status analysis

We determine the curvature of the lane and the lane postion.

![Figure_12](/image/Figure_12.png)

![Figure_14](/image/Figure_14.png)

## 3 - Lane agumentation

We invert the perspective transform based on detected lane position and print it back to original image

![Figure_15](/image/Figure_15.png)

## 4 - Vehicle detection and warning system

We adopt Yolov3 framework which helps detect vehicle on the road, and combine it with road lane detection
Give warning once hazard is detected.

![Figure_16](/image/Figure_16.png)

![Figure_17](/image/Figure_17.png)

Yolov3 is well explained in this flowchart. source from CSDN. Great thanks to the author. https://blog.csdn.net/leviopku/article/details/82660381

![Yolov3_Structure](/image/Yolov3_Structure.png)

When it comes to the output of Yolov3, for each image, in our project, Yolo will output a scale * scale*3*（5+1） = 18 vector. That means, for each grid in each scale, 3 bounding box will be output, and each bouding box contains the 5+1 = 6 metrics, which are x,y axis of center of bounding boxes, width and height of bounding boxes, objectiveness and class score of bounding boxes.

