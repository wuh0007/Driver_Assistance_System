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

## Convert byte file to image

![originfile](/image/originfile.png)

![bytefileImage](/image/byteFileImage.PNG)

## Result

**K-Nearest-Neighbor:**

![KNN1](/image/KNN1.PNG) 

**Support Vector Machine:**

![SVM1](/image/SVM1.PNG) 

**Neural Network:** 

![ANN1](/image/ANN1.PNG) 
