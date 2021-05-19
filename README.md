# Self-Driving-Robot-Using-Neural-Network 
### Python + Tensorflow + OpenCV
### Overview
   1) The Robot uses neural network to learn and predict decisions just like a human brain.
   2) The model is built using Neural Network and it's trained by feeding in labelled images of tracks.
   3) After the model is trained it will be capable of making its own decisions. The prediction will be made on the laptop due to larger memory and flexibility. Raspberry pi will be used to stream the video to laptop using Pi-camera.
   4) First we will train the model using the dataset that contains the labelled images of the track.
   5) Raspberry Pi will stream the live feed to the laptop and the predictions will be sent back to the raspberry pi.
   6) The raspberry pi will be connected to motor driver which will control the wheels of the bot. Ultrasonic sensor makes sure that the robot does not collide with anything. Once trained it can run autonomously and make its decisions.It will try to maintain its path along the track and prevent from collisions. 

### For more detailed explanation, please view my IEEE paper titled <a href="https://ieeexplore.ieee.org/document/8533870" target="_blank">Self Driving Robot using Neural Network</a> or view <a href="https://drive.google.com/file/d/1H88Ns1iP7Ow5b2O4E5hxdy6M_rmgmLdV/view?usp=sharing">PDF</a> 
    
### Watch the Robot in Action !!
<a href="https://photos.app.goo.gl/ajpMJDrQvKtk6TWy6" target="_blank">![alt text](https://github.com/akshay1997feb/Self-Driving-Robot-Using-Neural-Network/blob/master/pic.png)"</a>

    
### A) Hardware Design
   The Hardware components used for this project are as follows:
    
   1. Raspberry pi-3.
   2. Pi Camera.
   3. Robot (Chassis, Wheels (4), Motors (2).
   4. Ultrasonic Sensor.
   5. Motor control (L293D).
    
   Raspberry pi will work as a brain of the robot, which takes all the decision of the robot and live streams video to the laptop. Ultrasonic sensor is used to calculate the distance of the obstacles ahead.
    



<img src="https://github.com/akshay1997feb/Self-Driving-Robot-Using-Neural-Network/blob/master/IMG_20180208_192412353.jpg" width="400" height="400">

<img src="https://github.com/akshay1997feb/Self-Driving-Robot-Using-Neural-Network/blob/master/pic2.png" width="400" height="300">



### B) Software Used:-
   1) Python(2.7)
   2) TensorFlow
   3) OpenCV

## Working
For detailed explanation about the model and working visit on the IEEE link given above.
* **Data Collection**
    * Images for training is collected by driving the robot.
    * Run File **"rpi_connection_cam.py"** on Raspberry Pi and simultaneouly run File **"collect_images_via_rpi.py"** on your PC. A server client connection is established between Rpi and your computer. Ensure that they are on the same Network.
    * Pygame is used for driving the robot, it is driven by using Keyboard (w-forward, a-left, d-right). When a key is pressed the image with its label is saved.
    * NumPy is used to save the data and its corresponding label.
* **Model Training**
    * Tensorflow is used to create the model.
    * After sufficient data is collected, run file **"train_neural_network.py"** on your PC (Ensure path where the model will be saved in code is correct).
    * Training the model will take some time and post completion the model will be saved.
* **Final**
    * Ensure that the previous steps are completed without any errors.
    * Run files **"rpi_connection_cam.py"** and **"rpi_controlcar_distance.py"** on raspberry pi simultaneously, parallely run file **"run.py"** on your PC.
 
 
 
 
# Credits <a href="https://github.com/hamuchiwa/">@hamuchiwa</a>, as I have followed his <a href="https://github.com/hamuchiwa/AutoRCCar">tutorial</a> and used some of his code.
