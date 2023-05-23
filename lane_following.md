---
title: Systems Integration and development of Autonomous Lane Following
---

Autonomous lane following feature development and TurtleBot systems integration using ROS II

| Contributors         | 
| -------------------- |
| Mohammad Zaki Patel  |
| Anirudh Varadarajan  |
| Harkirat Singh Chahal|

## Introduction

In this group project, we aim to solve the problem of path/ lane following for autonomous robots using neural networks and deep-learning computer vision techniques. The goal of the project is to develop a model that can predict the steering angle of the robot based on the input frames from the camera. The project uses TurtleBot hardware equipped with the Oak-d camera for collecting the data, and then the collected data will be used to train and test the deep learning model. 
Since the first draft of the project goals, some of the objectives have been modified to fit within the timeline and project scope. Initially, we were planning to change the location of the camera on the bot to improve the image quality. However, after a feasibility study, we realized that it was better to increase the height and lower the angle of the camera. This adjustment helped us to get a better viewing angle and improve the quality of the image data.
Regarding the data collection experiments, we initially planned to use multiple styles of paths to increase the diversity of the dataset. However, it led to a huge dataset that required more computational resources to process. To overcome this challenge, we decided to use one style of path with passes from both directions and implement data augmentation techniques. This approach helped us to collect diverse data and build a smaller dataset, which reduced the complexity of the data pipeline.
To collect the data, we controlled the TurtleBot with a keyboard and made it go back and forth multiple times on each path. During this process, we captured frame-by-frame images from the Oak-D camera and recorded the wheel angle as well as linear velocity values as labels for each frame. Linear Velocity is a new parameter added to the dataset as it will make it easier to control the robot autonomously with predictions. This gave us a manually recorded dataset that consisted of frame-by-frame path images and linear velocity and steering values as labels.


<figure><center>
  <img src="./final_img/project_flow.png" alt="Process Workflow" width="300" height="650">
  <figcaption><center>Figure 1: Project Process Workflow</center></figcaption>
</center></figure>

## Project Scope
The impact of developing a neural network-based solution for autonomous path following for robots is significant, as it can contribute to the development of more advanced and efficient autonomous robots for a variety of applications such as autonomous cars, drones, warehouse material handling robots, etc. Deep Learning techniques provides better learning opportunities for the robot to perform a task as compared to traditional hard coding. This project will drive us to learn new material related to deep learning and computer vision, as well as reinforce our knowledge of robotics and ROS. This project will encourage learning in topics such as systems integration, data collection, data preprocessing, machine learning model design, performance evaluation, which are essential skills for a end-to-end robotics projects with Neural Network application. It will also provide an opportunity to explore different neural network architectures and their applications in robotics. Overall, this project will help me develop a more comprehensive understanding and hands-on experience systems integration and machine learning applications in robotics.

## Data Collection
For the purpose of data collection, We used the Teleop twist keyboard to drive the turtle bot on the Track with the help of the keyboard. This program publishes preset value to /cmd_vel. Feed from the oak-d camera will be published in the topic /color/preview/image. The main objective of the ROS2 node is to subscribe to topics, receive data, and store it on disk. The node subscribes to two topics, /color/preview/image and /cmd_vel, and saves the received data to a local directory. The DataCollectionNode class is the main class that handles the data collection and storage. The constructor initializes the subscribers to the two topics. The cmd_vel_callback method is called whenever a new message is received on the /cmd_vel topic and updates the current_cmd_vel instance variable with the received linear and angular velocities. The image_callback method is called whenever a new message is received on the /color/preview/image topic, converts the image data to a numpy array, saves it to a PNG file in the data directory with a timestamp, and stores the timestamp, image path, and current linear_x and angular_z velocities to a CSV file named data.csv. This can be used as labels for training.


<figure><center>
  <img src="./final_img/data_collect.png" alt="Data Collection" width="600" height="270">
  <figcaption><center>Figure 2: Data Collection Process Workflow</center></figcaption>
</center></figure>


<figure><center>
  <img src="./final_img/path.png" alt="Data Collection Setup" width="600" height="270">
  <figcaption><center>Figure 3: Experiment Setup</center></figcaption>
</center></figure>


## Building the Model

1. The preprocess_image function is defined to load an image and convert it to a NumPy array of pixel values normalized between 0 and 1.
2. The process_images_parallel function is defined to preprocess multiple images in parallel using a thread pool executor.
3. An array of preprocessed images is created by calling process_images_parallel with an array of image file paths.
4. The labels are preprocessed by encoding them with LabelEncoder and converting them to one-hot-encoded arrays using to_categorical.
5. The data is split into training and validation sets using train_test_split.
6. The model architecture is defined using the Sequential API from Keras, which includes several convolutional and pooling layers, followed by dense layers.
7. The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.
8. The model is trained on the training set with a batch size of 32 and for 10 epochs, with the validation set used for monitoring model performance during training.


## Model Training 
1. Challenges in Image Quality and Trade-offs with Filtered Images for Optimizing Speed and Efficiency of a Model


When constructing the model, our main priorities were to optimize its speed and efficiency. Although our dataset was good, it was not flawless. Specifically, the quality of the images posed a challenge due to limited resources. We were unable to provide optimal lighting conditions, resulting in 30-40% of our images being washed out by the reflected light from the floor. Consequently, the track was not clearly visible in some of our images as can be seen in Figure 4 below.



<figure>
  <table>
    <tr>
      <td><img src="./final_img/canny_filter.png" alt="Canny edge filter" width="350" height="250"></td>
      <td><img src="./final_img/canny_filter.png" alt="Canny edge filter" width="350" height="250"></td>
    </tr>
  </table>
  <figcaption><center>Figure 4: Canny Filtered Dataset sample</center></figcaption>
</figure>

We decided to stick with raw images instead of the filtered ones to prevent loss of information. 

2. Challenges with Utilizing 'angular_z' and 'linear_x' as Labels for Image Classification Model 


At the outset, we had intended to utilize both 'angular_z' and 'linear_x' as labels for our images. The former parameter had three unique values that indicate the direction of movement, namely straight, left, and right. The latter parameter had two distinct values that represent the speed of movement - a constant speed when the Turtlebot moves straight and 0.0 when it comes to a stop to turn. Consequently, six possible combinations of 'angular_z' and 'linear_x' exist. However, when we trained the model using these two labels, the accuracy was disappointingly low, between 55% to 60%. 
We decided to simplify the model by just using ‘angular_z’ as labels to our images and hardcoding a constant speed for the bot instead of using ‘linear_x’, this improved our model accuracy to between 90-92% on the validation dataset.

3. Balancing the dataset


Although our dataset comprised 7324 images, it was not well-proportioned. As illustrated in Figure 3, the Turtlebot spent the majority of the time moving in a straight line, and the track only had three turns.


<figure>
  <table>
    <tr>
      <td><img src="./final_img/dataset.png" alt="Raw Dataset" width="350" height="250"></td>
      <td><img src="./final_img/balance_ds.png" alt="Balanced Dataset" width="350" height="250"></td>
    </tr>
  </table>
  <figcaption><center>Figure 5: Canny Filtered Dataset sample</center></figcaption>
</figure>

## Model Prediction Evaluation


<figure>
  <table>
    <tr>
      <td><img src="./final_img/metric.JPG" alt="train metric" width="450" height="250"></td>
      <td><img src="./final_img/validation_metric.JPG" alt="validation metric" width="450" height="250"></td>
    </tr>
  </table>
  <figcaption><center>Figure 6: Evaluation metrics of model training from scratch</center></figcaption>
</figure>


<figure>
  <table>
    <tr>
      <td><img src="./final_img/pre_metric.JPG" alt="Pre-trained training" width="450" height="250"></td>
      <td><img src="./final_img/pre_val.JPG" alt="pre-trained validation" width="450" height="250"></td>
    </tr>
  </table>
  <figcaption><center>Figure 3: Evaluation metrics of model training from pre-trained weights</center></figcaption>
</figure>

