---
title: "3D-Object Detection for autonomous vehicles"
date: 2019-12-14
excerpt: "Project on 3D Object Detection using Lyft's level5 dataset: ranked in the top 20% in a live kaggle competition"
mathjax: "true"
comments: true

header:
    image: /posts/lyft.jpeg            # Twitter (use 'overlay_image')
    overlay_filter: 0.15
    teaser: /posts/lyft.jpeg   # Shrink image to 575x216
    caption: "Photo credit: [**Driving vision news**](https://www.drivingvisionnews.com/ireds-for-face-recognition-from-everlight/)"

---


Journey with 3D object detection using Lyft’s Level 5 Dataset

![Source: [https://www.drivingvisionnews.com/ireds-for-face-recognition-from-everlight/](https://www.drivingvisionnews.com/ireds-for-face-recognition-from-everlight/)](https://cdn-images-1.medium.com/max/2880/1*QKKoMVg2U5HjClRJgd8bIg.jpeg)*Source: [https://www.drivingvisionnews.com/ireds-for-face-recognition-from-everlight/](https://www.drivingvisionnews.com/ireds-for-face-recognition-from-everlight/)*


In this blog post, I share with you our learnings from a [Kaggle competition by Lyft](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles). Let me take you through our journey.

> **Abstract**: With the rapid evolution in the world of technology, it becomes increasingly important for the automotive sector to keep up with the fast pace, just like any other industry. In recent times, self-driving cars have gained a lot of traction but there is a huge gap in expectation and the current state. On those lines, our project focuses on 3D Object Detection of Lyft’s autonomous vehicles.

## **Why you don’t have an autonomous car yet?**

Self Driving Vehicles are one of the most hyped technologies of the modern decade. Even though many companies may brand their driver assistance technology as “Autopilot” a truly self-driving vehicle, without a human driver on open roads, has not yet become a reality.

![Source: [http://post.toutptit-toutbio.com/page-3/self-driving-car-timeline-28811.html](http://post.toutptit-toutbio.com/page-3/self-driving-car-timeline-28811.html)](https://cdn-images-1.medium.com/max/2200/1*FaHyQ7EO6qvoFPYQY5tOrg.png)*Source: [http://post.toutptit-toutbio.com/page-3/self-driving-car-timeline-28811.html](http://post.toutptit-toutbio.com/page-3/self-driving-car-timeline-28811.html)*

Currently, vehicles that are labeled as “Autonomous” are Level 4 vehicles. There are 5 levels of self-driving cars:

1. The lowest level is Level 0, in which a vehicle has no driver assistance technology. It also constitutes a majority of vehicles on US roads.

2. Level 1 is the level in which the vehicle is equipped with driver assistance technology such as a blind spot monitoring system (BLIS), adaptive cruise control (ACC), and automatic emergency braking (AEB). It comprises of a majority of new cars sold, and has active or inactive systems but only intervene when they detect another vehicle.

3. Level 2 vehicles can control steering, acceleration, and braking and include systems such as GM’s Supercruise or Tesla’s Autopilot system.

4. Level 3 systems such as Audi Traffic Jam Assist are completely autonomous below 37 mph.

5. Level 4 vehicles are the current vehicles being tested around the world including Lyft, Uber, and Waymo vehicles. These vehicles are autonomous but must have a human driver available for difficult driving situations.

6. Level 5 comprises of completely autonomous vehicles with no human intervention, of which the most well known is the Google car, unveiled a few years ago. However, even this car is only able to operate on pre-defined routes and hence, can’t be termed truly autonomous.

Given that there are many levels to evaluate self-driving vehicles, it leads us to wonder why there are no true self-driving vehicles on sale now at your local dealership (or online if you are Tesla). This is because of two main technological issues:

1. **Perception**: Identification and classification of objects around the vehicle

2. **Prediction**: Determination of the future position of predicted objects

Since perception was the cornerstone issue behind the slowed development of self-driving vehicles, I decided to focus our effort to improve perception around a vehicle.

## **Perception Problem**

Although perception is a primary issue, there is another reason that I chose to focus on perception over prediction in our project. A truly autonomous vehicle is projected to be a safer alternative to human drivers. Despite this, there has been one death related to a self-driving vehicle. The death was caused by an Uber self-driving car striking and killing a pedestrian because the vehicle was unable to sense the person before it hit them. Perception, therefore, is more than just interesting, it is a life-or-death issue relating to self-driving vehicles.

![Source: [https://www.theverge.com/2018/6/22/17492320/safety-driver-self-driving-uber-crash-hulu-police-report](https://www.theverge.com/2018/6/22/17492320/safety-driver-self-driving-uber-crash-hulu-police-report)](https://cdn-images-1.medium.com/max/2400/1*O9LmZN-SLGVcO2ThExVpag.png)*Source: [https://www.theverge.com/2018/6/22/17492320/safety-driver-self-driving-uber-crash-hulu-police-report](https://www.theverge.com/2018/6/22/17492320/safety-driver-self-driving-uber-crash-hulu-police-report)*

After seeing that perception is an extremely important issue, I decided to research into how vehicles are actually able to perceive the objects around them since they don’t exactly have eyes like humans. There are actually four different technologies enabling a car to see objects around it. The four types of sensors are ultrasonic, lidar, radar, and cameras. Of these, lidars and cameras are the most important sensors since ultrasonic sensors are mainly used in parking sensors and radar is only used for extremely long-distance tracking used for ACC.

In modern systems, cameras are generally only used to find lane markings and to display signs such as speed limits on the dashboard of a vehicle. On self-driving vehicles, cameras can be an additional input into perception algorithms, although it is complex.

Lidar sensors, on the other hand, is the main way that vehicles sense objects around them. Lidar, a modern update of sonar technology, finds objects by shooting millions of lasers, light beams, and finding the reflections of those lasers on objects. Behind the sensor, there is a computer that will make an almost instantaneous 3D map of the area around the vehicle called a point cloud, which will be discussed later. The map is generally low resolution, it does not capture extremely detailed information, but it does give the general shape of objects around the vehicle. Now that I knew about how perception works in cars  I looked into which companies were working in this space.

## **Why Lyft?**

Of the various companies competing in the autonomous vehicles space,  I chose to use Lyft because it believes in an open-source platform for the development of their self-driving vehicles. Based on the belief that no one person or company can solve self-driving technology, Lyft partners with Aptiv and Waymo to collaborate on the development of perception models for self-driving vehicles. Based on this open-source philosophy the company also sponsors Kaggle competitions.  I took part in one of their active [competitions](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles) where the first place winner with the best perception model using Lyft’s data, could win $25,000. Let’s dive into the dataset  I used.

## **The Dataset**

The Lyft dataset from the active Kaggle competition was a total of 85 GB. The data was split between testing and training sets and included a sample submission. The dataset gives a 3D point cloud and camera data from the Lyft test vehicles.

Our data was captured by 10 host cars on the roads of Palo Alto, California. Each of the host cars has seven cameras and one LiDAR sensor on the roof, and 2 smaller sensors underneath the headlights of the vehicle.

![Source: [https://mobiag.com/blog/lyft-unveils-a-new-self-driving-car/](https://mobiag.com/blog/lyft-unveils-a-new-self-driving-car/)](https://cdn-images-1.medium.com/max/2000/1*qQ1hDRz2gBE0CZxtVqKbEg.jpeg)*Source: [https://mobiag.com/blog/lyft-unveils-a-new-self-driving-car/](https://mobiag.com/blog/lyft-unveils-a-new-self-driving-car/)*

Each of the seven cameras will capture images of its surroundings at different fixed angles. These images have 55,000 human-labeled 3D annotations on objects, such as, cars, pedestrians, bicycles, which are the orange boxes in the image below.

![Rendered image of a scene](https://cdn-images-1.medium.com/max/2332/1*_AO6WlMfRuermgWdN6qHlQ.png)*Rendered image of a scene*

Each LiDAR sensor will shoot lasers 360 degrees to detect objects and get 3D spatial geometric information. These LiDAR sensors produce a point cloud each with 216,000 points at 10Hz. The data was captured across 13 files which served as inputs to our model. Let us see what the data comprised of, in detail:

* Sample_data is the data collected from a particular sensor in the car.

* Sample_annotation is an annotated instance of an object of interest.

* An instance is an enumeration of all object instances Iobserved, such as a labeled truck on the image.

* Categories are different types of objects observed. Examples include cars, pedestrians, etc.,

* Ego_pose is the record of the vehicle at a certain point in time which could be mapped for each sample.

* Map data is a binary semantic map of the roads around the vehicle

* Calibrated_sensor is the definition of a particular sensor as calibrated on a particular vehicle. It provides information about the environment, such as the distance and bearing to a feature, or directly measure the sensor’s position and orientation (pose). The intrinsic properties carried by calibrated_sensor are those that do not depend on the outside world and how the sensor is placed in it. The property and distance matrix will be used to combine three-point clouds produced at similar timestamp and map 3D bounding boxes.

* An attribute is a property of an instance that can change while the category remains the same.

All of these features along with the lidar and image data were split into training and testing sets and made up the large dataset for the competition.

## **The Approach**

After Ihad a sense of what the 85 GB of Lyft data comprised of, I worked on the lidar data to find some general insights before using it in our model. Since  I was unable to download the entire dataset to our local machines  I decided to use Google Cloud Platform (GCP) to import our data into an instance and run our exploratory data analysis and model building on GCP. These are the [steps](https://github.com/KezhenY/Lyft_3D_Object_Detection_for_Autonomous_Vehicles/blob/master/GCP%20commands.txt) of how  I set up GCP.

## **Exploratory Analysis**

After  I imported the data using the Kaggle API onto GCP  I explored the Lidar data.

![Source: [https://www.mathworks.com/help/driving/ug/coordinate-systems.html](https://www.mathworks.com/help/driving/ug/coordinate-systems.html)](https://cdn-images-1.medium.com/max/2000/1*9AR84gK3jXRZmqukKUIAqA.png)*Source: [https://www.mathworks.com/help/driving/ug/coordinate-systems.html](https://www.mathworks.com/help/driving/ug/coordinate-systems.html)*

The primary measures captured by the LiDAR are x,y,z coordinates of an object along with its length, width, height and yaw as shown in the visualization above. The metrics from the LiDAR sensors are also listed below with definitions for easy interpretation.

1. **centre_x, centre_y, and centre_z**correspond to the coordinates of an object’s location on the XYZ plane.

2. **yaw** is the angle of the volume around the *z*-axis, making ‘yaw’ the direction the front of the vehicle/bounding box is pointing at while on the ground.

3. **length, width, and height**represent the bounding volume in which the object lies.

After  I found what measurements the LiDAR sensor gave us  I plotted distributions of the various measurement metrics to see if  I can gather insights from the plots.  I started with the **x,y, and z measurements **then continued to the other variables.

The distributions of center_x and centre_y bring out the limitations of the LiDAR cameras in our analysis. The car’s cameras capture the objects more easily to its side than the front and back. This is because objects on the side of the car are more clearly detected by LiDAR and captured by the camera. In contrast, objects in the front and the back of the car are normally in line with cars, therefore, it is harder to be clearly detected. Closer objects on the side of the car are much more likely to be detected than objects in the x-axis, and objects that are far on the side of the car are less likely to be detected than in the x-axis. One more limitation is that objects that are either far in the x-axis or y-axis can be better detected than if objects are far in both x, y-axis.

![Frequency count of the classes](https://cdn-images-1.medium.com/max/2000/0*yKJGS-0zwV6S1oCy)*Frequency count of the classes*

As  I can see from the above plot, most of the objects that are observed are cars. While this can be a true scenario as there is generally more number of cars on the roads when compared to other vehicles, this could also reflect the limitations of the LiDAR sensor.

## **Data Preprocessing and Transformation**

Since  I was only using the lidar data as the starting point of our neural net,  I did not explore the image data or the other data any further. At this point,  I had explored all of our lidar data but found that since there are three different lidar sensors on the vehicle the data could not go directly into a neural network since there were duplicate parts of the surroundings captured by the three sensors. Because of this duplicate area problem,  I transformed our data and overlayed similar parts of the lidar points with each other.

Since no human can interpret the lidar point cloud overlap with their eyes  I decided to use the image data and find a way to overlay the information in the image with the lidar point cloud.

The source of the scene data is from the 7 cameras and 3 LiDars placed on the car. These images were recorded at different angles at a similar time. Information carried by point cloud captures surrounding objects in the form of x, y, z from the location of the sensor. However, each of these 3 sensors has its own coordinate system. To superimpose the data from all the 3 sensors to form a consolidated point cloud,  I will have to transform each of these inputs.

![Transformation of frame of reference](https://cdn-images-1.medium.com/max/2688/1*WZYi0BUOo3Hlmr6kvnCrHg.png)*Transformation of frame of reference*

To achieve this,  I can look at the collinear relationship between the 2D pixel coordinates and the 3D point cloud coordinates of LiDAR. This is expressed in the form of a transformation matrix which consists of a translation vector and rotation matrix.

![Data transformation matrix](https://cdn-images-1.medium.com/max/2000/1*i8qcPVzloiX5cXs4fLLKng.png)*Data transformation matrix*

Where x,y,z makes up the translation matrix and P denotes the rotation matrix. The rotation matrix is derived from the intrinsic properties of the cameras and sensors which is obtained from the calibrated_sensor input file.

Now that I have a background of the math concepts, our first step was to extract the lidar tokens for each scene and for each sensor. [The lyft_dataset_sdk](https://github.com/lyft/nuscenes-devkit) package was very useful in extracting and combining the data. Using the concept discussed above,  I implemented the matrix transformation in python using the Quarternion Package. First, the lidar data was transformed from the sensor’s frame of reference to the car’s frame of reference, post which the car’s frame of reference was transformed into the world’s frame of reference to create a bird’s eye view (BEV).

The next step is to voxelize the output which was a list of coordinates to an XYZ space. After this step,  I created the bird’s-eye view (BEV) point cloud by normalizing the voxel intensities to obtain inputs of 336 x 336 dimensions consisting of 3 channels that describe the differences in height of the lidar points.

![Transformation of lidars points into BEV](https://cdn-images-1.medium.com/max/3392/1*2nkstQWr2D5GvmfuK3BQ3Q.png)*Transformation of lidars points into BEV*

## **Method Selection**

After preprocessing our data to create the BEV for each lidar input  I analyzed different neural networks to see which was best for our 3D object detection problem.

To achieve 3D object detection,  I require both segmentation as well as localization. Within segmentation, there is instance segmentation and semantic segmentation.  I used Semantic segmentation since  I did not need to locate different instances within a class but only if a given object was a certain class. Localization is the method that allows us to pinpoint where a certain object is inside a lidar point cloud.

After deciding on semantic segmentation  I moved on to decide which method our neural network would use for object detection. There are many methods to model 3D data, however, not all methods can always be applied to all types of 3D data as they take different forms for different results.  I went through four different methods but found disadvantages with a few approaches for 3D object detection before selecting our final method for object detection.  I will quickly go through these four approaches, along with their disadvantages.

* Multiview CNNs render 3D point clouds into 2D images and then apply 2D convolutional neural networks to classify them. However, extending it to point classification, scene understanding and shape classification isn’t easy.

* Spectral CNNs which are used on meshes are still not equipped to handle non-isometric shapes such as furniture, cars, or people.

* Feature-based DNNs extract shape features, after converting the 3D data into a vector and use a fully connected net for classification. But the features don’t accurately represent the actual information.

Our model makes use of volumetric CNN. It uses 3D convolutional neural networks on voxelized 3D shapes. The main disadvantage of this is the lack of resolution due to data sparsity and computational cost. However, since I was dealing with objects which are huge,  I believe that if  I can fit a bounding box around the object, the model should be fine. Now that  I had decided on volumetric CNN  I decided to go ahead with U-Net.

## **Modeling**

The U-Net is a convolution neural network developed by [Olaf Ronneberger](https://arxiv.org/search/cs?searchtype=author&query=Ronneberger%2C+O), [Philipp Fischer](https://arxiv.org/search/cs?searchtype=author&query=Fischer%2C+P), [Thomas Brox](https://arxiv.org/search/cs?searchtype=author&query=Brox%2C+T) for Biomedical Image Segmentation. It takes its name from its symmetric architecture. U-Net architecture consists of three main parts:

1. The downsampling or the contracting path

2. The bottleneck

3. The upsampling or the expanding path

![Source: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)](https://cdn-images-1.medium.com/max/3110/1*lvXoKMHoPJMKpKK7keZMEA.png)*Source: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)*

The contracting path consists of four blocks, each of which has two convolutional layers and one max-pooling layer. The number of features doubles after each block, capturing contextual information in the images. ie, this is where the network captures the “What” information. The bottleneck has two convolutional layers with dropout. Similar to the contracting path, the expanding path also has four blocks with convolutional layers and concatenation with the cropped feature maps from the contracting path. In the expansion path, the depth of the image decreases while the size of the image gradually increases. This is how  I capture the “Where” information in the images.

Such an architecture allows us to do fast and precise segmentations.  I input the top-down projection of the world around the car into our U-Net architecture to create semantic maps. The result from the architecture is the semantic map in the BEV. Let’s look at a sample data point and its corresponding processed target.

![Input (left) and processed target (right)](https://cdn-images-1.medium.com/max/2000/1*9pWZfLAo0BsbFK2zAQ5nbw.png)*Input (left) and processed target (right)*

These inputs were passed on to the network to create the target on the right. Now  I know the input to the model, but how is the processed target generated?

The output from the network is the region of interest from the BEV. As you can see from Fig. A below, not all the predictions are ground truths.

![Fig.A Visualization of predictions. Fig.B) Thresholded predictions Fig.C) Processed targets](https://cdn-images-1.medium.com/max/2476/1*tbFyzZmNf2GnWUL1Hf5B3A.png)*Fig.A Visualization of predictions. Fig.B) Thresholded predictions Fig.C) Processed targets*

A pixel value of 127.5 (in our case) is used as a threshold on our predictions to create binary target variables.

## **Evaluation metric**

The average precision is calculated at different thresholds of Intersection over Union (IoU) to evaluate the object detection model. The IoU of a 2D bounding box is the area of the overlapping region divided by the total area of union. The IoU is calculated at thresholds starting from 0.55 to 0.95 with a step size of 0.05.

![Source: kaggle](https://cdn-images-1.medium.com/max/2000/1*1mpgVEXsk9HO56fVpRIG9w.png)*Source: kaggle*

For example, the object is a prediction at a threshold of 0.55, if the IoU is greater than 0.55. At each such threshold, a precision value is calculated using the True Positives (TP), False Positives (FP) and False Negatives (FN) by comparing the predicted object to all the ground-truth objects. The precision value is the proportion of true positives to false positives and true positives. A true positive (TP) is counted when the network correctly predicts the presence of the object. If the predicted object has no associated ground truth, then  I count such a prediction as false positive (FP). To evaluate the model,  I calculate the mean precision of the above precision values over different thresholds using the below equation:

![mean absolute precision (mAP)](https://cdn-images-1.medium.com/max/2000/1*Xeu_GM2Poe94ZO0ruBdZ0w.png)*mean absolute precision (mAP)*

In a 3D context,  I also evaluate the overlap in the Z-axis while calculating the average precision. ie. the one-dimensional intersection over union is calculated for the z-axis (height of the object).

## **Phase 1 of training**

The competition journey is divided into two phases: the one before the ‘AHA’ moment and the one after. As explained earlier the input to our network was the BEV of the world around the car. The basic model that was trained for 10 epochs scored around 0.034 on the leaderboard. As this was a Kaggle competition, I was training our model twenty-four times seven and submitted the results to Kaggle. One of the main challenges was the training time: thirty epochs took around seventeen hours and I was limited to making only one to two changes per day. The few changes that mattered and boosted our score on the leaderboard are:

1. Reducing the weight of the car’s loss to 1/5th of the other classes while training to account for the class imbalance.

2. Since  I are using BEV as the input, the height of the object is not captured. The individual heights were replaced with the conditional mean heights of the classes they belonged to.

3. Along the way, a lot of parameters were changed including the batch size, learning rate, optimizer (Adam & Rectified Adam — no, not Adam’s Family!) and so on.

After this phase, the model scored 0.039 on the Kaggle public leaderboard. The network that resulted in this score was an ensemble of three models with epoch variations — eight, nine and ten.

## **Phase 2 of training**

 I extracted the map mask around the corresponding ego region from the dataset and used it along with the BEV of the lidar point as three additional channels.

![Fig A) BEV, Fig B) map mask, Fig C) Processed target](https://cdn-images-1.medium.com/max/2440/1*hmQrcSB8rZ-2C2Xxf3FE1g.png)*Fig A) BEV, Fig B) map mask, Fig C) Processed target*

 I can see the map mask in Fig B) of the above image. The map mask is the BEV of the roads and side paths. This gave us a boost in the leaderboard score — to 0.040.  I tried tuning various aspects including learning rates, epochs, batch sizes, optimizers and so on. One of the biggest challenges was that, even in GCP,  I could not do cross-validation due to memory limitations. This had a huge impact on our journey.

## **Results**

Among all the combinations  I tried, the model that worked the best was an ensemble of five models from epochs eight, nine, ten, twenty-nine and thirty. As you can see from the loss vs epoch plot on the left, the loss stabilizes after a certain epoch.

![epoch vs loss for a few models  I trained](https://cdn-images-1.medium.com/max/2000/0*JN8etvf1HhoyH1Ux)*epoch vs loss for a few models  I trained*

Let’s look at a few of the predictions from the final network  I created. The predictions from the model were overlaid on the map masks as well as on the BEV.

![Predictions on BEV (left) and map mask (right)](https://cdn-images-1.medium.com/max/2000/1*1ao1ohKLoGBUhHdha0dxPg.png)*Predictions on BEV (left) and map mask (right)*

As  I can see, there are a lot of red spots in the top-down projects. Those are the regions where the model is predicting the presence of an object, but there isn’t any. The green area is the region where there is an object, but the model fails to predict an object. This is especially true for extremely small objects, such as pedestrians. Finally, the yellow-colored spots are the regions where our model matches the ground truth — True positives! The final model scored 0.046 on the public leaderboard (Using 30% of the test data) and 0.045 on the private leaderboard (Using 70% of the test data). After the competition ended on November 12th, 2019,  I secured a place in the top 20% of teams that participated in the competition.

![Final leaderboard score](https://cdn-images-1.medium.com/max/2854/1*DrGwO89YtyFzXep7SLcSsw.png)*Final leaderboard score*

Here is a video  I made using the predictions from our final model:

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/5LN6mFjK6go" frameborder="0" allowfullscreen></iframe></center>

## **Challenges**

 I have learned quite a bit from this project and have had a lot of challenges along the way. Few of them are:

*  I could not do cross-validation due to memory constraints, even though  I was using the Google Cloud Platform.

* This also did not allow increasing the batch size beyond 32 during training

* Our training times for models were extremely long: 30 Epochs took around 16–18 hours approximately to train a batch size of 16, around 30–40 minutes to ensemble 3–4 models. This sums up to 20 hours for one submission.

## **Learnings**

Even though our learning curve through the journey was steep, our model still has quite a few limitations. It fails to predict the small objects from the BEV such as pedestrians, cyclists, etc., This is because one pixel in the BEV is a 40cm x 40 cm box in the real world. So,  I think  I could use the other available data (especially the images) to capture such objects. Here,  I have used only one Lidar Sweep, but  I can also think of using more than one! One other limitation is that the model assumes that all objects are the same height as that of the ego vehicles. This automatically adds to the error in the data.

## Conclusion

Given the above,  I believe that our biggest constraint was computational power.  I later came to know from our Professor Dr. Joydeep Ghosh that he has multiple GPUs set up in his lab but still finds it tough to tune these models (for his research paper on a similar topic). Well, that gave us some solace!

As one of the next steps,  I could try out other model architectures. Hopefully, the architectural variations along with our increased computational power should land us better accuracy, thus capturing the tiny pedestrians better!

## **References**
[**Lyft 3D Object Detection for Autonomous Vehicles**
*Can you advance the state of the art in 3D object detection?*www.kaggle.com](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles)

[**Reference Model**
*Download Open Datasets on 1000s of Projects + Share Projects on One Platform. Explore Popular Topics Like Government…*www.kaggle.com](https://www.kaggle.com/gzuidhof/reference-model)

[**U-Net: Convolutional Networks for Biomedical Image Segmentation**
*There is large consent that successful training of deep networks requires many thousand annotated training samples. In…*arxiv.org](https://arxiv.org/abs/1505.04597)

[**lyft/nuscenes-devkit**
*Welcome to the devkit for the Lyft Level 5 AV dataset! This devkit shall help you to visualise and explore our dataset…*github.com](https://github.com/lyft/nuscenes-devkit)

[**IREDs For Face Recognition From Everlight - DVN**
*Everlight have released three new infrared emitters for automotive applications. Their VS-FI3535 series is suitable for…*www.drivingvisionnews.com](https://www.drivingvisionnews.com/ireds-for-face-recognition-from-everlight/)

[**Coordinate Systems in Automated Driving Toolbox - MATLAB & Simulink**
*Automated Driving Toolbox™ uses these coordinate systems: World: A fixed universal coordinate system in which all…*www.mathworks.com](https://www.mathworks.com/help/driving/ug/coordinate-systems.html)

[**Understanding Semantic Segmentation with UNET**
*A Salt Identification Case Study*towardsdatascience.com](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47)


## Project repository & Kaggle Profile

[**sijopkd/3D-Object-Detection-Lyft**
*You can't perform that action at this time. You signed in with another tab or window. You signed out in another tab or…*github.com](https://github.com/sijopkd/3D-Object-Detection-Lyft)

[**Sijo VM | Kaggle**
*Download Open Datasets on 1000s of Projects + Share Projects on One Platform. Explore Popular Topics Like Government…*www.kaggle.com](https://www.kaggle.com/sijovm)
