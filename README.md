# Design of Raspberry Pi for a car park entrance monitor system

**Author: Xiyu Wang**

**Time: 27/12/2029**

---

## Description of Artefacts

This Artefacts contains three **Python files**, one **README** file, and one MP4 file: **stableVideo_trim.mp4**. Three Python files:  **back_sub.py**, **back_sub_trim.py**  and **vehicle_counter.py**, 

* **main.py** represents the main class for this project, where Background subtraction is implemented.

* **detect_vehicle.py** represents the class where the algorithm for counting vehicles is implemented. 

* **model.py** represents upload the CNN classifier and do the predict

* **drew.py** represents display UML

* **ROI.py** represents select the region of interest.

-------------------------------------------------------------

## Download

In the Raspberry Pi, Raspbian is used for its operation system, which is similar to Linux.  

### On the Raspbian

1. Download Raspbian at https://www.raspberrypi.org/downloads/raspbian/. Then, choose Raspbian Buster with desktop. 

2. Use Win32DiskImager, which can be download at https://www.raspberrypi.org/Downloads/, to write the Raspbian to the SD card using your own laptop.

3. Then insert the SD card to the Raspberry Pi

*Note: Raspbian Buster with desktop is enough for this project. However, if you choose to download the Raspbian Buster Lite, you have to install many dependencies as well as other commonly used tools. You can refer to https://www.learnopencv.com/install-opencv-4-on-raspberry-pi/*

--------------------------------

## Setup

### For the software
To run this system, 
1. The Raspbian downloaded has its own inbuilt Python 3.6.6, which is compatibale with the OpenCV verison
2. Install OpenCV via command line 
    
    *sudo apt install python3-opencv*
3. Install TensorFlow

    *pip3 install --upgrade tensorflow*
    *pip3 install -q tensorflow-model-optimization*

### For the hardware
1. To connect the Raspberry Pi with the power supply

    a). Connect to the battery pack via BattBorg. Another side of the BattBorg is plugged into **GPIO 40 pin header**

    b). Connect to the portable charger via the **Power Connector** on the Raspberry Pi.

2. To connect the Raspberry Pi with the camera: connected the camera with the Raspberry Pi via **CSI Camera Connector**.

3. To Configure the Camera:

    a). Run *sudo raspi-config* in the terminal

    b). Choose **Interfacing Opetions**

    c). Choose **Camera** option in the menu of Raspberry Pi Software Configuration Tool.

    d). Choose **YES** to enable the camera interface.

4. Split the memory for GPU:

    a). Run *sudo raspi-config* in the terminal

    b). Choose **Advanced Options**

    c). Choose **Memory Split** to split the memory for GPU

    *Note: 128MB is minumum if the camera is needed*

------------------------

## Run
### Via Terminal
1. Use **cd** command to open the folder that contains this project.
2. Type **python3 main.py** to run the project.
3. If want to use the camera, change the **VIDEO_SOURCE = 0** in **main.py**.
-----------------------


