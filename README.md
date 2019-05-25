# Visual Reward for Autonomous Driving

All in one package for demonstrating visual reward in reinforcement learning with Carla driving simulator. This code is used in bachelor's thesis Visual Reward for Autonomous Driving (http://bsc.laurialho.fi).

## How to install

Download and install all used progams and libraries, which are listed beneath. After that, copy the files from this repository into Carla folder.

## How to run

1. Start Carla server with arguments listed in sub section 'Start arguments for server'.
2. Create demonstration videos for visual reward. Create them by driving the route with make_demonstration.py.
3. Run main.py script with arguments. You can take a look of start arguments with --help command. Default arguments also exist.

## Used programs and libraries

Precompiled Carla 0.9.4 for Windows (http://carla.org/2019/03/01/release-0.9.4/)

python==3.7.0 
keras==2.2.4 
numpy==1.16.2 
tensorflow-gpu==1.13.1 
sklearn==0.0 
opencv==3.4.2 

numpy==1.16.2 
matplotlib==3.0.3 
sklearn==0.0 
pygame==1.9.6 

## Start arguments for server

CarlaUE4.exe /Game/Carla/Maps/Town03 -windowed -ResX=960 -ResY=960 -benchmark -fps=60 -carla-server -carla-settings="settings.ini"

## Tested environment

Tested to work with following setup:

Windows Server 2019, 
Intel Core i7-7820X, 
64 GB RAM, 
2x GTX 1080 Ti,
512 GB NVME SSD.

## License 

Take a look of license file.
