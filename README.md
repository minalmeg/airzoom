# AirZoom

Control zoom with your fingers using hand detection by [mediapipe](https://google.github.io/mediapipe/solutions/hands.html). 

## Installation of Libraries

It's recommended that you use virtualenv so that existing packages aren't tempered with. You'll need Python 3 installed with pip. Run:

`pip install - requirements.txt`

## Usage

`python airzoom.py --p test.png`

###### Note : use of png images is preferred

## Results

Left Hand           |  Right Hand
:-------------------------:|:-------------------------:
![alt text](https://github.com/minalmeg/airzoom/blob/main/output/l_hand.gif "Left Hand") |  ![alt text](https://github.com/minalmeg/airzoom/blob/main/output/right_hand.gif "Right Hand")

## Future Work

[Use of GANs to generate a 3D zoomable image] (https://blogs.nvidia.com/blog/2021/04/16/gan-research-knight-rider-ai-omniverse/)

## References
[Murtaza's Workshop - Robotics and AI](https://youtu.be/9iEPzbG-xLE)
[Transperancy adjustment of alpha channel](https://gist.github.com/clungzta/b4bbb3e2aa0490b0cfcbc042184b0b4e)
