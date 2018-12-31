Project Details
- state_size=33, action_size=4 as default UnityML - Reacher environment provides
- 20 arms environment, used shared actor + critic for controlling all arms
  - goal of environment is keep arms attached to moving target, positive reward for doing so
- Policy Gradients used, namely DDPG algorithm
- How to install :
  - environment itself : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
    - unpack project to ./data/ folder inside this project
  - install anaconda : https://www.anaconda.com/download/
    - this could come with preinstalled numpy as well
  - then follow : 
  ```
  conda install -y pytorch -c pytorch
  pip install unityagents
  pip install matplotlib
  ```
  - then replicate my results by running Report.ipynb
  - and you can download my pretrained weights : https://github.com/rezer0dai/UnityMLEnvs/tree/master/models/reacher/benchmark
