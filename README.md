Project Details
- state_size=33, action_size=4 as default UnityML - Reacher environment provides
- 20 arms environment, used shared actor + critic for controlling all arms
  - goal of environment is keep arms attached to moving target, positive reward for doing so
- Policy Gradients used, namely DDPG algorithm
- environment is considered as Udacity asks ~ mean of last 100 episodes during training > 30.
  - however this can be far off, as we train TARGET network. Aka i dont think it is reasonable measuring
  - especially when it comes to model itself change every x-timestamps, therefore question is : What we evaluating robustness of model to changes, or his capability to solve given task ? Appareantelly by rubric it is former case
  - Solution ? in this environment i skipped it and comply with rubric, however you can see my solution for Tennis, where i demonstrated that Former case of evaluation is really off, and difference between *evaluating* TARGET vs LOCAL network
- number of episodes to solve ? based on rubric it is 0, based on real ep executed about 90, based on real capability of target ~ i did not meassure should be 0in between 0~90
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
- if anything more needed in Readme please specify in more detail
