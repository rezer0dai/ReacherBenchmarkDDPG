all informations are provided in .ipynb
- for actor i used NoisyNetworks for exploration
- i used batchnorm
- used DDPG
- well there are many improvements ~ see wheeler framework for example ( lot of different features and things to tune ~ GAE, RNN/CNN, globalnormalizer, network sizes, neural network architecture, RUDDER approah, etc etc)

- How to run the code : run Report.ipynb

- details : 
- for actor i used NoisyNetworks for exploration
- i used batchnorm
- network sizes 400 + 300, critic is reused from DDPG algorithm, learning rates actor i get bigger 1e-3 than critic 1e-4
- updating network every 40*20 steps and repeat learning 20*20 times, postponed update for 7 learnings
- used 3-step estimator, and Advantage as TD-learning, batch size 256, and buffer size 1e6, rerandomizing noisy nets for explooration every 3rd step
- one actor one critic ( shared for every arm )
- used Prioritized Experience Replay

weights to trained network : https://github.com/rezer0dai/UnityMLEnvs/tree/master/models/reacher/benchmark