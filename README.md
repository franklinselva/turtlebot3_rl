# turtlebot3_rl

This repository contains the implementation for mobile robot navigation using RL algorithms. The planned algorithms to be tested are Q-learning, SARSA and extended-SARSA. The work contains a total of 6 categories: implementation of above original algorithm with HITL. The environments are tested with static obstacle but should be fine with dynamic obstacles.

## INSTALLATION

We will need the turtlebot3 and turtlebot3_simulation packages for these to work. Installation procedure are presented in their respective github pages.

## PROCEDURE

## Turtlebot3 Navigation

The commands are concentrated on the simulated robot model but it stays the same for real robot. Initially an environment variable need to set the packages to a specific robot. Turtlebot3 Burger is used and the command to set is as follows.

To enter the bash script of the linux terminal, run

```sudo nano ~/.bashrc```

At the end of the document, add

```export TURTLEBOT3_MODEL=burger```

Save the document and reopen the terminal window. 
Once the robot model is set, to launch the gazebo environment with Turtlebot3 robot model,

```roslaunch turtlebot3_gazebo turtlebot3_world.launch```

To start the navigation, run

```roslaunch turtlebot3_navigation turtlebot3_navigation.launch```

## Turtlebot3 Reinforcement Learning

The package can be engaged assuming the initial setup has been carried out, the procedure is as follows:

To start Q-learning agent,

```roslaunch turtlebot3_rl q_learning.launch```

To start SARSA agent, 

```roslaunch turtlebot3_rl sarsa.launch```

To start SAC agent,

```roslaunch turtlebot3_rl sac_discrete.launch```

To allow the above agents to use HITL, a parameter is set in the rosparameter server from the launch file

```<param name-“use_hitl” value=”true” />```
