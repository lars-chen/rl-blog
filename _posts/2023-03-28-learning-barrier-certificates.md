---
title:  "Learning Barrier Certificates in Higher Dimensional Mujoco Environments"
permalink: posts/2023-03-28-learning-barrier-certificates
mathjax: true
layout: post
categories: media
author: Lars Chen, Jeremiah Flannery
giscus_comments: true
date: 2023-03-28
---

#### Authors: *Lars Chen*,  *Jeremiah Flannery*

<a name="abs"></a>
## Abstract

Having a barrier function verify safe states is an often-used strategy to guarantee that one doesn’t incur training-time errors in Safe RL. Depending on how one sets up this barrier function, it can require effortful hand-tuning specific to any new environment.  Last year, Luo and Ma proposed a method that sidesteps this effort by co-learning three elements: 1) improving the confidence of the physics model, 2) increasing the size of verified regions, and 3) optimizing the policy. They posit that any of the three elements will incrementally improve after benefitting from improvements in the other two elements, creating a complimentary sequential structure. Instead of requiring a pre-made barrier function, their algorithm now requires an initial safe policy as a starting point. They showed in simulations with low dimensional environments that their algorithm was capable of expanding the safe region while incurring no training errors. We introduced the algorithm into two environments with higher dimensionality: double inverted pendulum and hopper, and we performed an analysis on the safety of pre-trained agents in the two environments. We found that pretraining achieved safety in less steps than expected, but the behavior of the agent when training the barrier certificates did not arrive to a safe behavior in the number of epochs we were able to train. 

<!-- pagebreak -->

## Table of Contents
  * [Abstract](#abs)
  * [Introduction](#intro)
  * [What are barrier certificates?](#bc)
  * [How do barrier certificates fit into the CRABS algorithm?](#lbc)
  * [Environments](#env)
  * [Pre-training](#pre)
  * [Discussion](#disc)
      - [Double Pendulum with CRABS](#double)
      - [Hopper with CRABS](#hopper)
  * [References](#ref)


<a name="intro"></a>
## Introduction

In reinforcement learning (RL), an agent is trained to navigate an environment and maximize its reward using a function crafted by a human investigator (Sutton and Barto, 2018). [^SuttonBarto2018] The simplest classical algorithms maintain expectations of rewards in different states and update them after taking actions. For example, an instantiation of Temporal Difference Learning (Sutton, 1988)[^Sutton1988] maintains a table of expected values $$V$$ for states $$s \in S$$ and actions taken with policy $$\pi$$. It updates a state’s expectation with the difference between the previous expectation and the current reward, as well as a discounted (multiplied by a $$\gamma < 1$$) future reward term. 

$$
\Delta V(s) = \alpha (r + \gamma V(s') - V(s)) 
$$

Modern, more sophisticated versions of RL have been shown to complete high-dimensional tasks in a variety of environments, including robotic simulation [^Akkaya2019]. In real world applications of RL, such as biomedical robotics, low reward areas could be states that hurt the patient or damage the agent itself. One can imagine how in TD Learning one must visit several unsafe states and thus incur unwanted consequences. Safe RL tries to learn a high reward policy while either maintaining integrity of the agent or not violating external constraints.

There are already many unique and diverse strategies to accomplish the goal of safety. In a review written in 2013, García and Fernández [^GarciaFernandez2015] divided safe RL algorithms into two main genres: algorithms that modify the optimality criterion with a safety factor (like primal-dual problems which add penalty terms to the Lagrangian) and algorithms that modify the exploration behavior. Nearly ten years and many new Safe RL methods later, Gu et al, [^Gu2023] chose to make the main distinction between algorithms in whether they have a dynamics model that predicts not only rewards, but future states (model based) or not (model free.)

$$
T : S \times A \Rightarrow S 
$$

The above dynamics model from CRABS maps a state-action pair onto a distribution of state spaces.


CRABS falls firmly into being model-based and modifying exploration behavior. Before going into details of CRABS we will mention other strategies that exemplify the diversity of the categories above. One sub-area of staying within constraints is Trust Region Policy Optimization, which guarantees policies do not make too large of leaps by requiring new policies have a low KL-Divergence from an old safe policy.[^Schulman2017] Grown out of Trust Region methods is Constraint Policy Optimization [^Achiam2017] (CPO), one of the most well-known recent model-free developments in RL. It is a gradient method that optimizes a constraint cost, $$J$$,  for a policy update. CPO ultimately learns to approximate the constraint cost by encountering unsafe states and comparing these to safe states. The actor-critic method, which isolates the reward function and value function to separate agents has also been combined with CPO, again needing to encounter unsafe states but yielding even higher rewards.[^Balakrishna2017] 

Some “teacher” algorithms take human input to guide the agent to learn faster and avoid bad states.[^Zimmer2014] Other classes of algorithm modify the exploration behavior by first learning from batches of existing data[^Le2019] or by training beforehand in a sandbox environment where healthy errors can be made before training in the “real” environment where fatal errors would occur.[^Zhang2020] CRABS needing a pre-trained safe policy in some ways parallels this approach. Depending on the researcher, these methods may or may not incorporate a dynamics model. 

Foundational to our paper are approaches which fit Lyapunov functions with a dynamics model to approximate a barrier around safe states.[^Chow2018] Guaranteeing Lyapunov stability in the broader sense guarantees that states near an equilibrium point stay there forever, which translates easily to reinforcement learning. This model, however needs a pre-calibrated dynamics model, something CRABS circumvents. A similar barrier idea is used by other papers which guarantee all visited states are “reversible” – meaning the agent is known to be safe if it can travel back to other safe areas [^Molodovan2012]. Does learning about other barrier methods make you curious about CRABS' barrier function? We hope so. Let's dive into details. 


<a name="bc"></a>
## What are barrier certificates?

The name of a barrier certificate gives most readers a good idea of the goal it wishes to accomplish: having a function that tells us whether a state lands within a boundary. But what is the boundary in question and how does the barrier certificate guarantee that? To begin, we don't just want to find states that are safe, but also states which *never* will encounter unsafe states. States that meet this strict criterion are called valid. The barrier certificate  $$h: S \rightarrow \mathbb{R}$$, maps the state space to real numbers, with the below property: 

$$
h(s) > 0 , h(f(s)) > 0
$$

We can choose our function to be the one that moves the time-discrete dynamics model forward one step, $$f(s_t) = s_{t+1}$$. Letting valid states be defined by $$h(s) > 0$$, then we ensure that if an agent is safe at time $$t$$ then it will be safe at time $$t + 1$$. Because it is safe at time $$t + 1$$ it will be safe at time $$t + 2$$ and so forth. 

$$
h(s_{t+1}) = h(f(s_t)) > 0, h(s_{t+n}) > 0
$$

How do we learn a barrier certificate that guarantees this feature? By training a neural network $$h_{\phi}$$ that satisfies the following three requirements:

$$
\mathbf{1.} \quad h(s_{unsafe}) < 0 \qquad \qquad  \mathbf{2.} \quad h_{0} \geq 0 \qquad \qquad  \mathbf{3.} \quad \underset{s' \in \hat{T}(s, \pi(s))}{min} h(s) \geq 0
$$

Namely that all unsafe states give a negative barrier value, the starting state is safe, and the barrier certificate for the worst-next-state from an already safe state will give a positive value. 

The first two requirements are satisfied by always beginning the model in a pre-known safe state and formulating the network, $$h_{\phi}$$, in such a way:

$$
h_{\phi} = 1 - Softplus(f_{\phi}(s) - f_{\phi}(s_0)) - B_{unsafe}(s)
$$

Where $$B_{unsafe} > 1$$ for all unsafe states. 

The third requirement is satsified by first finding the aforementioned worst next state that stems from an aleady known valid state, called $$s*$$. This is calculated with the MALA algorithm, a stochstic gradient Langevin algorithm. Remember that since $$s*$$ is the next state of a valid state, we want our neural network to also certify it as safe. For this worst case scenario the barrier function $$\phi$$ is trained to maximize its ouput. The authors formulated this as the below min-max problem

$$
\underset{\phi}{C*} (h_{\phi}, U, \pi) = \underset{\phi}{min} \quad \underset{s' \in \hat{T}}{max} -h(s')
$$

Where the term $$\underset{\phi}{C*}$$ is derived from $$C_{h}$$ being the set of all valid states.

<a name="lbc"></a>
## How do barrier certificates fit into the CRABS algorithm?

Now that we have covered the novel part of CRABS, we need to address the infrastructure of the algorithm and how it co-trains the barrier certificates along with the dynamics model.

The first step of CRABS is to pretrain a soft-actor-critic policy, $$\pi_{init}$$[^Haarnoja2018] until one is satisfied that the agent behaves safely.
The second step is to safely explore. Exploration is performed by adding gaussian noise to the SAC agent and having it make actions. When any action leaves certified space, the agent falls back on a safeguard policy. During the first epoch the safeguard policy will simply be $$\pi_{init}$$.
Exploration has added new trajectories to our buffer of simulations, $$\hat{D}$$. This allows us to recalibrate our dynamics model, $$\hat{T}$$. $$\hat{T}$$ is paramaterized multiple neural networks, $$\omega$$, and updated by minimzing the negative log likelihood of the seqeunces in the replay buffer happening:

$$
\mathcal{L}_{\hat{T}}(\omega) = -\mathbb{E}_{(s, a, s')\in\hat{D}}( - \log f_{\omega}(s'|s,a))
$$

Because the dynamics model has become more confident about our environment, it allows us to retrain the barrier certificate to expand the number of verified regions.

Finally we re-optimize our policy $$\pi$$ with a modified Soft-Actor-Critic implementation while it is constrained by the barrier certificate. We train to maximize against the worst case scenario $$s*$$ again, just like in the barrier section's min-max problem.

<a name="env"></a>
## Environments

Pendulum             |  Cartpole
:-------------------------:|:-------------------------:
![Swing](https://github.com/lars-chen/rl-blog/blob/master/assets/images/pendulum_examp.gif?raw=true)    |  ![Cartpole](https://github.com/lars-chen/rl-blog/blob/master/assets/images/single_examp.gif?raw=true)

> **Figure 1.** The paper tested the CRABS algorithm on the classic 'swing up' pendulum and 'cartpole' gym/mujoco environments. The swing GIF is taken from [Pendulum Gym](https://www.gymlibrary.dev/environments/classic_control/pendulum/) documentation. Cartpole GIF was our generated during our verification process of the CRABS algorithm.


In the paper, the authors focused on low-dimensionality, high risk environments based on Cartpole and Pendulum in the Gym-Mujoco[^TodorovErezTasa2012] simulation suite. They were able to consistently find that CRABS has zero training-time violations while performing admirably (and sometimes better than other well known algorithms) in terms of reward maximization. 

We chose to expand into two new Mujoco environments with two cases: One where we increase the risk and one where we increase the complexity of the dynamics. 

Double Inverted Pendulum         |  Hopper
:-------------------------:|:-------------------------:
![Double](https://github.com/lars-chen/rl-blog/blob/master/assets/images/double_pendulum_71000%20.gif?raw=true)    |  ![Hopper](https://github.com/lars-chen/rl-blog/blob/master/assets/images/hopper860pre.gif?raw=true)

> **Figure 2.** Our project aimed to extend the code to work on [double inverted pendulum](https://www.gymlibrary.dev/environments/mujoco/inverted_double_pendulum/) and [hopper](https://www.gymlibrary.dev/environments/mujoco/hopper/). This table shows the double inverted pendulum and hopper agents pre-trained successfully SAC, before the CRABS algorithm was applied. 

In the paper, the authors focused on low-dimensionality, high risk environments based on Cartpole and Pendulum. They were able to consistently find that CRABS has zero training-time violations while performing admirably (and sometimes better than other well known algorithms) in terms of reward maximization. We chose to expand the environments in two cases: One where we increase the risk and one where we increase the complexity of the dynamics. 

The first environment we chose is called "Hover." It uses the double cartpole environment and rewards the agent when the tip of the second pole is halfway to its maximum height, while being unsafe if the first joint bends further than a strict threshold. 

The second environment is called "zoom" where we set up the Mujoco Hopper environment to reward fast z-axis movement while the angle of the top stayed within a threshold. This did not incentivize dangerous behaviour as much as the Hover environment, however we wished to show that the algorithm could expand the barrier certificate and better learn dynamics of the system in this setup.

<a name="pre"></a>
## Pre-training 
In the methodology section we mentioned that this algorithm requires a pre-trained safe agent. The authors pre-trained with SAC for 10,000 steps and checked every following 1,000 steps whether the policy was safe, taking the first safe policy they found. Firt we verified their results on the cartpole environment.

We found that the environment reached a plateau of safety around 2000 steps. When we ran this on Hover and Zoom, more complex environments, we were surprised to find that the more complex environments needed less steps (between 500 and 1000) than the cart pole environment. The below plots show the safety for different environments across multiple pretraining runs. The double pendulum and hopper environments are combined into one graph so the reader can contrast the time-frame in which it reaches safety.


![Pre-training Safety Violations Per Episode](https://github.com/lars-chen/rl-blog/blob/master/assets/images/pre-training-safety.png?raw=true)

> **Figure 3.** Number of Pretraining Safety Violations in 'Double Inverted Pendulum' and 'Hopper' pretraining with SAC.

<a name="disc"></a>
## Discussion

In the Open Review discussion of the CRABS paper, the main criticism was that the dimensionalities of the environements investigated were too low. Our project aimed to adapt CRABS to work in the environments Double Inverted Pendulum and Hopper, which increased the dimensionality of the observation and action space significantly.


Mujoco Gym Environment      |  Observation Dimension    | Action Dimension
:-------------------------:|:-------------------------:|:-------------------------:
Pendulum                  						   		 |         3                                   |     1
Cartpole               								       |         4                                   |     1
<span style="color:green"> *Double Inverted Pendulum*  </span>           | <span style="color:green">*11*</span>         |     1
<span style="color:green"> *Hopper*                    </span>           | <span style="color:green">*11*</span>         | <span style="color:green">*3*</span>


> **Table 1.** Hopper and Double Inverted Pendulum increased the dimension of the observation space by 11 and the action space by 2.


In the scope of this class project, we were able to successfully pre-train these new environments with SAC, both acheiving policies with high rewards that were certified safe initial policies. Due to time constraints and limitations on the number of parameters we could test at once, we only were able to complete full runs on the high dimensional environments a few times. Our first few attempts in both Double Inverted Pendulum and Hopper un-learned the safe initial policy and returned highly negative mean rewards after a few epochs and clearly behave unsafely as shown below. We still see progress in the CRABS algorithm becoming more certain about the environment.

There were a battery of tests that we could still try in the future, given more time: various hyperparameters to tweak, reward functions to play around with as well as restricting the safety conditions to a more stable domain. For example, we found that restricting the angle of the first pendulum in double-pendulum resulted in slightly better learning, and our next step was to go further and restrict the second pendulum. Therefore, we cannot definitively state that the CRABS algorithm works or does not work on higher dimensional environments. Given the qualitative results below, we do have hope that CRABS can work in Double Inverted Pendulum, since though it has a high observation space, its action space is still low.


<a name="double"></a>
#### Double Pendulum with CRABS

The CRABS algorithm we implemented rewarded the Double Inverted Pendulum for being in a more unsafe state, namely if the tip of the second pole was halfway to its maximum height such that bottom pole is straight and the top pole is bent at 90° angle.



Epoch 5                    |  Epoch 10                |  Epoch 15
:-------------------------:|:-------------------------:|:-------------------------:
![epoch 5](https://github.com/lars-chen/rl-blog/blob/master/assets/images/double_train_5k.gif?raw=true)  |  ![epoch 10](https://github.com/lars-chen/rl-blog/blob/master/assets/images/double_train_10k.gif?raw=true) |  ![epoch 15](https://github.com/lars-chen/rl-blog/blob/master/assets/images/double_train_15k.gif?raw=true)

> **Figure 4.** CRABS agent visualized in Double Inverted Pendulum over the first few training epochs. 

What we see in **Figure 4.** may be that it is attempting to learn this area of maximum reward. However, instead of learning the threshold where it is safe to bend its poles, in some trajectories the agent begins spinning the top pole (which is a desired result) but can’t help but fly off the track or bending the first pole further than our safety bound (undesired).

<a name="hopper"></a>
#### Hopper with CRABS

Epoch 5                    |  Epoch 10                |  Epoch 15
:-------------------------:|:-------------------------:|:-------------------------:
![epoch 5](https://github.com/lars-chen/rl-blog/blob/master/assets/images/hopper_0.gif?raw=true)  |  ![epoch 10](https://github.com/lars-chen/rl-blog/blob/master/assets/images/hopper_20.gif?raw=true) |  ![epoch 15](https://github.com/lars-chen/rl-blog/blob/master/assets/images/hopper_30.gif?raw=true)

> **Figure 5.** CRABS agent visualized in Hopper over the first few training epochs. 

In the behavior of the hopper, we can see that it has slowed down to almost not move. We believe this is because noise is added in the probabilistic policy development, too many moves that attempt jumping are taking it outside of its safe zone, and it gets stuck in a state where it hasn’t yet found a safe way to get into a jumping animation. 

{% include uncert_hopper_100.html %}

> **Figure 6.** Grid of states applied to the ensemble dynamics trained with CRABS in the Hopper environment. 

To better to understand what was happening in the CRABS learning process in a higher dimensional environment, we created a grid of possible states over the leg and foot joint angles, keeping all other 9 observation dimensions constant at the starting position of the Hopper simulation. Additionally, the action space was limited to $$\vec{a} = (0, 0, 0)^\intercal$$ . The foot and leg joints were chosen as observations since, in **Figure 5.**, these are the ones that are still moving, whereas the other joint angles remain static. The grid of states was then plugged into the ensemble $$N=5$$ dynamics model and the the standard deviation was calculated over the ensemble prediction for the next state.

What can been seen over the 11 epochs that are visualized in **Figure 6.** is that the the dynamics model had some variance across the grid at epoch 0 and as training progresses the uncertainty of of the ensemble model becomes very low in the angle space that the agent stays in and loses certainty in the states that do not get explored anymore. Hence, we see that the dynamics model is likely working and that the difficulties learning may lie elsewhere than the model. 


<a name="ref"></a>
#### References
----------------

[^SuttonBarto2018]: Sutton, R. S., &amp; Barto, A. G. (2018). Reinforcement learning: An introduction. MIT Press Ltd. 

[^Sutton1988]: Sutton, R. S. (1988). Learning to Predict by the Methods of Temporal Differences. Machine Learning, 3, 9-44. 

[^GarciaFernandez2015]: Garcıa, J., & Fernández, F. (2015). A comprehensive survey on safe reinforcement learning. Journal of Machine Learning Research, 16(1), 1437-1480.

[^Akkaya2019]: Akkaya, I., Solving Rubik's Cube with a Robot Hand _arXiv.org_, p.07113 available at: [https://arxiv.org/abs/1910.07113](https://arxiv.org/abs/1910.07113).

[^Gu2023]: Gu, S., 2023. A Review of Safe Reinforcement Learning: Methods, Theory and Applications. _arXiv.org_, p.10330. Available at: [https://arxiv.org/abs/2205.10330](https://arxiv.org/abs/2205.10330).

[^Schulman2017]: Schulman, J., 2023. Trust Region Policy Optimization _arXiv.org_, p.05477. Available at: [https://arxiv.org/abs/1502.05477](https://arxiv.org/abs/1502.05477)

[^Achiam2017]: Achiam, J., Held, D., Tamar, A., & Abbeel, P. Constrained Policy Optimization _arXiv.org_, p.10528. Available at: [https://arxiv.org/abs/1705.10528](https://arxiv.org/abs/1705.10528)

[^Balakrishna2017]: Thananjeyan, B., &amp; Balakrishna, A 2021. Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones _arXiv.org_, p.15920. Available at: [https://arxiv.org/pdf/2010.15920.pdf](https://arxiv.org/pdf/2010.15920.pdf)

[^Le2019]: Le, H., 2019. Recovery RL: Batch Policy Learning under Constraints _arXiv.org_, p.08738.

[^Zhang2020]: Zhang, J., Cheung, B., Finn, C., Levine, S., & Jayaraman, D. (2020, November). Cautious adaptation for reinforcement learning in safety-critical settings. In International Conference on Machine Learning (pp. 11055-11065). PMLR. Available at: [https://proceedings.mlr.press/v119/zhang20e.html](https://proceedings.mlr.press/v119/zhang20e.html)

[^Zimmer2014]: Zimmer, M., Viappiani, P., & Weng, P. (2014, May). Teacher-student framework: a reinforcement learning approach. In AAMAS Workshop Autonomous Robots and Multirobot Systems. Available at: [https://hal.science/hal-01215273/](https://hal.science/hal-01215273/)

[^Chow2018]:  Chow, Y., Nachum, O., Duenz-Guzman, E., Ghavamzadeh, M.  2018. A Lyapunov-based Approach to Safe Reinforcement Learning _arXiv.org_, p.07708v1. Available at: [https://arxiv.org/pdf/1805.07708.pdf](https://arxiv.org/pdf/1805.07708.pdf)

[^Molodovan2012]: Moldovan, T., Abbeel, P.,  2012. Safe Exploration in Markov Decision Processes _arXiv.org_, p.4810. Available at: [https://arxiv.org/pdf/1205.4810.pdf](https://arxiv.org/pdf/1205.4810.pdf)

[^Haarnoja2018]: Haarnoja, T., Zhou, A., Abbeel, P., Levine, S., 2012. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor _arXiv.org_, p.01290. Available at: [https://arxiv.org/abs/1801.01290](https://arxiv.org/abs/1801.01290)

[^TodorovErezTasa2012]: Todorov, E., and Erez, T., and Tassa, Y. 2012. MuJoCo: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems. Available at [https://github.com/deepmind/mujoco](https://github.com/deepmind/mujoco)
 
