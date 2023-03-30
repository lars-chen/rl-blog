---
title:  "learning barrier certificates"
mathjax: true
layout: post
categories: media
date: 2023-03-28
---


## abstract

Having a barrier function verify safe states is an often-used strategy to guarantee that one doesn’t incur training-time errors in Safe RL. Depending on how one sets up this barrier function, it can require effortful hand-tuning specific to any new environment.  Last year, Luo and Ma proposed a method that sidesteps this effort by co-learning three elements: 1) improving the confidence of the physics model, 2) increasing the size of verified regions, and 3) optimizing the policy. They posit that any of the three elements will incrementally improve after benefitting from improvements in the other two elements, creating a complimentary sequential structure. Instead of requiring a pre-made barrier function, their algorithm now requires an initial safe policy as a starting point. They showed in simulations with low dimensional environments that their algorithm was capable of expanding the safe region while incurring no training errors. We introduced the algorithm into two environments with higher dimensionality: double-cartpole and hopper, and we performed an analysis on the safety of pre-trained agents in the two environments. We found that  _______

## introduction

In reinforcement learning (RL), an agent is trained to navigate an environment and maximize its reward using a function crafted by a human investigator (Sutton and Barto, 1998). The simplest classical algorithms maintain expectations of rewards in different states and update them after taking actions. Modern, more sophisticated versions of RL have even been shown to complete high-dimensional tasks in robotic simulation environments (Akkaya et al., 2019). In real world applications of RL, such as biomedical robotics, low reward areas could be states that hurt the patient or damage the agent itself. Safe RL is concerned with learning a high reward policy while either maintaining integrity of the agent or not violating external constraints. 

There are already many unique and diverse strategies to accomplish the goal of safety. In a review written in 2013, García and Fernández (https://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf) divided safe RL algorithms into two main genres: algorithms that modify the optimality criterion with a safety factor (like primal-dual problems which add penalty terms to the Lagrangian) and algorithms that modify the exploration behavior. Nearly ten years later many new sophisticated and inventive methods have been proposed, Gu et al.[^Gu2023] chose to make the main distinction between algorithms in whether they are model based or model free. 

CRABS falls firmly into being model-based and modifying exploration behavior. Before going into details of CRABS we will mention other strategies that exemplify the diversity of the categories above. One sub-area of staying within constraints is Trust Region Policy Optimization, which guarantees policies do not make too large of leaps by requiring new policies have a low KL-Divergence from an old safe policy (https://arxiv.org/pdf/1502.05477.pdf). Grown out of Trust Region methods is Constraint Policy Optimization (CPO), one of the most well-known recent model-free developments in RL. It is a gradient method that optimizes a constraint cost for a policy update. CPO ultimately learns to approximate the constraint cost by encountering unsafe states and comparing these to safe states. 
The actor-critic method, which isolates the reward function and value function to separate agents has also been combined with CPO, again needing to encounter unsafe states but yielding high rewards (https://arxiv.org/pdf/2010.15920.pdf). Some researchers modify the exploration behavior by providing batches of existing data (https://arxiv.org/pdf/1903.08738.pdf) or by training beforehand in a sandbox environment where healthy errors can be made before training in the “real” environment where fatal errors would occur (http://proceedings.mlr.press/v119/zhang20e/zhang20e.pdf). Some allow for humans to act as teachers and guide the agent to learn faster and also avoid bad states (https://hal.science/hal-01215273/document).  Depending on the researcher, these methods may or may not incorporate a dynamics model. 
Foundational to our paper are approaches which fit Lyapunov functions with a dynamics model to approximate a barrier around safe states (https://arxiv.org/pdf/1805.07708.pdf). Guaranteeing Lyapunov stability in the broader sense guarantees that states near an equilibrium point stay there forever, which translates easily to reinforcement learning. A similar idea is used by other papers which want to guarantee that all visited states are “reversible” – meaning the agent is known to be safe if it can travel back to other safe areas (https://arxiv.org/pdf/1205.4810.pdf)



## what are barrier certificates?

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

## How do barrier certificates fit into the CRABS algorithm?

Now that we have covered the novel part of CRABS, we need to address the infrastructure of the algorithm and how it co-trains the barrier certificates along with the dynamics model.
The first step of CRABS is to pretrain a soft-actor-critic policy, $$\pi_{init}$$ (https://arxiv.org/abs/1801.01290) until one is satisfied that the agent behaves safely.
The second step is to safely explore. Exploration is performed by adding gaussian noise to the SAC agent and having it make actions. When any action leaves certified space, the agent falls back on a safeguard policy.
Exploration has added new trajectories to our buffer of simulations, which allows us to recalibrate our dynamics model, $$\hat{T}$$ #todo put in pi and thats here. 
Because the dynamics model has become more confident about our environment, it allows us to retrain the barrier certificate to expand the number of verified regions.
Finally we re-optimize our policy while it is constrained by the barrier certificate. 

## Environments

In the paper, the authors focused on low-dimensionality, high risk environments based on Cartpole and Pendulum. They were able to consistently find that CRABS has zero training-time violations while performing admirably (and sometimes better than other well known algorithms) in terms of reward maximization. We chose to expand the environments in two cases: One where we increase the risk and one where we increase the complexity of the dynamics. 
The first environment we chose is called "Hover." It uses the double cartpole environment and rewards the agent when the tip of the second pole is halfway to its maximum height, while being unsafe if the first joint bends further than a strict threshold. 
The second environment is called "zoom" where we set up the Mujoco Hopper environment to reward fast z-axis movement while the angle of the top stayed within a threshold. This did not incentivize dangerous behaviour as much as the Hover environment, however we wished to show that the algorithm could expand the barrier certificate and better learn dynamics of the system in this setup.


References
-----------

[^Gu2023]: Gu, S., 2023. A Review of Safe Reinforcement Learning: Methods, Theory and Applications. _arXiv.org_, p.10330. Available at: [https://arxiv.org/abs/2205.10330](https://arxiv.org/abs/2205.10330).
