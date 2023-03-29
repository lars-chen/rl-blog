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

There are already many unique and diverse strategies to accomplish the goal of safety. In a review written in 2013, García and Fernández (https://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf) divided safe RL algorithms into two main genres: algorithms that modify the optimality criterion with a safety factor (like primal-dual problems which add penalty terms to the Lagrangian) and algorithms that modify the exploration behavior. Nearly ten years later many new sophisticated and inventive methods have been proposed, Gu et al. (https://arxiv.org/pdf/2205.10330.pdf) chose to make the main distinction between algorithms in whether they are model based or model free. 

CRABS falls firmly into being model-based and modifying exploration behavior. Before going into details of CRABS we will mention other strategies that exemplify the diversity of the categories above. One sub-area of staying within constraints is Trust Region Policy Optimization, which guarantees policies do not make too large of leaps by requiring new policies have a low KL-Divergence from an old safe policy (https://arxiv.org/pdf/1502.05477.pdf). Grown out of Trust Region methods is Constraint Policy Optimization (CPO), one of the most well-known recent model-free developments in RL. It is a gradient method that optimizes a constraint cost for a policy update. CPO ultimately learns to approximate the constraint cost by encountering unsafe states and comparing these to safe states. 
The actor-critic method, which isolates the reward function and value function to separate agents has also been combined with CPO, again needing to encounter unsafe states but yielding high rewards (https://arxiv.org/pdf/2010.15920.pdf). Some researchers modify the exploration behavior by providing batches of existing data (https://arxiv.org/pdf/1903.08738.pdf) or by training beforehand in a sandbox environment where healthy errors can be made before training in the “real” environment where fatal errors would occur (http://proceedings.mlr.press/v119/zhang20e/zhang20e.pdf). Some allow for humans to act as teachers and guide the agent to learn faster and also avoid bad states (https://hal.science/hal-01215273/document).  Depending on the researcher, these methods may or may not incorporate a dynamics model. 
Foundational to our paper are approaches which fit Lyapunov functions with a dynamics model to approximate a barrier around safe states (https://arxiv.org/pdf/1805.07708.pdf). Guaranteeing Lyapunov stability in the broader sense guarantees that states near an equilibrium point stay there forever, which translates easily to reinforcement learning. A similar idea is used by other papers which want to guarantee that all visited states are “reversible” – meaning the agent is known to be safe if it can travel back to other safe areas (https://arxiv.org/pdf/1205.4810.pdf)
![image](https://user-images.githubusercontent.com/57923541/228575070-5f97369d-4f37-42b4-b3dc-483ca8161633.png)



## what are barrier certificates?

A barrier certificates,  $$h: S \rightarrow \mathbb{R}$$, maps the state space to real numbers, such that given a time-discrete dynamics model, $$f(s_t) = s_{t+1}$$ :

$$
h(s) > 0, h(f(s)) > 0
$$

Letting valid states, i.e. states that do not lead to unsafe states, be defined by $$h(s) > 0$$, then we ensure the agent never enters an unsafe state.

$$
h(s_{t+1}) = h(f(s_t)) > 0
$$

Of course, these barrier functions need to be learned over many iterations which is done in the paper by training a neural network $$h_{\phi}$$ that satisfies the following three requirements:

$$
\mathbf{1.} \quad h(s_{unsafe}) < 0 \qquad \qquad  \mathbf{2.} \quad h_{0} \geq 0 \qquad \qquad  \mathbf{3.} \quad \underset{s' \in \hat{T}(s, \pi(s))}{min} h(s) \geq 0
$$

The first two  requirements can be satisfied by formulating the network, $$h_{\phi}$$, in such a way:

$$
h_{\phi} = 1 - Softplus(f_{\phi}(s) - f_{\phi}(s_0)) - B_{unsafe}
$$

The third requirement is satsified by first adversarially calculated the worst possible next state $$s*$$. Then, for that worst case scenario the barrier function $$\phi$$ must be parameteritzed such that the worst case has the lowest barrier certificate value. The aim is to only include values in the set of valid states if their certificate values are positive:   

$$
\underset{\phi}{C*} (h_{\phi}, U, \pi) = \underset{\phi}{min} \quad \underset{s' \in \hat{T}}{max} -h(s')
$$



