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

In reinforcement learning (RL), an agent is trained to navigate an enivronment and maxmize its reward using a function crafted by human investigator (Sutton and Barto, 1998). RL has been shown it can complete high-dimensional tasks in robotic simulation environments (Akkaya et al., 2019) by exploring many possible states and optimizing its policy. In real world applications of RL, such as biomedical robotics, low reward areas could be states that hurt the patient or damage the agent itself. 

A current area of research uses constrained policy optimization (CPO) that reduce training-time safety incursions. This method approximates a constraint cost of a policy update and ensures that the new policy stays within the limitations. However, CPO ulimately learns to approximate the constraint cost by encountering unsafe states and comparing these to safe states - therefore, the set of safe states is learned after running into unsafe states. Another line of work uses Lyaponov functions to verify the safety of a set of states but requires provided dynamics model that is non-trivially crafted by the researcher.

The paper that we adapt in this project, ["Learning Barrier Certificates: Towards Safe Reinforcement Learning with Zero Training-time Violations"](https://arxiv.org/pdf/2108.01846.pdf) (Yuping Luo and Tengyu Ma, 2022) uses barrier certificate function similar to Lyaponov functions that certify viable states. The novel approach of this work is that the barrier certificates are learned iteratively while calibrating a dynamics model, optimizing the policy and adding safe trajectories to a replay buffer. 


## what are barrier certificates?

A barrier certificates,  $$h: S \rightarrow \mathbb{R}$$, maps the state space to real numbers, such that given a time-discrete dynamics model, $$f(s_t) = s_{t+1}$$ :

> $$
> h(s) > 0, h(f(s)) > 0
> $$

Letting valid states, i.e. states that do not lead to unsafe states, be defined by $$h(s) > 0$$, then we ensure the agent never enters an unsafe state.

> $$
> h(s_{t+1}) = h(f(s_t)) > 0
> $$

Of course, these barrier functions need to be learned over many iterations which is done in the paper by training a neural network $$h_{\phi}$$ that satisfies the following three requirements:

>  $$\mathbf{1.} h(s_{unsafe}) < 0$$ \\
>  $$\mathbf{2.} h_{0} \geq 0$$ \\
>  $$\mathbf{3.} \underset{s' \in \hat{T}(s, \pi(s))}{min} h(s) \geq 0$$

The first two  requirements can be satisfied by formulating the network, $$h_{\phi}$$, in such a way:

> $$
> h_{\phi} = 1 - Softplus(f_{\phi}(s) - f_{\phi}(s_0)) - B_{unsafe}
> $$

The third requirement is satsified by first adversarially calculated the worst possible next state $$s*$$. Then, for that worst case scenario the barrier function $$\phi$$ must be parameteritzed such that the worst case has the lowest barrier certificate value. The aim is to only include values in the set of valid states if their certificate values are positive:   

> $$
> \underset{\phi}{C*} (h_{\phi}, U, \pi) = \underset{\phi}{min} \underset{s' \in \hat{T}{max} -h(s')
> $$



