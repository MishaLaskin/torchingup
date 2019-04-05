# Implementation of the REINFORCE on-policy algorithm. 

The REINFORCE algorithm is perhaps the simplest variant of on-policy learning. The agent acts according to the policy $$\pi_\theta$$ parameterized by $$\theta$$ over $$N$$ episodes. The algorithm has two parts:

## Part 1: Rollout

The agent collects $$N$$ trajectories $$\tau_{j=1,...,N}$$ by acting according $$\pi_{\theta}$$. A trajectory $$\tau$$ consists of $$(s_t,a_t,r_t)$$ tuples.

## Part 2: Optimization

Optimization - after rollout, we optimize parameters $$\theta$$ to maximize the likelihood of the policy $$\pi_\theta$$ weighed by the discounted rewards-to-go $$R_t$$. The gradient update is defined as

$$\theta_{k+1}=\theta_k + \alpha g$$

where

$$g= \frac 1 {|B_k|} \nabla_\theta \sum_\tau^N \sum_t^T \log \pi_\theta (a_t|s_t)|_{\theta_k} R_t$$

REINFORCE is an on-policy algorithm, because optimization is performed on the trajectories $$\tau$$ collected by running the latest policy $$\pi_\theta$$.

## Choice of Policy

The policies used in this example depend on whether the action space is discrete or continuous. For a discrete action space, we use an epsilon greedy policy where actions are sampled from a multinomial distribution.

# The REINFORCE Algorithm


1. Initialize the buffer $$B$$, logger, actor policy $$\pi$$, and optimizer
2. for `epoch` in `epochs`:
  1. for `t` in `steps_per_epoch`:
    1. observe state `obs`
    2. given `obs`, sample action `act` from policy $$\pi$$
    3. perform action `act` and observe next state `obs2`, reward `rew`, and done `done`
    4. store `(obs,act,rew)` in episode buffer
    5. if done:
      1. store trajectory (the episode buffer) in epoch buffer $$B$$
      2. reset episode buffer and reset environment
    6. if epoch is done:
      1. get trajectories from epoch buffer
      2. calculate log_likelihood(all_obs,all_acts)
      3. compute policy loss `loss_pi = - mean(log_likelihood * R(tau))`
      4. perform on step of optimization on policy network `pi`
