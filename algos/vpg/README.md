# Vanilla Policy Gradient

Vanilla Policy Gradients extend the REINFORCE algorithm by reducing the variance of the policy `pi` loss.
The key insight with VPGs is that we can add any function `b(s_t)` that only depends on the state to
the likelihood weight - `R_t` for REINFORCE. The reason is that the likelihood gradient weighed by `b(s_t)`
disappears ([see the EGLP lemma](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)). 

We can then change the weight `R_t` in the loss `mean(log_probs * R_t)` to any value that satisfies the EGLP lemma
and returns the gradient of the expected reward. The reason we would want to do this is to reduce variance of the loss,
which would speed up training by eliminating a swath of suboptimal paths. For VPGs, we replace the reward weight `R_t` 
with the advantage `A_t = Q_t - V_t`, where `Q_t` is the action-value function and `V_t` is the value function
(see the [relevant section of SpinningUp](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)).

We estimate the value `V` with a new function
with parameters `phi` that also gets optimized during gradient descent but with a 
different loss. This is our first example of an actor-critic method. We have an actor `pi` that produces actions
and a critic `V` that judges the quality of the actor's actions by computing its advantage `A_t`.

Finally, we still don't have an explicit value for `Q_t` and hence `A_t`.
Since `A_t` cannot be computed explicitly, we need a technique to estimate it. 
We use the Generalized Advantage Estimation method
(see [the original paper here](https://arxiv.org/abs/1506.02438))


# Vanilla Policy Gradient (VPG) Algorithm:

1. > initialize the buffer and logger
2. > initialize the actor policy `pi`, critic policy `vf`, actor optimizer, and critic optimizer
3. > for epoch in epochs:
4. >>> for t in episode:
5. >>>>> observe state `obs`
6. >>>>> given obs, sample action `act` from policy `pi`
7. >>>>> estimate the value of the of the observation `v_t`
8. >>>>> perform action `act` and observe next state `obs2`, reward `rew`, and done `done`
9. >>>>> store `(obs,act,rew,v_t)` in episode buffer
10. >>>>> if done:
11. >>>>>>> compute advantage estimates `advs` using Generalized Advantage Estimation (GAE)
12. >>>>>>> store trajectory (the episode buffer) and advantages `advs` in epoch buffer
13. >>>>>>> reset episode buffer and reset environment
14. >>>>> if epoch_done:
15. >>>>>>> get trajectories from epoch buffer
16. >>>>>>> calculate `log_likelihood(all_obs,all_acts)`
17. >>>>>>> compute actor loss `loss_pi = - mean(log_likelihood * advantages)`
18. >>>>>>> compute critic loss `loss_v = mean((rewards - value_estimates)**2)`
19. >>>>>>> perform on step of optimization on both actor and critic networks

## Generalized Advantage Estimation (GAE) algorithm:

1. > Given episode rewards `ep_rtgs`, episode value estimates `ep_vals`, and parameters `gamma` & `lambda`
2. > Compute `delta_t = r_t + gamma * v_{t+1} - v_t` for all `t`
3. > Compute advantage estimate `adv_t = sum_j^T (gamma lambda)^j delta_{t+1}` for all `t`
-  >>> Note: although the trajectory has length `T`, the last element in the sum has a `delta_{T+1}` term
         meaning we will need to append an extra term to the trajectory temporarily to compute the estimate
