# Proximal Policy Optimization

While VPGs reduce the variance of the loss in the standard REINFORCE algorithm
by weighing the log likelihood of the policy `pi` with the advantage `A` in the loss

`loss_vpg = -mean(log_prob_pi * A)` vs `loss_reinforce = -mean(log_prob_pi * R)`

and speed up training, they can still be unstable and lead to suboptimal policies. The problem is
that the new policy after a step of optimization `pi_new` can be far from the old policy
`pi_old` as measured by their KL divergence. The intuition behind PPOs is that we want to
keep `pi_new` close to `pi` to increase stability and hence convergence to an optimal policy.

There are two PPO variants - one that penalizes policy updates that significantly
increase the KL divergence, and another that explicitly clips the new policy. We'll implement
the clipped version.

PPOs replace the VPG loss with a surrogate loss that approximates the optimal loss function.

`loss_ppo_unclipped = pi/pi_old * A`

and bounds the update by clipping the approximate loss

`loss_ppo = -mean(loss_ppo_unclipped,1-epsilon,1+epsilon)`

where `epsilon` is the clipping threshold. The unclipped surrogate loss is actually just the first order Taylor expansion of the original VPG loss. The PPO is therefore an approximation
of the orignal loss, clipped to keep the new policy near the old one during update, which
increases stability.

Like REINFORCE and VPG, the PPO algorithm works for both continuous and discrete action spaces.
A minor change to the rollout, is that we store `(obs,act,rew,logp_t)` in the buffer in order
to access `pi_old` when evaluating the PPO loss.

# The PPO Algorithm


1. > Initialize the buffer `B`, logger, actor policy `pi`, critic `V`, and optimizers
2. > for `epoch` in `epochs`:
3. >>> for `t` in `steps_per_epoch`:
4. >>>>> observe state `obs`
5. >>>>> given `obs`, sample action `act` from policy `pi`
6. >>>>> perform action `act` and observe next state `obs2`, reward `rew`, `logp_t`, and done `done`
7. >>>>> store `(obs,act,rew,logp_t)` in episode buffer
8. >>>>> if done:
9. >>>>>>> store trajectory (the episode buffer) in epoch buffer `B`
10. >>>>>>> reset episode buffer and reset environment
11. >>>>>if epoch is done:
12. >>>>>>> get trajectories from epoch buffer
13. >>>>>>> calculate `pi/pi_old` and `A` using GAE (see VPG)
14. >>>>>>> compute policy loss `loss_pi = - mean(clip(pi/pi_old * A,1-eps,1+eps))`
15. >>>>>>> compute values loss `loss_v = .5*mean(R-V)**2`
16. >>>>>>> perform step of optimization for both actor and critic networks
