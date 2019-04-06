# Deep Deterministic Policy Gradients

Recall that DQNs only work in discrete actions spaces. The reason is that actions are selected as `argmax_a Q`, where `Q` is the action value function. In discrete action spaces, the `Q` value for any given transition, is a finite vector. But for continuous action spaces, there are an infinite number of actions one can take at any given point. To implement DQNs in a continuous action space, `Q` is an infinite vector, and it's therefore impossible to compute `argmax_a Q` efficiently and without prior knowledge of the environment.

A DDPG is a modification of the DQN algorithm that extends it to continuous action spaces. The main insight is that instead of computing `Q` and then selecting an action, we can compute the optimal action `a` directly.

## The Policy Network

We introduce a policy network `pi` parameterized by `phi` that inputs a state `s` and output the optimal action `a`. Of course, we don't know what action is optimal a priori, but we can learn that by maximizing `max_a Q`. Indeed, if `pi` generates actions that maximize `max_a Q`, then those are by definition the optimal actions.

For this reason, the `pi` loss is just `max_a Q`:

`loss_pi = -mean(max_Q)`

(The minus sign is there because we're performing gradient ascent, and autograd libraries like PyTorch and Tensorflow perform gradient descent.) Since `Q` is a function of `s,a` so is `max_Q`, but whereas `Q` outputs a vector with dimension `act_dim`, `max_Q` outputs a single value (the maximum), so we need to modify out loss:

`loss_pi = -mean(max_Q(s,a))`

Finally, the optimal action `a` is given by the ouput of `pi` so the loss becomes:

`loss_pi=-mean(max_q(s,pi(s)))`


## The Q Network

In addition to the policy network `pi`, DDPGs also optimize the `Q` in a manner similar to a DQN by ensuring that `Q` satisfies the Bellman equation. `Q` optimization for DDPGs is slightly updated from the DQN treatment, since we're optimizing `max_Q` now. The new Bellman loss reads:

`loss_Q = mean((max_Q(s,a)-(r+(1-d)*gamma*max_Q_target(s',a')))**2)`

As before, we have the `max_Q` and `max_Q_target` networks to stabilize training. The action `a` is sampled from the replay buffer, along with `s,r,d,s'`, the only remaining variable to account for is `a'`. Since `a'` is not sampled from the buffer, we approximate it with `a'=pi(s')`. However, if we use `pi` directly we run into a similar instability problem as we had with the `Q` network. For this reason, we introduce another target network `pi_target` to select `a'` that gets updated in the same way as `max_Q_target`, and the final loss reads:

`loss_Q = mean((max_Q(s,a)-(r+(1-d)*gamma*max_Q_target(s',pi_target(s'))))**2)`

## Gaussian Policy

Since DDPGs function in continuous action spaces, we use a Gaussian policy for sampling actions (see the REINFORCE algorithm for details).

Finally, DDPGs update target networks in the same way as DQNs, and also store transitions in a large replay buffer.

# The DDPG Algorithm

1. > Initialize the replay buffer `B` and network parameters for `Q`, `Q_target`,`pi`, and `pi_target`
2. > for `epoch` in `epochs`:
3. >>> for `t` in `steps_per_epoch`:
4. >>>>> observe state `obs`
5. >>>>> given `obs`, sample action `act` from epsilon greedy policy (random or `argmax_a Q`)
6. >>>>> perform action `act` and observe next state `obs2`, reward `rew`, and done `done`
7. >>>>> store `(obs,act,rew,obs2,done)` in episode buffer
8. >>>>> perform one step of gradient optimization:
9. >>>>>>> sample minibatch `m` from `B`, and minimize the Bellman loss and Pi loss
10. >>>>>>> `loss_ddpg = .5*mean((max_Q(s,a)-(r+gamma*(1-done)*max_a' max_Q(s',pi_target(s'))))**2)`
12. >>>>>>> `loss_pi = -mean(max_Q(s,pi(s)))``
12. >>>>>>> copy weights from `Q` to `Q_target` and `pi` to `pi_target` if ready for update
11. >>>>> if done:
12. >>>>>>> reset environment
13. >>>>> if epoch is done:
14. >>>>>>> print average returns and loss values during the epoch
