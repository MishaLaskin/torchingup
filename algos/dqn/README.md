# Deep Q Network

Deep Q Nets are one of the simplest variants of off-policy learning. Instead of optimizing
the policy `pi` directly, Deep Q Networks select actions that maximize `max_a Q`. Recall that
the `Q` function is the expected return given an initial state `s` and action `a`.

`Q(s,a) = sum_t gamma^t * r_t |_{s,a}`

and that `Q` functions satisfy the following property by definition

`Q(s,a) = r + gamma * Q(s',a')`

The optimal `Q` function maximizes future expected returns and satisfies the Bellman Equation

`Q(s,a) = r + gamma * max_a' Q(s',a')`

In some simple environments, it's possible to compute the Q function explicitly, but in most settings
(including some very simple ones like CartPole), we need to approximate it. DQNs parametrize
the `Q` function with parameters `theta` and optimize the parameters so that `Q` satisfies the
Bellman equation. The loss is just the MSE of the Bellman Equation.

`loss_dqn = .5*mean((Q(s,a)-(r+gamma*max_a' Q(s',a')))**2)`

Finally, if the episode is done, then `Q(s',a')=0`, so we modify the loss accodingling

`loss_dqn = .5*mean((Q(s,a)-(r+gamma*(1-done)*max_a' Q(s',a')))**2)`


Note that during rollout, we need to store `(s,a,r,s_next,d)` transition tuples to evaluate the Bellman equation.

## Epsilon Greedy Policy

During training, we use an Epsilon Greedy policy. We select a random action, with
probability `epsilon` and a greedy action that maximizes `Q` with probability `1-epsilon`.

## Target Network

Deep Q learning can be highly unstable since the Bellman loss uses the same network `Q`
to generate predictions `Q(s,a)` and targets `Q(s',a')`, which results in a moving target
and often leads to suboptimal approximations of `Q`.

To solve this issue, DQNs introduce a target network `Q_target`, which is a copy of the `Q` network
whose weights are updated less frequently. Once every fixed number of steps, the `Q` network weights
are copied to `Q_target`, which is called a hard update. Another way to update `Q_target` is by performing a soft update during every step of optimization, defined

`Q_target.weights = (1-tau)*Q_target.weights + tau*Q.weights`

where `tau` is a scalar value usually close to zero (e.g. `tau=1e-2`).

## Replay Buffer

The other trick used for training DQNs, and any off-policy algorithm, is a replay buffer. The replay buffer contains transitions from all previous experiences. In practice, the buffer has a maximum
capacity (for example `1e6` experiences), and once capacity is reached old transitions are replaced by
new ones on a first-in first-out basis.

During optimization, a minibatch of transitions (e.g. `n=128`) is sampled from the replay buffer
randomly, and the `Q` network parameters are optimized to satisfy the Bellman equation. This scheme is the reason a DQN is an off-policy algorithm. During optimization, the sampled minibatch of transitions contains experiences that could have been collected by any previous policy, not just the current one.

## Rollout and Optimization Decoupled

Lastly, notice how rollout and optimization are completely decoupled in off-policy algorithms. Since optimization can be done at any point `size(buffer) >= minibatch_size`, we don't have to wait for the agent to finish the episode to perform optimization. This is completely unlike on-policy algorithms, such as REINFORCE, where we need to wait to rollout `N` trajectories before performing optimization.

For this reason, you'll often see different implementations of off-policy algorithms run the optimization at different times - e.g. once every timestep or multiple times at the end of the episode. This is mostly a design choice.

# The DQN Algorithm

1. > Initialize the replay buffer `B` and network parameters for `Q` and `Q_target`
2. > for `epoch` in `epochs`:
3. >>> for `t` in `steps_per_epoch`:
4. >>>>> observe state `obs`
5. >>>>> given `obs`, sample action `act` from epsilon greedy policy (random or `argmax_a Q`)
6. >>>>> perform action `act` and observe next state `obs2`, reward `rew`, and done `done`
7. >>>>> store `(obs,act,rew,obs2,done)` in episode buffer
8. >>>>> perform one step of gradient optimization:
9. >>>>>>> sample minibatch `m` from `B`, and minimize the Bellman loss
10. >>>>>>> `loss_dqn = .5*mean((Q(s,a)-(r+gamma*(1-done)*max_a' Q(s',a')))**2)`
11. >>>>> if done:
12. >>>>>>> reset environment
13. >>>>> if epoch is done:
14. >>>>>>> print average returns and loss values during the epoch
