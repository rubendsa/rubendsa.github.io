---
title: "PPO, visualized inside Isaac Lab"
subtitle: "Follow one training iteration from thousands of parallel robot worlds to a shuffled PPO update."
description: "An interactive, step-by-step visual guide to PPO in Isaac Lab: observations, actions, rollout storage, GAE, mini-batches, and the clipped policy update."
excerpt: "An interactive, step-by-step visual guide to PPO in Isaac Lab: observations, actions, rollout storage, GAE, mini-batches, and the clipped policy update."
date: 2026-09-01 01:00:00 +0000
last_modified_at: 2026-09-01 01:00:00 +0000
permalink: /blog/ppo-visualized-in-isaac-lab/
categories:
  - robot-learning
tags:
  - PPO
  - Isaac Lab
  - RSL-RL
  - reinforcement learning
classes:
  - notebook-theme
author_profile: false
sidebar: false
toc: false
ppo_visualizer: true
header:
  og_image: /images/ppo-visualized-isaac-lab-social.png
---

PPO can look like one large optimization algorithm from the outside. Inside an Isaac Lab run, it is easier to understand as two alternating phases: **collect a rectangle of experience from many robot worlds, then revisit that fixed rectangle in shuffled mini-batches to improve the policy.**

<div class="notebook-callout">
  <p class="notebook-callout__label">The shape to remember</p>
  <p><strong><span data-ppo-horizon>24</span> environment steps × <span data-ppo-envs>4,096</span> parallel environments = <span data-ppo-transitions>98,304</span> transitions.</strong> That rectangle is one rollout. PPO computes targets over it, learns from it, then starts a fresh one.</p>
</div>

This article follows [Isaac Lab develop on September 1, 2026](https://github.com/isaac-sim/IsaacLab/tree/8c3bc5e8e8dd4553fa160a11836e90922832c968), which pins RSL-RL 5.4.1. Its Unitree Go2 rough-terrain configuration uses 4,096 environments, 24 rollout steps, four mini-batches, and five learning epochs. The dimensions are representative, not requirements. Change the two collection settings below to see how the batch grows.

## One PPO iteration, step by step

<div class="ppo-stepper" data-ppo-stepper aria-labelledby="ppo-stepper-title">
  <div class="ppo-stepper__topline">
    <div>
      <p class="ppo-stepper__eyebrow">Interactive walkthrough · one training iteration</p>
      <h3 id="ppo-stepper-title">Follow the data, not the acronym</h3>
    </div>
    <span class="ppo-stepper__counter" data-ppo-counter>01 / 07</span>
  </div>

  <div class="ppo-stepper__settings" aria-label="Illustrative rollout settings">
    <label for="ppo-env-count">
      <span>Parallel worlds · N</span>
      <input id="ppo-env-count" type="range" min="0" max="4" step="1" value="3" data-ppo-env-input>
      <output for="ppo-env-count" data-ppo-env-output>4,096</output>
    </label>
    <label for="ppo-horizon">
      <span>Rollout horizon · T</span>
      <input id="ppo-horizon" type="range" min="0" max="4" step="1" value="2" data-ppo-horizon-input>
      <output for="ppo-horizon" data-ppo-horizon-output>24</output>
    </label>
    <div class="ppo-stepper__stat">
      <span>Rollout · B = N × T</span>
      <strong data-ppo-transitions>98,304</strong>
      <small>transitions</small>
    </div>
    <div class="ppo-stepper__stat">
      <span>One mini-batch · B ÷ 4</span>
      <strong data-ppo-minibatch>24,576</strong>
      <small>samples · 20 updates total</small>
    </div>
  </div>

  <div class="ppo-stepper__nav" role="tablist" aria-label="PPO iteration stages">
    <button type="button" class="is-active" role="tab" aria-selected="true" aria-controls="ppo-stage-observe" data-ppo-step="0"><span>01</span> Observe</button>
    <button type="button" role="tab" aria-selected="false" aria-controls="ppo-stage-act" data-ppo-step="1"><span>02</span> Act + step</button>
    <button type="button" role="tab" aria-selected="false" aria-controls="ppo-stage-store" data-ppo-step="2"><span>03</span> Store</button>
    <button type="button" role="tab" aria-selected="false" aria-controls="ppo-stage-advantage" data-ppo-step="3"><span>04</span> Advantage</button>
    <button type="button" role="tab" aria-selected="false" aria-controls="ppo-stage-sample" data-ppo-step="4"><span>05</span> Sample</button>
    <button type="button" role="tab" aria-selected="false" aria-controls="ppo-stage-optimize" data-ppo-step="5"><span>06</span> Optimize</button>
    <button type="button" role="tab" aria-selected="false" aria-controls="ppo-stage-repeat" data-ppo-step="6"><span>07</span> Repeat</button>
  </div>

  <div class="ppo-stepper__panels">
    <section class="ppo-stage is-active" id="ppo-stage-observe" role="tabpanel" tabindex="0" data-ppo-panel="0">
      <div class="ppo-stage__copy">
        <p class="ppo-stage__label">01 · ObservationManager</p>
        <h4>Observe every world as one batch</h4>
        <p>Isaac Lab computes named observation groups for all <span data-ppo-envs>4,096</span> environments. Terms such as joint state, base velocity, commands, and previous actions are normally concatenated along the feature axis.</p>
        <p>The actor and critic can select different groups. That lets the critic use privileged simulation state while the deployed actor sees only policy observations.</p>
        <code>policy: [N, O<sub>policy</sub>] · critic: [N, O<sub>critic</sub>]</code>
      </div>
      <div class="ppo-observe-visual" aria-hidden="true">
        <div class="ppo-worlds" data-ppo-world-grid></div>
        <p><span data-ppo-envs>4,096</span> synchronized world slots</p>
        <div class="ppo-tensor-sheet">
          <span>TensorDict · batch [N]</span>
          <b>policy</b><i></i><i></i><i></i><i></i><i></i>
          <b>critic</b><i></i><i></i><i></i><i></i><i></i>
        </div>
      </div>
    </section>

    <section class="ppo-stage" id="ppo-stage-act" role="tabpanel" tabindex="0" data-ppo-panel="1">
      <div class="ppo-stage__copy">
        <p class="ppo-stage__label">02 · Actor, critic, environment</p>
        <h4>Sample once, step every world</h4>
        <p>One batched actor samples <code>[N, A]</code> actions—not one network per robot. Before the simulator moves, PPO keeps the action, value estimate, old log-probability, and distribution parameters beside the current observation.</p>
        <p>Isaac Lab transforms each policy action, advances several physics ticks, computes reward and termination, resets finished slots, and returns the next observation.</p>
      </div>
      <div class="ppo-act-visual" aria-hidden="true">
        <div class="ppo-flow-chip">obs<sub>t</sub><small>[N, O]</small></div>
        <span aria-hidden="true">→</span>
        <div class="ppo-model-pair">
          <div><small>actor</small><strong>π<sub>θ</sub></strong><span>a<sub>t</sub>, log π<sub>old</sub></span></div>
          <div><small>critic</small><strong>V<sub>θ</sub></strong><span>V<sub>t</sub></span></div>
        </div>
        <span aria-hidden="true">→</span>
        <div class="ppo-sim-chip"><small>Isaac Lab</small><strong>ActionManager</strong><span>decimation × physics</span></div>
        <div class="ppo-act-visual__return"><span>reward r<sub>t</sub></span><span>done d<sub>t</sub></span><span>obs<sub>t+1</sub></span><b aria-hidden="true">↩</b></div>
      </div>
    </section>

    <section class="ppo-stage" id="ppo-stage-store" role="tabpanel" tabindex="0" data-ppo-panel="2">
      <div class="ppo-stage__copy">
        <p class="ppo-stage__label">03 · RolloutStorage</p>
        <h4>Write one complete time row</h4>
        <p>The post-step reward and done flag complete the transition that began with <code>obs<sub>t</sub></code>. Storage copies one row across every environment, then advances its time cursor.</p>
        <p>After <span data-ppo-horizon>24</span> environment steps, the rectangle holds <strong data-ppo-transitions>98,304</strong> transitions. RSL-RL does not keep a separate <code>next_obs</code> tensor.</p>
      </div>
      <div class="ppo-buffer-visual" aria-hidden="true">
        <div class="ppo-buffer-visual__axis"><span>time · T</span><b>environment · N →</b></div>
        <div class="ppo-buffer-grid" data-ppo-buffer-grid></div>
        <div class="ppo-buffer-visual__fields"><span>obs</span><span>action</span><span>reward</span><span>done</span><span>V</span><span>log π<sub>old</sub></span></div>
        <p>[<span data-ppo-horizon>24</span>, <span data-ppo-envs>4,096</span>, …]</p>
      </div>
    </section>

    <section class="ppo-stage" id="ppo-stage-advantage" role="tabpanel" tabindex="0" data-ppo-panel="3">
      <div class="ppo-stage__copy">
        <p class="ppo-stage__label">04 · Bootstrap + GAE</p>
        <h4>Walk backward to assign credit</h4>
        <p>The critic first evaluates the final observation <code>obs<sub>T</sub></code>. Generalized Advantage Estimation then walks from the last stored step toward the first, mixing one-step surprise with later evidence.</p>
        <p>A done mask stops information from leaking across asynchronous episode resets. The resulting return becomes the critic target; normalized advantage tells the actor which sampled actions were better or worse than expected.</p>
      </div>
      <div class="ppo-gae-visual">
        <div class="ppo-formula"><span>TD surprise</span><code>δ<sub>t</sub> = r<sub>t</sub> + γ(1−d<sub>t</sub>)V<sub>t+1</sub> − V<sub>t</sub></code></div>
        <div class="ppo-formula"><span>GAE</span><code>A<sub>t</sub> = δ<sub>t</sub> + γλ(1−d<sub>t</sub>)A<sub>t+1</sub></code></div>
        <div class="ppo-gae-visual__steps" aria-label="Advantages are calculated backward from the final step">
          <i>t</i><i>t+1</i><i>t+2</i><i>…</i><i>T−1</i><b aria-hidden="true">← compute backward</b>
        </div>
      </div>
    </section>

    <section class="ppo-stage" id="ppo-stage-sample" role="tabpanel" tabindex="0" data-ppo-panel="4">
      <div class="ppo-stage__copy">
        <p class="ppo-stage__label">05 · Flatten + shuffle</p>
        <h4>Turn the rectangle into mini-batches</h4>
        <p>For a feed-forward policy, storage flattens time and environment: <code>[T, N, …] → [B, …]</code>. One random permutation divides <strong data-ppo-transitions>98,304</strong> samples into four mini-batches of <strong data-ppo-minibatch>24,576</strong>.</p>
        <p>The same shuffled partition is revisited for five learning epochs. Recurrent policies take a different route: trajectories are split at done flags, padded, and sampled with masks.</p>
      </div>
      <div class="ppo-sample-visual" aria-hidden="true">
        <div class="ppo-sample-visual__label"><span>rollout order</span><b>[T × N]</b></div>
        <div class="ppo-sample-source" data-ppo-sample-source></div>
        <div class="ppo-shuffle-arrow"><span>randperm</span><b>⇣</b></div>
        <div class="ppo-sample-visual__label"><span>mini-batch 1 of 4</span><b>[<span data-ppo-minibatch>24,576</span>, …]</b></div>
        <div class="ppo-minibatch" data-ppo-minibatch-grid></div>
      </div>
    </section>

    <section class="ppo-stage" id="ppo-stage-optimize" role="tabpanel" tabindex="0" data-ppo-panel="5">
      <div class="ppo-stage__copy">
        <p class="ppo-stage__label">06 · PPO update</p>
        <h4>Improve the policy with a clipped incentive</h4>
        <p>The current actor re-evaluates each stored action. The ratio <code>r = exp(log π<sub>θ</sub> − log π<sub>old</sub>)</code> measures how much its probability changed. PPO clips the incentive outside <code>1 ± ε</code>, usually 0.8–1.2 when ε = 0.2.</p>
        <p>Each mini-batch also trains the critic toward the return target and rewards policy entropy. Gradient clipping precedes one optimizer step: four batches × five epochs = <strong>20 updates</strong>.</p>
      </div>
      <div class="ppo-loss-visual" aria-hidden="true">
        <div class="ppo-ratio">
          <span class="ppo-ratio__zone"></span>
          <span class="ppo-ratio__tick ppo-ratio__tick--left">0.8</span>
          <span class="ppo-ratio__tick ppo-ratio__tick--mid">1.0</span>
          <span class="ppo-ratio__tick ppo-ratio__tick--right">1.2</span>
          <i></i>
        </div>
        <div class="ppo-loss-stack">
          <div><span>actor</span><strong>clipped surrogate</strong><small>do not move too far</small></div>
          <b>+</b>
          <div><span>critic</span><strong>value error</strong><small>fit return targets</small></div>
          <b>−</b>
          <div><span>explore</span><strong>entropy</strong><small>retain action spread</small></div>
        </div>
      </div>
    </section>

    <section class="ppo-stage" id="ppo-stage-repeat" role="tabpanel" tabindex="0" data-ppo-panel="6">
      <div class="ppo-stage__copy">
        <p class="ppo-stage__label">07 · Fresh on-policy data</p>
        <h4>Reset the cursor, keep the weights</h4>
        <p>The actor and critic were updated in place, so the next call to <code>act()</code> automatically uses the new parameters. Rollout storage only resets its write cursor; the old contents will be overwritten.</p>
        <p>This is the meaning of <em>on-policy</em> here: the next optimization phase learns from a new rollout generated by the policy that now exists—not from an ever-growing replay database.</p>
      </div>
      <div class="ppo-repeat-visual" aria-hidden="true">
        <div><span>new θ</span><strong>Actor</strong><small>batched inference</small></div>
        <b aria-hidden="true">→</b>
        <div><span>fresh B</span><strong>Rollout</strong><small><span data-ppo-transitions>98,304</span> transitions</small></div>
        <b aria-hidden="true">→</b>
        <div><span>20 steps</span><strong>Update</strong><small>4 batches × 5 epochs</small></div>
        <p><span aria-hidden="true">↖</span> repeat until the training iteration budget is exhausted</p>
      </div>
    </section>
  </div>

  <div class="ppo-stepper__controls">
    <button type="button" disabled data-ppo-prev><span aria-hidden="true">←</span> Previous</button>
    <p><span data-ppo-stage-name>Observe</span> · use the stage tabs or arrow keys</p>
    <button type="button" data-ppo-next>Next: Act + step <span aria-hidden="true">→</span></button>
  </div>
  <p class="ppo-sr-status" aria-live="polite" data-ppo-status></p>
</div>

<noscript>
  <p class="notebook-callout">JavaScript is off, so all seven stages are shown in sequence instead of as tabs.</p>
</noscript>

The important separation is temporal. During collection, RSL-RL runs the actor and critic under inference mode while Isaac Lab advances the world. During learning, the simulator stops advancing while the optimizer revisits the fixed rollout. Network weights do not change halfway through a rollout.

## The collection phase in code

The current on-policy runner reduces to this shape:

<pre><code>obs = env.get_observations()

for t in range(num_steps_per_env):
    actions = ppo.act(obs)                # caches obs, action, V, old log π
    next_obs, reward, done, extras = env.step(actions)
    ppo.process_env_step(next_obs, reward, done, extras)
    obs = next_obs

ppo.compute_returns(obs)                  # bootstrap + backward GAE
ppo.update()                              # shuffled mini-batches, then clear cursor</code></pre>

There is a subtle but useful detail in that ordering. The transition is assembled across two calls. <code>act()</code> records what was known before physics: <code>obs<sub>t</sub></code>, <code>a<sub>t</sub></code>, <code>V<sub>t</sub></code>, the old action log-probability, and the action distribution parameters. After Isaac Lab steps, <code>process_env_step()</code> attaches <code>r<sub>t</sub></code> and <code>d<sub>t</sub></code> and copies the finished record into time row <code>t</code>.

Isaac Lab resets ended environment slots before returning the next observation. For a slot whose done flag is true, <code>obs<sub>t+1</sub></code> is already the initial state of its next episode. The done mask is therefore essential when GAE runs backward.

## What is actually in the rollout buffer?

With <code>N</code> environments, <code>T</code> rollout steps, action width <code>A</code>, and observation-group width <code>O<sub>g</sub></code>, the core tensors are:

<div class="notebook-table-scroll" role="region" aria-label="RSL-RL rollout storage tensor shapes" tabindex="0" markdown="1">

| Field | Shape | Why PPO keeps it |
| --- | --- | --- |
| Observation group <code>g</code> | <code>[T, N, O<sub>g</sub>]</code> | Re-run actor and critic during learning |
| Sampled action | <code>[T, N, A]</code> | Score the exact behavior action again |
| Reward, done | <code>[T, N, 1]</code> | Build return and stop credit at episode boundaries |
| Old value | <code>[T, N, 1]</code> | Compute advantage and optionally clip value updates |
| Old action log-probability | <code>[T, N, 1]</code> | Form the PPO probability ratio |
| Old Gaussian mean and standard deviation | <code>[T, N, A]</code> | Measure analytic KL for adaptive learning rate |
| Return, advantage | <code>[T, N, 1]</code> | Critic target and actor learning signal |

</div>

This structure is usually called <code>RolloutStorage</code> rather than a replay buffer for a reason. It holds one fixed-horizon, on-policy batch. After the PPO update, only the write cursor is reset, and the next rollout overwrites the old one.

## GAE turns rewards into a learning signal

Rewards alone do not say whether one action was surprisingly good given the state. The critic supplies a baseline. RSL-RL computes a temporal-difference residual and then accumulates it backward:

<div class="ppo-equations" role="group" aria-label="Generalized Advantage Estimation equations">
  <p><span>TD residual</span><code>δ<sub>t</sub> = r<sub>t</sub> + γ(1 − d<sub>t</sub>)V<sub>t+1</sub> − V<sub>t</sub></code></p>
  <p><span>Advantage</span><code>A<sub>t</sub> = δ<sub>t</sub> + γλ(1 − d<sub>t</sub>)A<sub>t+1</sub></code></p>
  <p><span>Return target</span><code>R<sub>t</sub> = A<sub>t</sub> + V<sub>t</sub></code></p>
</div>

The <code>γ</code> term discounts distant outcomes. <code>λ</code> controls how much later evidence flows backward: lower values lean toward low-variance one-step estimates; higher values use longer credit chains. Before optimization, advantages are normalized over the rollout by default.

Timeouts need separate care. For a slot marked <code>time_outs</code>, RSL-RL v5.4.1 applies <code>r<sub>t</sub> ← r<sub>t</sub> + γV<sub>t</sub></code> before storage, using the value cached before the step—not a value of a terminal observation. The done mask remains set, so the backward GAE recursion still cannot cross the reset boundary.

## Sampling changes the view, not the data

The rollout rectangle is time-major because that makes collection and GAE natural. A feed-forward policy does not need that structure during the gradient update, so RSL-RL flattens <code>[T, N, …]</code> to <code>[B, …]</code>, where <code>B = T × N</code>, then selects a random permutation.

For the Go2 example:

- <strong>Rollout:</strong> <code>24 × 4,096 = 98,304</code> transitions
- <strong>Mini-batch:</strong> <code>98,304 ÷ 4 = 24,576</code> transitions
- <strong>Optimizer steps:</strong> <code>4 mini-batches × 5 epochs = 20</code>
- <strong>Reuse:</strong> every transition participates once per epoch, then is discarded after the update

The current feed-forward generator creates one permutation and reuses that partition across the learning epochs. If the batch is not divisible by the mini-batch count, the remainder is omitted; common Isaac Lab configurations choose divisible sizes. Recurrent policies cannot freely flatten across resets, so their generator preserves padded trajectory fragments and accompanying masks.

## What PPO clips—and what it does not

For every mini-batch, the actor calculates a new log-probability for the action that the old policy actually sampled. Their difference becomes a ratio:

<div class="ppo-clip-explainer">
  <div>
    <span>probability ratio</span>
    <code>r<sub>t</sub>(θ) = exp(log π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) − log π<sub>old</sub>(a<sub>t</sub>|s<sub>t</sub>))</code>
  </div>
  <p><strong>r = 1</strong> means the action is just as likely now. PPO limits the benefit of moving this ratio beyond <strong>1 ± ε</strong>; it does not simply clamp every gradient or replace the sampled action.</p>
</div>

The minimized training loss combines three signals:

1. **Clipped surrogate loss:** improve probability for positive-advantage actions and reduce it for negative-advantage actions, without earning extra objective improvement for moving too far.
2. **Value loss:** move the critic toward <code>R<sub>t</sub></code>; the Go2 configuration used here enables clipped value loss.
3. **Entropy bonus:** resist collapsing the Gaussian action distribution too quickly.

After backpropagation, actor and critic gradients are norm-clipped and the optimizer updates both networks. An optional adaptive schedule compares the old and current distributions with analytic KL and changes the learning rate; it does not stop the epoch early.

## Three details that make the mental model click

### The old policy is stored as numbers

Thousands of Isaac Lab environments do not each own a policy copy. One actor handles the whole <code>[N, …]</code> batch. PPO's behavior-policy reference comes from the stored old log-probabilities and distribution parameters. The same actor module is updated in place before the next rollout.

### “Done” belongs to the transition that just ended

Isaac Lab steps, computes reward and termination, resets ended slots, and only then computes the returned observation. Storage keeps the pre-step observation and the post-step reward/done pair; it does not need a second full observation tensor for every transition.

### Action clipping is a different clip

PPO's ratio clip constrains the learning objective. If the Isaac Lab RSL-RL wrapper is also configured to clip actions, that clamp happens on the tensor sent to the environment. The stored action and its log-probability still describe the actor's original sample.

## Takeaway

One PPO iteration in Isaac Lab is a disciplined change of tensor layout:

1. Isaac Lab produces observation groups shaped by parallel environments.
2. The actor and critic add action, value, and old-policy statistics.
3. Environment feedback completes one row in <code>[T, N, …]</code> rollout storage.
4. A final value bootstrap and backward GAE create returns and advantages.
5. Feed-forward training flattens to <code>[T × N, …]</code> and selects mini-batches.
6. PPO revisits the stored actions, clips the actor's optimization incentive, updates actor and critic, and discards the rollout.

The simulator makes experience wide; time makes it tall; GAE makes it useful; mini-batch sampling makes it trainable. Then the updated policy goes back to the worlds and earns the next rectangle.

## Sources and further reading

- [Isaac Lab's RSL-RL 5.4.1 dependency pin at this source snapshot](https://github.com/isaac-sim/IsaacLab/blob/8c3bc5e8e8dd4553fa160a11836e90922832c968/pyproject.toml#L80-L90)
- [Isaac Lab ObservationManager](https://github.com/isaac-sim/IsaacLab/blob/8c3bc5e8e8dd4553fa160a11836e90922832c968/source/isaaclab/isaaclab/managers/observation_manager.py)
- [Isaac Lab manager-based environment step](https://github.com/isaac-sim/IsaacLab/blob/8c3bc5e8e8dd4553fa160a11836e90922832c968/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)
- [Isaac Lab's RSL-RL vectorized wrapper](https://github.com/isaac-sim/IsaacLab/blob/8c3bc5e8e8dd4553fa160a11836e90922832c968/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)
- [Go2 PPO runner configuration](https://github.com/isaac-sim/IsaacLab/blob/8c3bc5e8e8dd4553fa160a11836e90922832c968/source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/agents/rsl_rl_ppo_cfg.py#L12-L38)
- [Default 4,096-environment velocity scene](https://github.com/isaac-sim/IsaacLab/blob/8c3bc5e8e8dd4553fa160a11836e90922832c968/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py#L340-L350)
- [RSL-RL v5.4.1 on-policy runner](https://github.com/leggedrobotics/rsl_rl/blob/v5.4.1/rsl_rl/runners/on_policy_runner.py#L56-L109)
- [RSL-RL v5.4.1 rollout storage](https://github.com/leggedrobotics/rsl_rl/blob/v5.4.1/rsl_rl/storage/rollout_storage.py#L125-L255)
- [RSL-RL v5.4.1 PPO implementation](https://github.com/leggedrobotics/rsl_rl/blob/v5.4.1/rsl_rl/algorithms/ppo.py#L115-L343)
- [Original PPO paper](https://arxiv.org/abs/1707.06347)
