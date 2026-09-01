---
title: "PPO, visualized inside Isaac Lab"
subtitle: "Follow one training iteration from thousands of parallel robot worlds to a shuffled PPO update."
description: "An interactive, step-by-step visual guide to PPO in Isaac Lab: observations, actions, rollout storage, GAE, mini-batches, and the clipped policy update."
excerpt: "An interactive, step-by-step visual guide to PPO in Isaac Lab: observations, actions, rollout storage, GAE, mini-batches, and the clipped policy update."
date: 2026-09-01 01:00:00 +0000
last_modified_at: 2026-09-01 08:47:00 +0000
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

<figure class="ppo-cycle" aria-labelledby="ppo-cycle-title">
  <header class="ppo-cycle__header">
    <div>
      <p>One iteration · the complete data path</p>
      <h3 id="ppo-cycle-title">Collect → build targets → learn → repeat</h3>
    </div>
    <div class="ppo-legend" aria-label="Diagram color key">
      <span><i class="ppo-swatch ppo-swatch--policy"></i> policy + forward tensors</span>
      <span><i class="ppo-swatch ppo-swatch--sim"></i> simulator + boundaries</span>
      <span><i class="ppo-swatch ppo-swatch--learn"></i> critic + learned targets</span>
      <span><i class="ppo-swatch ppo-swatch--storage"></i> fixed storage</span>
    </div>
  </header>

  <div class="ppo-cycle__phasebar" aria-hidden="true">
    <span class="ppo-cycle__phase ppo-cycle__phase--collect">Collect · repeat T times</span>
    <span class="ppo-cycle__phase ppo-cycle__phase--targets">Build targets · once</span>
    <span class="ppo-cycle__phase ppo-cycle__phase--learn">Learn · 4 batches × 5 epochs</span>
    <span class="ppo-cycle__phase ppo-cycle__phase--repeat">Outer loop</span>
  </div>

  <ol class="ppo-cycle__nodes">
    <li data-kind="policy"><small>Collect</small><span>01</span><strong>Observe</strong><code>[N, O]</code></li>
    <li data-kind="policy"><small>Collect</small><span>02</span><strong>Act + step</strong><code>a, V, r, d</code></li>
    <li data-kind="storage"><small>Collect</small><span>03</span><strong>Store</strong><code>[T, N, …]</code></li>
    <li data-kind="learn"><small>Targets</small><span>04</span><strong>GAE</strong><code>A, R</code></li>
    <li data-kind="storage"><small>Learn</small><span>05</span><strong>Sample</strong><code>[M, …]</code></li>
    <li data-kind="learn"><small>Learn</small><span>06</span><strong>Optimize</strong><code>θ → θ′</code></li>
    <li data-kind="sim"><small>Repeat</small><span>07</span><strong>Fresh rollout</strong><code>cursor → 0</code></li>
  </ol>

  <div class="ppo-cycle__loop">
    <span><strong>Weights stay fixed while collecting.</strong> They change after every mini-batch while learning.</span>
    <span aria-hidden="true">θ′ loops back to stage 01 ↺</span>
  </div>
  <figcaption>Only stage 02 advances physics. Stages 01–03 collect transition fields under fixed weights; stage 04 writes advantages and returns once; stages 05–06 revisit that complete frozen batch. Stage 07 keeps the updated networks and replaces the rollout.</figcaption>
</figure>

<div class="ppo-stepper ppo-stepper--crisp" data-ppo-stepper aria-labelledby="ppo-stepper-title">
  <div class="ppo-stepper__topline">
    <div>
      <p class="ppo-stepper__eyebrow">Interactive walkthrough · trace every tensor</p>
      <h3 id="ppo-stepper-title">Open one stage at a time</h3>
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
      <span>One mini-batch · M = B ÷ 4</span>
      <strong data-ppo-minibatch>24,576</strong>
      <small>samples · 20 optimizer steps</small>
    </div>
  </div>

  <div class="ppo-stepper__nav" role="tablist" aria-label="PPO iteration stages">
    <button type="button" class="is-active" role="tab" aria-selected="true" aria-controls="ppo-stage-observe" id="ppo-stage-tab-0" data-ppo-step="0"><span>01</span><small>Collect</small><b>Observe</b></button>
    <button type="button" role="tab" aria-selected="false" aria-controls="ppo-stage-act" tabindex="-1" id="ppo-stage-tab-1" data-ppo-step="1"><span>02</span><small>Collect</small><b>Act + step</b></button>
    <button type="button" role="tab" aria-selected="false" aria-controls="ppo-stage-store" tabindex="-1" id="ppo-stage-tab-2" data-ppo-step="2"><span>03</span><small>Collect</small><b>Store</b></button>
    <button type="button" role="tab" aria-selected="false" aria-controls="ppo-stage-advantage" tabindex="-1" id="ppo-stage-tab-3" data-ppo-step="3"><span>04</span><small>Targets</small><b>Advantage</b></button>
    <button type="button" role="tab" aria-selected="false" aria-controls="ppo-stage-sample" tabindex="-1" id="ppo-stage-tab-4" data-ppo-step="4"><span>05</span><small>Learn</small><b>Sample</b></button>
    <button type="button" role="tab" aria-selected="false" aria-controls="ppo-stage-optimize" tabindex="-1" id="ppo-stage-tab-5" data-ppo-step="5"><span>06</span><small>Learn</small><b>Optimize</b></button>
    <button type="button" role="tab" aria-selected="false" aria-controls="ppo-stage-repeat" tabindex="-1" id="ppo-stage-tab-6" data-ppo-step="6"><span>07</span><small>Repeat</small><b>Fresh rollout</b></button>
  </div>

  <div class="ppo-stepper__panels">
    <section class="ppo-stage is-active" id="ppo-stage-observe" role="tabpanel" tabindex="0" aria-labelledby="ppo-stage-tab-0" data-ppo-panel="0">
      <div class="ppo-stage__intro">
        <div class="ppo-stage__copy">
          <p class="ppo-stage__label">01 · ObservationManager</p>
          <h4>Turn N worlds into model-ready rows</h4>
          <p>Isaac Lab computes every configured observation term across all environments at once, processes each term, and concatenates terms along the feature axis. One row belongs to one environment slot.</p>
        </div>
        <dl class="ppo-stage__io">
          <div><dt>Input</dt><dd>simulator state + commands</dd></div>
          <div><dt>Transform</dt><dd>compute → modify → noise → clip → scale</dd></div>
          <div><dt>Output</dt><dd><code>TensorDict · batch [N]</code></dd></div>
          <div><dt>Invariant</dt><dd>one shared network, N rows</dd></div>
        </dl>
      </div>

      <figure class="ppo-diagram" aria-labelledby="ppo-observe-caption">
        <div class="ppo-flow ppo-flow--observe">
          <div class="ppo-node" data-kind="sim">
            <span>Isaac Lab · vectorized scene</span>
            <strong><span data-ppo-envs>4,096</span> synchronized worlds</strong>
            <div class="ppo-token-row"><code>base velocity</code><code>joint state</code><code>commands</code><code>previous action</code></div>
          </div>
          <div class="ppo-arrow"><span>read all N slots</span><b aria-hidden="true">→</b></div>
          <div class="ppo-node" data-kind="policy">
            <span>ObservationManager</span>
            <strong>Process each named term</strong>
            <div class="ppo-process-row"><i>compute</i><b>›</b><i>modifiers</i><b>›</b><i>noise</i><b>›</b><i>clip</i><b>›</b><i>scale</i></div>
            <small>configured terms concatenate on the last dimension</small>
          </div>
          <div class="ppo-arrow"><span>group by consumer</span><b aria-hidden="true">→</b></div>
          <div class="ppo-node-stack">
            <div class="ppo-node" data-kind="policy">
              <span>Actor input · Go2 default</span>
              <strong>policy group</strong>
              <code>[N, O<sub>policy</sub>]</code>
            </div>
            <div class="ppo-node" data-kind="learn">
              <span>Critic input</span>
              <strong>Go2 reuses policy</strong>
              <code>[N, O<sub>policy</sub>]</code>
              <small>optional asymmetric setups use a separate privileged critic group</small>
            </div>
          </div>
        </div>
        <figcaption id="ppo-observe-caption"><strong>Shape rule:</strong> the leading dimension is always the environment batch <code>N</code>; observation terms only change the feature width <code>O</code>.</figcaption>
      </figure>
    </section>

    <section class="ppo-stage" id="ppo-stage-act" role="tabpanel" tabindex="0" aria-labelledby="ppo-stage-tab-1" data-ppo-panel="1">
      <div class="ppo-stage__intro">
        <div class="ppo-stage__copy">
          <p class="ppo-stage__label">02 · Actor, critic, environment</p>
          <h4>Cache first, then advance physics</h4>
          <p>The actor samples one action row per environment and the critic predicts one value. PPO freezes that pre-step snapshot before Isaac Lab transforms actions, runs four physics ticks, computes feedback, and resets only finished slots.</p>
        </div>
        <dl class="ppo-stage__io">
          <div><dt>Input</dt><dd><code>obs<sub>t</sub> · [N, O]</code></dd></div>
          <div><dt>Policy</dt><dd><code>a<sub>t</sub> · [N, A]</code></dd></div>
          <div><dt>Critic</dt><dd><code>V<sub>t</sub> · [N, 1]</code></dd></div>
          <div><dt>Weights</dt><dd>fixed for all T collection steps</dd></div>
        </dl>
      </div>

      <figure class="ppo-diagram" aria-labelledby="ppo-act-caption">
        <div class="ppo-act-models">
          <div class="ppo-node ppo-node--compact" data-kind="policy">
            <span>Observation at t</span>
            <strong>policy / critic rows</strong>
            <code>[N, O]</code>
          </div>
          <div class="ppo-arrow"><span>one batched forward per model</span><b aria-hidden="true">→</b></div>
          <div class="ppo-model-split">
            <div class="ppo-node" data-kind="policy">
              <span>Actor · π<sub>θ</sub></span>
              <strong>sample Gaussian action</strong>
              <code>a<sub>t</sub> [N,A]</code>
              <small>also μ, σ and summed log π<sub>old</sub></small>
            </div>
            <div class="ppo-node" data-kind="learn">
              <span>Critic · V<sub>φ</sub></span>
              <strong>estimate state value</strong>
              <code>V<sub>t</sub> [N,1]</code>
            </div>
          </div>
          <div class="ppo-arrow"><span>cache before physics</span><b aria-hidden="true">→</b></div>
          <div class="ppo-node" data-kind="storage">
            <span>Pending transition</span>
            <strong>obs, a, V, log π<sub>old</sub>, μ, σ</strong>
            <small>the behavior-policy snapshot stays frozen</small>
          </div>
        </div>

        <div class="ppo-act-environment">
          <div class="ppo-node ppo-node--compact" data-kind="policy">
            <span>Original sample</span>
            <strong>a<sub>t</sub></strong>
            <small>this is what storage keeps</small>
          </div>
          <div class="ppo-arrow"><span>optional wrapper clamp</span><b aria-hidden="true">→</b></div>
          <div class="ppo-node ppo-node--compact" data-kind="sim">
            <span>ActionManager</span>
            <strong>transform action</strong>
          </div>
          <div class="ppo-arrow"><span>decimation = 4</span><b aria-hidden="true">→</b></div>
          <div class="ppo-node ppo-node--compact" data-kind="sim">
            <span>Simulator</span>
            <strong>physics × 4</strong>
            <small>terminate → reward → reset done slots</small>
          </div>
          <div class="ppo-arrow"><span>return</span><b aria-hidden="true">→</b></div>
          <div class="ppo-node ppo-node--compact" data-kind="sim">
            <span>Feedback</span>
            <strong>r<sub>t</sub>, done<sub>t</sub>, obs<sub>t+1</sub>, extras</strong>
            <small>done = terminated OR truncated</small>
          </div>
        </div>
        <figcaption id="ppo-act-caption"><strong>Boundary rule:</strong> for a done slot, <code>obs<sub>t+1</sub></code> is already the reset-state observation. The stored action remains the actor’s original sample, not the optional action-clamped tensor.</figcaption>
      </figure>
    </section>

    <section class="ppo-stage" id="ppo-stage-store" role="tabpanel" tabindex="0" aria-labelledby="ppo-stage-tab-2" data-ppo-panel="2">
      <div class="ppo-stage__intro">
        <div class="ppo-stage__copy">
          <p class="ppo-stage__label">03 · RolloutStorage</p>
          <h4>Complete one time row, then move the cursor</h4>
          <p><code>process_env_step()</code> attaches post-step reward and done to the cached snapshot. After <span data-ppo-horizon>24</span> rows, storage contains <strong data-ppo-transitions>98,304</strong> transitions arranged time-major.</p>
        </div>
        <dl class="ppo-stage__io">
          <div><dt>Before step</dt><dd>obs, action, value, old policy stats</dd></div>
          <div><dt>After step</dt><dd>reward + done</dd></div>
          <div><dt>One row</dt><dd><code>[N, …]</code></dd></div>
          <div><dt>Full rollout</dt><dd><code>[T, N, …]</code></dd></div>
        </dl>
      </div>

      <figure class="ppo-diagram" aria-labelledby="ppo-store-caption">
        <div class="ppo-record-assembly">
          <div class="ppo-node" data-kind="storage">
            <span>Cached by act()</span>
            <strong>pre-step record</strong>
            <div class="ppo-token-row"><code>obs [N,O]</code><code>action [N,A]</code><code>V [N,1]</code><code>log π [N]</code><code>μ, σ [N,A]</code></div>
          </div>
          <div class="ppo-plus" aria-hidden="true">+</div>
          <div class="ppo-node" data-kind="sim">
            <span>Attached by process_env_step()</span>
            <strong>post-step feedback</strong>
            <div class="ppo-token-row"><code>reward [N]</code><code>done [N]</code></div>
            <small>timeout first adds γ times cached pre-step V<sub>t</sub> to reward; done stays 1</small>
          </div>
          <div class="ppo-arrow"><span>copy row t</span><b aria-hidden="true">→</b></div>
          <div class="ppo-node ppo-node--compact" data-kind="storage">
            <span>Cursor</span>
            <strong>t → t + 1</strong>
            <code>scalar fields reshape to [N,1]</code>
          </div>
        </div>

        <div class="ppo-buffer-map">
          <div class="ppo-buffer-matrix" role="img" aria-label="Time-major rollout matrix with T rows and N environment columns; every cell is one transition record">
            <b>time ↓ / env →</b><b>n = 0</b><b>n = 1</b><b>…</b><b>n = N−1</b>
            <b>t = 0</b><span>[0,0]</span><span>[0,1]</span><span>…</span><span>[0,N−1]</span>
            <b>t = 1</b><span>[1,0]</span><span>[1,1]</span><span>…</span><span>[1,N−1]</span>
            <b>⋮</b><span>⋮</span><span>⋮</span><span>⋱</span><span>⋮</span>
            <b class="is-current">t = T−1</b><span class="is-current">[T−1,0]</span><span class="is-current">[T−1,1]</span><span class="is-current">…</span><span class="is-current">[T−1,N−1]</span>
          </div>
          <aside class="ppo-buffer-record">
            <span>Every [t,n] cell indexes the same tensor channels</span>
            <ul>
              <li><code>obs · O<sub>g</sub></code></li>
              <li><code>action · A</code></li>
              <li><code>reward, done, V, log π · 1</code></li>
              <li><code>μ, σ · A</code></li>
            </ul>
            <strong><span data-ppo-horizon>24</span> × <span data-ppo-envs>4,096</span> = <span data-ppo-transitions>98,304</span></strong>
          </aside>
        </div>
        <figcaption id="ppo-store-caption"><strong>Storage rule:</strong> fields are separate tensors that share the same <code>[T,N]</code> leading axes. RSL-RL does not allocate a separate <code>next_obs</code> tensor.</figcaption>
      </figure>
    </section>

    <section class="ppo-stage" id="ppo-stage-advantage" role="tabpanel" tabindex="0" aria-labelledby="ppo-stage-tab-3" data-ppo-panel="3">
      <div class="ppo-stage__intro">
        <div class="ppo-stage__copy">
          <p class="ppo-stage__label">04 · Bootstrap + GAE</p>
          <h4>Walk backward without crossing resets</h4>
          <p>The critic evaluates the final observation once. GAE then scans from <code>T−1</code> to <code>0</code>, carrying later evidence backward until a done mask cuts the chain.</p>
        </div>
        <dl class="ppo-stage__io">
          <div><dt>Bootstrap</dt><dd><code>V(obs<sub>T</sub>) · [N,1]</code></dd></div>
          <div><dt>Direction</dt><dd><code>t = T−1 … 0</code></dd></div>
          <div><dt>Actor target</dt><dd><code>A · [T,N,1]</code></dd></div>
          <div><dt>Critic target</dt><dd><code>R · [T,N,1]</code></dd></div>
        </dl>
      </div>

      <figure class="ppo-diagram" aria-labelledby="ppo-gae-caption">
        <div class="ppo-bootstrap">
          <div class="ppo-node ppo-node--compact" data-kind="policy"><span>Final observation</span><strong>obs<sub>T</sub></strong><code>[N,O]</code></div>
          <div class="ppo-arrow ppo-arrow--dashed"><span>critic only</span><b aria-hidden="true">⇢</b></div>
          <div class="ppo-node ppo-node--compact" data-kind="learn"><span>Bootstrap</span><strong>V<sub>T</sub> = V(obs<sub>T</sub>)</strong><code>[N,1]</code></div>
        </div>

        <div class="ppo-backward">
          <div class="ppo-backward__label"><span>computed right → left</span><b aria-hidden="true">←</b></div>
          <div class="ppo-backward__steps">
            <span>t = 0</span><span>t = 1</span><i><b>done = 1</b><small>mask = 0 · stop</small></i><span>…</span><span>t = T−2</span><span>t = T−1</span>
          </div>
        </div>

        <div class="ppo-formula-grid">
          <div data-kind="learn"><span>1 · TD residual</span><code>δ<sub>t</sub> = r<sub>t</sub> + γ(1−d<sub>t</sub>)V<sub>t+1</sub> − V<sub>t</sub></code></div>
          <div data-kind="learn"><span>2 · Advantage</span><code>A<sub>t</sub> = δ<sub>t</sub> + γλ(1−d<sub>t</sub>)A<sub>t+1</sub></code></div>
          <div data-kind="learn"><span>3 · Return target</span><code>R<sub>t</sub> = A<sub>t</sub> + V<sub>t</sub></code></div>
        </div>

        <div class="ppo-output-pair">
          <div class="ppo-node ppo-node--compact" data-kind="policy"><span>Actor signal</span><strong>normalize A across T × N</strong><code>[T,N,1]</code></div>
          <div class="ppo-node ppo-node--compact" data-kind="learn"><span>Critic target</span><strong>keep return R unnormalized</strong><code>[T,N,1]</code></div>
        </div>
        <figcaption id="ppo-gae-caption"><strong>Boundary rule:</strong> <code>V<sub>t+1</sub></code> comes from the next stored value, except at the last row where <code>V<sub>T</sub></code> is used. Timeout rewards are already corrected, but the recursion is still masked.</figcaption>
      </figure>
    </section>

    <section class="ppo-stage" id="ppo-stage-sample" role="tabpanel" tabindex="0" aria-labelledby="ppo-stage-tab-4" data-ppo-panel="4">
      <div class="ppo-stage__intro">
        <div class="ppo-stage__copy">
          <p class="ppo-stage__label">05 · Flatten + shuffle</p>
          <h4>Change the view, not the data</h4>
          <p>For a feed-forward policy, every field needed by the PPO update is flattened in the same order, indexed by one random permutation, and split into four equal mini-batches. RSL-RL reuses that partition for five learning epochs.</p>
        </div>
        <dl class="ppo-stage__io">
          <div><dt>Before</dt><dd><code>[T, N, …]</code></dd></div>
          <div><dt>Flatten</dt><dd><code>[B, …] · B = T × N</code></dd></div>
          <div><dt>Mini-batch</dt><dd><code>[M, …] · M = B ÷ 4</code></dd></div>
          <div><dt>Reuse</dt><dd>4 batches × 5 epochs = 20 updates</dd></div>
        </dl>
      </div>

      <figure class="ppo-diagram" aria-labelledby="ppo-sample-caption">
        <div class="ppo-sample-transform">
          <div>
            <span>1 · time-major indices</span>
            <strong>[T,N]</strong>
            <div class="ppo-id-matrix" aria-label="Representative time and environment indices">
              <i>0,0</i><i>0,1</i><i>0,2</i><i>0,3</i>
              <i>1,0</i><i>1,1</i><i>1,2</i><i>1,3</i>
              <i>2,0</i><i>2,1</i><i>2,2</i><i>2,3</i>
            </div>
          </div>
          <div class="ppo-arrow"><span>flatten update fields</span><b aria-hidden="true">→</b></div>
          <div>
            <span>2 · flat positions</span>
            <strong>[B]</strong>
            <div class="ppo-id-vector"><i>0</i><i>1</i><i>2</i><i>3</i><i>N</i><i>N+1</i><i>…</i><i>B−1</i></div>
          </div>
          <div class="ppo-arrow"><span>randperm once</span><b aria-hidden="true">→</b></div>
          <div>
            <span>3 · shuffled indices</span>
            <strong>perm[B]</strong>
            <div class="ppo-id-vector ppo-id-vector--shuffled"><i>7</i><i>0</i><i>N+1</i><i>B−1</i><i>3</i><i>12</i><i>…</i><i>2</i></div>
          </div>
        </div>

        <div class="ppo-batches">
          <div><span>Mini-batch 1</span><strong>perm[0 : M]</strong><code>[<span data-ppo-minibatch>24,576</span>, …]</code></div>
          <div><span>Mini-batch 2</span><strong>perm[M : 2M]</strong><code>[<span data-ppo-minibatch>24,576</span>, …]</code></div>
          <div><span>Mini-batch 3</span><strong>perm[2M : 3M]</strong><code>[<span data-ppo-minibatch>24,576</span>, …]</code></div>
          <div><span>Mini-batch 4</span><strong>perm[3M : 4M]</strong><code>[<span data-ppo-minibatch>24,576</span>, …]</code></div>
        </div>

        <div class="ppo-epoch-loop">
          <span>Same frozen partition</span>
          <ol><li>epoch 1</li><li>epoch 2</li><li>epoch 3</li><li>epoch 4</li><li>epoch 5</li></ol>
          <strong>Each epoch visits all 4 mini-batches → 20 optimizer steps</strong>
        </div>

        <div class="ppo-field-strip">
          <span>Each sampled index selects every field together:</span>
          <code>obs [M,O]</code><code>action [M,A]</code><code>old V / log π / A / R [M,1]</code><code>old μ, σ [M,A]</code>
        </div>
        <figcaption id="ppo-sample-caption"><strong>Frozen-data rule:</strong> targets and old-policy statistics never change during the five epochs. If <code>B</code> is not divisible by four, the remainder is omitted; this Go2 batch divides exactly.</figcaption>
      </figure>
    </section>

    <section class="ppo-stage" id="ppo-stage-optimize" role="tabpanel" tabindex="0" aria-labelledby="ppo-stage-tab-5" data-ppo-panel="5">
      <div class="ppo-stage__intro">
        <div class="ppo-stage__copy">
          <p class="ppo-stage__label">06 · PPO update</p>
          <h4>Re-score stored actions, then update once</h4>
          <p>The current networks evaluate the exact actions collected earlier. PPO compares current and frozen behavior probabilities, combines actor, critic, and entropy terms, clips gradients, and runs one optimizer step per mini-batch.</p>
        </div>
        <dl class="ppo-stage__io">
          <div><dt>Frozen</dt><dd>action, old log π, old V, A, R, μ, σ</dd></div>
          <div><dt>Recomputed</dt><dd>new log π, value, entropy, μ, σ</dd></div>
          <div><dt>Clip</dt><dd>objective incentive at <code>1 ± ε</code></dd></div>
          <div><dt>Output</dt><dd>updated actor θ′ + critic φ′</dd></div>
        </dl>
      </div>

      <figure class="ppo-diagram" aria-labelledby="ppo-optimize-caption">
        <div class="ppo-optimize-flow">
          <div class="ppo-node" data-kind="storage">
            <span>Frozen mini-batch</span>
            <strong>behavior data + targets</strong>
            <div class="ppo-token-row"><code>obs</code><code>stored action</code><code>log π<sub>old</sub></code><code>V<sub>old</sub></code><code>A</code><code>R</code></div>
          </div>
          <div class="ppo-arrow"><span>re-score stored action</span><b aria-hidden="true">→</b></div>
          <div class="ppo-node" data-kind="policy">
            <span>Current actor + critic</span>
            <strong>new forward pass</strong>
            <div class="ppo-token-row"><code>log π<sub>θ</sub></code><code>V<sub>φ</sub></code><code>entropy</code><code>μ<sub>θ</sub>, σ<sub>θ</sub></code></div>
            <small>later mini-batches see the latest weights</small>
          </div>
          <div class="ppo-arrow"><span>compose loss</span><b aria-hidden="true">→</b></div>
          <div class="ppo-node" data-kind="learn">
            <span>Total loss</span>
            <strong>surrogate + c<sub>v</sub> value − c<sub>e</sub> entropy</strong>
            <small>Go2 also clips the value prediction around V<sub>old</sub></small>
          </div>
          <div class="ppo-arrow"><span>backward</span><b aria-hidden="true">→</b></div>
          <div class="ppo-node" data-kind="learn">
            <span>One optimizer step</span>
            <strong>zero grad → backward → clip → step</strong>
            <small>clip actor and critic gradient norms to 1.0</small>
          </div>
        </div>

        <div class="ppo-ratio-demo">
          <div>
            <span>Clipped surrogate · example with A &gt; 0</span>
            <code>r = exp(log π<sub>θ</sub> − log π<sub>old</sub>) = 1.28</code>
          </div>
          <div class="ppo-ratio-band" role="img" aria-label="Example probability ratio 1.28 lies above the positive-advantage clip boundary 1.2">
            <i class="ppo-ratio-band__safe"></i>
            <span class="ppo-ratio-band__tick ppo-ratio-band__tick--low">0.8</span>
            <span class="ppo-ratio-band__tick ppo-ratio-band__tick--one">1.0</span>
            <span class="ppo-ratio-band__tick ppo-ratio-band__tick--high">1.2</span>
            <b>1.28</b>
          </div>
          <div class="ppo-ratio-demo__result">
            <span>ratio remains 1.28</span>
            <b aria-hidden="true">→</b>
            <strong>objective uses min(1.28A, 1.20A) = 1.20A</strong>
          </div>
        </div>

        <div class="ppo-update-notes">
          <span><strong>Adaptive KL:</strong> analytic KL(old ‖ current) adjusts the learning rate before the step; it does not end the epoch.</span>
          <span><strong>20 total steps:</strong> 4 mini-batches × 5 epochs, each compared with the same frozen behavior statistics.</span>
        </div>
        <figcaption id="ppo-optimize-caption"><strong>Clip rule:</strong> PPO clips the improvement incentive in the surrogate objective. It does not clamp the actual policy ratio and is not a hard trust region.</figcaption>
      </figure>
    </section>

    <section class="ppo-stage" id="ppo-stage-repeat" role="tabpanel" tabindex="0" aria-labelledby="ppo-stage-tab-6" data-ppo-panel="6">
      <div class="ppo-stage__intro">
        <div class="ppo-stage__copy">
          <p class="ppo-stage__label">07 · Fresh on-policy data</p>
          <h4>Keep the weights, overwrite the rollout</h4>
          <p>The actor and critic were updated in place. Storage resets only its write cursor, unfinished episodes continue from <code>obs<sub>T</sub></code>, and the next <code>act()</code> starts a fresh rollout with the new policy.</p>
        </div>
        <dl class="ppo-stage__io">
          <div><dt>Keep</dt><dd>θ′, φ′, obs<sub>T</sub>, allocated tensors</dd></div>
          <div><dt>Reset</dt><dd>storage write cursor → 0</dd></div>
          <div><dt>Do not reset</dt><dd>all environments globally</dd></div>
          <div><dt>Replace</dt><dd>old rollout with fresh on-policy data</dd></div>
        </dl>
      </div>

      <figure class="ppo-diagram" aria-labelledby="ppo-repeat-caption">
        <div class="ppo-state-change">
          <div data-state="keep">
            <span>Keep across iterations</span>
            <strong>updated actor θ′ + critic φ′</strong>
            <strong>current obs<sub>T</sub></strong>
            <small>unfinished episodes continue; only done slots reset inside env.step()</small>
          </div>
          <div data-state="reset">
            <span>Reset logically</span>
            <strong>RolloutStorage cursor → 0</strong>
            <small>allocated tensors stay in memory and are overwritten</small>
          </div>
        </div>

        <ol class="ppo-outer-loop">
          <li data-kind="policy"><span>1</span><strong>act with θ′</strong><small>one batched policy</small></li>
          <li data-kind="sim"><span>2</span><strong>collect T new steps</strong><small><span data-ppo-envs>4,096</span> continuing world slots</small></li>
          <li data-kind="storage"><span>3</span><strong>fill fresh rollout</strong><small><span data-ppo-transitions>98,304</span> transitions</small></li>
          <li data-kind="learn"><span>4</span><strong>run 20 updates</strong><small>θ′ → θ″, φ′ → φ″</small></li>
        </ol>
        <div class="ppo-outer-loop__return"><span>No replay database · no separate old-policy network · no policy copy per robot</span><b aria-hidden="true">θ″ returns to step 1 ↺</b></div>
        <figcaption id="ppo-repeat-caption"><strong>On-policy rule:</strong> once the update finishes, the old rollout is logically discarded. The next optimization phase learns only from experience collected by the policy that now exists.</figcaption>
      </figure>
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
