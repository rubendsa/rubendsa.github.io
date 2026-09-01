---
title: "Explaining RLinf—and how it plugs into Isaac Lab"
subtitle: "Isaac Lab supplies the world; RLinf supplies the distributed post-training runtime."
description: "A practical mental model for RLinf, its three-worker training loop, and the extension and YAML mapping that connect it to Isaac Lab."
excerpt: "A practical mental model for RLinf, its three-worker training loop, and the extension and YAML mapping that connect it to Isaac Lab."
date: 2026-09-01 00:00:00 +0000
last_modified_at: 2026-09-01 00:00:00 +0000
permalink: /blog/explaining-rlinf-and-isaac-lab/
categories:
  - robot-learning
tags:
  - RLinf
  - Isaac Lab
  - VLA
  - reinforcement learning
classes:
  - notebook-theme
author_profile: false
sidebar: false
toc: false
header:
  og_image: /images/rlinf-isaac-lab-social.png
---

The shortest useful explanation of the integration is this: **Isaac Lab owns the robotics problem; RLinf owns the learning system around it.** Isaac Lab simulates the robot and its sensors, defines observations and actions, computes rewards, and decides when an episode ends. RLinf coordinates model inference, experience collection, distributed policy updates, weight synchronization, and checkpoints.

That boundary is easy to miss because both projects use the language of reinforcement learning. They solve different layers of the same loop.

<div class="notebook-callout">
  <p class="notebook-callout__label">The mental model</p>
  <p><strong>Isaac Lab supplies the world.</strong> <strong>RLinf supplies the distributed post-training runtime.</strong> The integration is the adapter that keeps observations and actions meaningful as they cross between them.</p>
</div>

## Why a VLA needs more machinery than a classic policy

A conventional Isaac Lab run can keep simulation, a compact policy, and optimization in a relatively tight process. A vision-language-action model changes the shape of the workload. Camera rendering and physics want one resource profile; large-model inference wants another; backpropagation through a VLA may require sharding across several GPUs.

RLinf is designed for that heterogeneous case. Its system model separates the readable RL workflow from the way that workflow executes. A Runner controls the order of operations; WorkerGroups own simulation, generation, and learning; Channels move data directly between workers; and `component_placement` maps those logical groups to hardware. RLinf's broader macro-to-micro machinery can turn the same graph into collocated, disaggregated, or hybrid execution.

The architecture is easier to understand when control flow, training data, and placement are shown separately:

<figure class="rlinf-architecture" aria-labelledby="rlinf-architecture-title" aria-describedby="rlinf-architecture-caption">
  <header class="rlinf-architecture__header">
    <div>
      <p>RLinf × Isaac Lab · reference architecture</p>
      <h3 id="rlinf-architecture-title">Control and data move on separate paths</h3>
    </div>
    <div class="rlinf-architecture__legend" aria-label="Diagram legend">
      <span><i class="rlinf-architecture__swatch rlinf-architecture__swatch--worker" aria-hidden="true"></i>RLinf worker</span>
      <span><i class="rlinf-architecture__swatch rlinf-architecture__swatch--isaac" aria-hidden="true"></i>Isaac Lab</span>
      <span><i class="rlinf-architecture__swatch rlinf-architecture__swatch--seam" aria-hidden="true"></i>Integration seam</span>
    </div>
  </header>

  <div class="rlinf-architecture__runtime">
    <section class="rlinf-architecture__runner" aria-labelledby="rlinf-runner-title">
      <div class="rlinf-architecture__runner-copy">
        <span>Control plane</span>
        <h4 id="rlinf-runner-title">EmbodiedRunner</h4>
        <p>Invokes WorkerGroup methods, waits at synchronization barriers, and triggers actor checkpoints. Training tensors do not pass through it.</p>
      </div>
      <ol class="rlinf-architecture__cycle" aria-label="One synchronous training iteration">
        <li><span>01</span>Sync weights when due</li>
        <li><span>02</span>Generate rollouts</li>
        <li><span>03</span>Compute GAE</li>
        <li><span>04</span>Train actor</li>
      </ol>
    </section>

    <div class="rlinf-architecture__workers">
      <div class="rlinf-architecture__seam rlinf-architecture__seam--spanning">
        <span>Configuration plane · translation seam · not a worker</span>
        <strong>Extension + YAML contract</strong>
        <small><b>Env side:</b> task registration and camera/state selection <i>·</i> <b>Rollout side:</b> GR00T key mapping, model configuration, and action padding</small>
      </div>

      <section class="rlinf-architecture__worker rlinf-architecture__worker--env" aria-labelledby="rlinf-env-title">
        <header>
          <span>01 · World</span>
          <h4 id="rlinf-env-title">EnvWorker</h4>
          <small>EnvGroup</small>
        </header>
        <div class="rlinf-architecture__isaac">
          <span>Isaac Lab task</span>
          <strong>ManagerBasedRLEnv</strong>
          <small>physics · sensors · observations · rewards · termination</small>
        </div>
        <p><code>IsaacLabGenericEnv</code> wraps simulator output into RLinf's canonical observation fields and applies returned actions.</p>
      </section>

      <div class="rlinf-architecture__exchange" aria-hidden="true">
        <div>
          <span>observations</span>
          <small>Env Channel</small>
          <b>→</b>
        </div>
        <div class="rlinf-architecture__exchange-return">
          <b>←</b>
          <span>action chunks</span>
          <small>Rollout Channel</small>
        </div>
      </div>

      <section class="rlinf-architecture__worker rlinf-architecture__worker--rollout" aria-labelledby="rlinf-rollout-title">
        <header>
          <span>02 · Act + collect</span>
          <h4 id="rlinf-rollout-title">MultiStepRolloutWorker</h4>
          <small>RolloutGroup</small>
        </header>
        <div class="rlinf-architecture__model">
          <span>Reference policy</span>
          <strong>GR00T · Hugging Face</strong>
          <small>observation converter · VLA inference · action converter</small>
        </div>
        <p>Generates action chunks and assembles rollout-horizon trajectory batches with rewards, dones, old log-probs, values, and model inputs.</p>
      </section>

      <div class="rlinf-architecture__training-exchange" aria-hidden="true">
        <div>
          <span>trajectory batches</span>
          <small>Actor Channel</small>
          <b>↓</b>
        </div>
        <div>
          <b>↑</b>
          <span>full state dict</span>
          <small>direct sync</small>
        </div>
      </div>

      <section class="rlinf-architecture__worker rlinf-architecture__worker--actor" aria-labelledby="rlinf-actor-title">
        <header>
          <span>03 · Learn</span>
          <h4 id="rlinf-actor-title">EmbodiedFSDPActor</h4>
          <small>ActorGroup</small>
        </header>
        <div class="rlinf-architecture__actor-steps">
          <span>advantages + returns</span>
          <span>actor–critic objective</span>
          <span>FSDP update</span>
        </div>
        <p>Owns the trainable policy state; the Runner periodically saves it and synchronizes fresh weights back to rollout.</p>
      </section>
    </div>

    <ol class="rlinf-architecture__flow-key" aria-label="The three data flows in the RLinf Isaac Lab training loop">
      <li><span>Act</span>EnvWorker sends mapped observations to RolloutWorker; action chunks return.</li>
      <li><span>Learn</span>RolloutWorker sends trajectory batches to the actor for advantage computation and training.</li>
      <li><span>Refresh</span>At a configured boundary, ActorWorker sends its full state dict; RolloutWorker loads it and records a model ID.</li>
    </ol>
  </div>

  <figcaption id="rlinf-architecture-caption">The Runner controls the workflow, but Channels carry the data directly between workers. The extension and YAML configure the simulator/model boundary; they are not a fourth runtime worker.</figcaption>
</figure>

This diagram follows the `rlinf==0.2.0dev2` dependency currently documented by Isaac Lab. RLinf's `main` branch has since moved some trajectory assembly into `EnvWorker`, but the durable architectural idea is unchanged: the simulator, inference engine, and trainer are independent worker groups connected by explicit data paths.

That decomposition is the reason the integration matters. The task stays an Isaac Lab task, while the expensive VLA post-training loop uses RLinf's control, communication, and placement machinery.

<figure class="rlinf-diagram rlinf-placement-map" aria-labelledby="rlinf-placement-map-title" aria-describedby="rlinf-placement-map-caption">
  <header class="rlinf-diagram__header">
    <div>
      <p>Execution view · same logical graph</p>
      <h3 id="rlinf-placement-map-title">Three ways to place the workers on hardware</h3>
    </div>
    <span class="rlinf-diagram__stamp">Placement changes · channels remain</span>
  </header>

  <div class="rlinf-placement-map__modes">
    <section class="rlinf-placement-map__mode is-reference" aria-labelledby="rlinf-collocated-title">
      <header>
        <span>01 · Reference</span>
        <h4 id="rlinf-collocated-title">Collocated</h4>
        <small>time-share one pool</small>
      </header>
      <div class="rlinf-placement-map__shared-pool">
        <span>Same GPU set</span>
        <div>
          <b>Env</b>
          <b>Rollout</b>
          <b>Actor</b>
        </div>
      </div>
      <div class="rlinf-placement-map__time" aria-label="Collection and training run as separate stages">
        <span>collect</span><i aria-hidden="true">→</i><span>train</span>
      </div>
      <code>actor,env,rollout: all</code>
      <p>The Isaac Lab reference maps every WorkerGroup to all available GPUs. Stages run in turn on the shared devices.</p>
    </section>

    <section class="rlinf-placement-map__mode" aria-labelledby="rlinf-disaggregated-title">
      <header>
        <span>02 · Separate</span>
        <h4 id="rlinf-disaggregated-title">Disaggregated</h4>
        <small>dedicate each pool</small>
      </header>
      <div class="rlinf-placement-map__separate-pools">
        <div><span>GPU group A</span><b>Env</b></div>
        <div><span>GPU group B</span><b>Rollout</b></div>
        <div><span>GPU group C</span><b>Actor</b></div>
      </div>
      <div class="rlinf-placement-map__pipeline" aria-label="Channels can pipeline work across dedicated worker pools">
        <span>Env ↔ Rollout → Actor</span>
      </div>
      <p>Separate pools make overlap possible and remove GPU swapping, but introduce pipeline balance and transfer costs.</p>
    </section>

    <section class="rlinf-placement-map__mode" aria-labelledby="rlinf-hybrid-title">
      <header>
        <span>03 · Mixed</span>
        <h4 id="rlinf-hybrid-title">Hybrid</h4>
        <small>split, then reunite</small>
      </header>
      <div class="rlinf-placement-map__hybrid-pools">
        <div><span>GPU 0–3</span><b>Env</b></div>
        <div><span>GPU 4–7</span><b>Rollout</b></div>
        <div class="rlinf-placement-map__actor-pool"><span>GPU 0–7 · after collection</span><b>Actor</b></div>
      </div>
      <div class="rlinf-placement-map__pipeline" aria-label="Environment and rollout pipeline before the actor uses the combined pool">
        <span>Env ↔ rollout overlap</span><i aria-hidden="true">→</i><span>train</span>
      </div>
      <p>The official embodied example pipelines simulation and generation on separate subsets, then lets training use their union.</p>
    </section>
  </div>

  <figcaption id="rlinf-placement-map-caption">Placement is a hardware decision, not a rewrite of the algorithm. The reference trocar configuration is the collocated case; the other two cards show execution modes exposed by RLinf.</figcaption>
</figure>

## The integration seam

The Isaac Lab-specific extension lives in `isaaclab_contrib`, Isaac Lab's incubator for community-maintained features. The wider path also uses Isaac Lab's unified backend dispatcher and a task-specific YAML configuration. From the user's side, RLinf appears as another backend behind the unified `train` and `play` entry points. Underneath, the launcher tells RLinf to load the extension module and points it at the active Hydra configuration.

The extension performs three jobs at startup:

1. It registers the train and evaluation task IDs from the YAML file in RLinf's Isaac Lab environment registry. The task can remain in Isaac Lab; no fork of RLinf is required.
2. It registers observation and action converters for the VLA—in the reference path, GR00T.
3. When a robot uses a custom embodiment, it wires in the embodiment tag and model data configuration needed to interpret that robot's state and action layout.

The YAML file is therefore more than a bag of hyperparameters. It is the **semantic contract** between the simulator and the model.

<figure class="rlinf-diagram rlinf-contract" aria-labelledby="rlinf-contract-title" aria-describedby="rlinf-contract-caption">
  <header class="rlinf-diagram__header">
    <div>
      <p>Integration view · reference trocar task</p>
      <h3 id="rlinf-contract-title">One boundary, three different data paths</h3>
    </div>
    <span class="rlinf-diagram__stamp">Meaning matters more than shape</span>
  </header>

  <div class="rlinf-contract__lanes">
    <section class="rlinf-contract__lane" aria-labelledby="rlinf-contract-inputs">
      <header>
        <span>01</span>
        <div><h4 id="rlinf-contract-inputs">Policy inputs</h4><p>Isaac Lab → GR00T</p></div>
      </header>
      <div class="rlinf-contract__cell rlinf-contract__cell--isaac">
        <span>Isaac Lab fields</span>
        <ul>
          <li><code>front_camera</code></li>
          <li><code>left/right_wrist_camera</code></li>
          <li><code>robot_joint_state[15:29]</code></li>
          <li><code>robot_dex3_joint_state</code></li>
          <li>task description</li>
        </ul>
      </div>
      <i class="rlinf-contract__arrow" aria-hidden="true"></i>
      <div class="rlinf-contract__cell rlinf-contract__cell--seam">
        <span>YAML configures two owners</span>
        <ul>
          <li><strong>EnvWorker wrapper:</strong> select and stack views; concatenate the 28D state</li>
          <li><strong>RolloutWorker converter:</strong> add <code>T=1</code>; map video, state, and language keys</li>
        </ul>
      </div>
      <i class="rlinf-contract__arrow" aria-hidden="true"></i>
      <div class="rlinf-contract__cell rlinf-contract__cell--rlinf">
        <span>GR00T model inputs</span>
        <ul>
          <li><code>video.room_view</code></li>
          <li><code>video.*_wrist_view</code></li>
          <li>four 7D arm/hand states</li>
          <li>language annotation<sup>†</sup></li>
        </ul>
      </div>
    </section>

    <section class="rlinf-contract__lane" aria-labelledby="rlinf-contract-actions">
      <header>
        <span>02</span>
        <div><h4 id="rlinf-contract-actions">Robot actions</h4><p>GR00T → Isaac Lab</p></div>
      </header>
      <div class="rlinf-contract__cell rlinf-contract__cell--rlinf">
        <span>GR00T action dictionary</span>
        <ul>
          <li><code>action.left_arm</code> · 7D</li>
          <li><code>action.right_arm</code> · 7D</li>
          <li><code>action.left_hand</code> · 7D</li>
          <li><code>action.right_hand</code> · 7D</li>
        </ul>
      </div>
      <i class="rlinf-contract__arrow" aria-hidden="true"></i>
      <div class="rlinf-contract__cell rlinf-contract__cell--seam">
        <span>Rollout-side action path</span>
        <ul>
          <li><strong>Model wrapper:</strong> reverse modality transforms</li>
          <li><strong>Registered converter:</strong> select <code>K=1</code> and concatenate four 7D groups into <code>[B,1,28]</code></li>
          <li><strong>Registered converter:</strong> prefix 15 zeros to emit <code>[B,1,43]</code></li>
        </ul>
      </div>
      <i class="rlinf-contract__arrow" aria-hidden="true"></i>
      <div class="rlinf-contract__cell rlinf-contract__cell--isaac">
        <span>Isaac Lab action</span>
        <strong>43D ordered joint targets</strong>
        <small>15 zero-padded body targets + 28 VLA-controlled arm and hand targets. EnvWorker converts numpy to torch, then applies the chunk.</small>
      </div>
    </section>

    <section class="rlinf-contract__lane" aria-labelledby="rlinf-contract-learning">
      <header>
        <span>03</span>
        <div><h4 id="rlinf-contract-learning">Learning signals</h4><p>Isaac Lab → ActorGroup</p></div>
      </header>
      <div class="rlinf-contract__cell rlinf-contract__cell--isaac">
        <span>Transition result</span>
        <ul>
          <li>reward</li>
          <li>terminated / truncated</li>
        </ul>
      </div>
      <i class="rlinf-contract__arrow" aria-hidden="true"></i>
      <div class="rlinf-contract__cell rlinf-contract__cell--collector">
        <span>RolloutWorker packs</span>
        <ul>
          <li>actions and model inputs</li>
          <li>old log-probs and values</li>
          <li>rewards and episode signals</li>
        </ul>
      </div>
      <i class="rlinf-contract__arrow" aria-hidden="true"></i>
      <div class="rlinf-contract__cell rlinf-contract__cell--rlinf">
        <span>Actor Channel</span>
        <strong>Trajectory shards</strong>
        <small>rollout-horizon batches, not necessarily completed episodes</small>
      </div>
    </section>
  </div>

  <div class="rlinf-contract__invariant"><strong>Contract invariant</strong><span>A valid tensor shape is not proof of valid semantics: verify camera order, joint order, slice bounds, time axes, and action coordinates.</span></div>
  <div class="rlinf-contract__source-note"><strong>† Source-version check</strong><span>The extension currently emits <code>annotation.human.action.task_description</code>, while the reference data config declares <code>annotation.human.task_description</code>. Verify the language key in the exact pinned sources you install.</span></div>
  <figcaption id="rlinf-contract-caption">The extension installs the task and converter types; it does not relay tensors at runtime. Environment wrapping happens on the EnvWorker side, while the registered GR00T converters and trajectory packing run in MultiStepRolloutWorker for the pinned reference.</figcaption>
</figure>

The reference trocar assembly task makes the contract concrete. It maps a front camera and two wrist cameras into GR00T video keys, selects the relevant proprioceptive state slices, carries a language task description, and pads the model's output so it lines up with the full Isaac Lab joint-action vector. A new task may use different values, but it must answer the same questions: *which pixels, which state dimensions, which language instruction, and which action coordinates does the model mean?*

## One step through the loop

One training iteration contains a fast simulator/inference loop inside a slower optimization loop:

<figure class="rlinf-diagram rlinf-iteration" aria-labelledby="rlinf-iteration-title" aria-describedby="rlinf-iteration-caption">
  <header class="rlinf-diagram__header">
    <div>
      <p>Sequence view · <code>rlinf==0.2.0dev2</code></p>
      <h3 id="rlinf-iteration-title">One iteration advances through explicit barriers</h3>
    </div>
    <span class="rlinf-diagram__stamp">Runner controls · Channels carry data</span>
  </header>

  <div class="rlinf-iteration__runner">
    <span>Control plane</span>
    <strong>EmbodiedRunner</strong>
    <p>Starts remote WorkerGroup methods, keeps their handles, and waits before advancing the algorithm. It never becomes the tensor relay.</p>
  </div>

  <ol class="rlinf-iteration__phases">
    <li class="rlinf-iteration__phase">
      <header><span>01</span><div><small>when the interval fires</small><h4>Refresh rollout</h4></div></header>
      <div class="rlinf-iteration__route">
        <b>ActorGroup · θ<sub>k</sub></b>
        <i aria-hidden="true">→</i>
        <b>RolloutGroup · θ<sub>k</sub></b>
      </div>
      <p>The actor sends a full state dict directly. Rollout loads it and derives its local model ID.</p>
      <span class="rlinf-iteration__barrier">Weight-sync barrier</span>
    </li>

    <li class="rlinf-iteration__phase rlinf-iteration__phase--collect">
      <header><span>02</span><div><small>repeat to rollout horizon</small><h4>Interact + collect</h4></div></header>
      <ol class="rlinf-iteration__inner" aria-label="Inner environment and inference loop">
        <li><span>EnvGroup</span>wrap observation + signals</li>
        <li><span>Env Channel</span>send to RolloutGroup</li>
        <li><span>RolloutGroup</span>convert · infer · record</li>
        <li><span>Rollout Channel</span>return action chunk</li>
        <li><span>Isaac Lab</span>step physics and sensors</li>
      </ol>
      <p>RolloutWorker packs the horizon and sends trajectory shards through the Actor Channel.</p>
      <span class="rlinf-iteration__barrier">Rollout / data-ready barrier</span>
    </li>

    <li class="rlinf-iteration__phase">
      <header><span>03</span><div><small>ActorGroup</small><h4>Estimate</h4></div></header>
      <div class="rlinf-iteration__actor-work">
        <span>trajectory batch</span><i aria-hidden="true">→</i><strong>GAE + returns</strong>
      </div>
      <p>The pinned trocar recipe uses generalized advantage estimation before any optimizer step begins.</p>
      <span class="rlinf-iteration__barrier">Advantage barrier</span>
    </li>

    <li class="rlinf-iteration__phase">
      <header><span>04</span><div><small>ActorGroup</small><h4>Optimize</h4></div></header>
      <div class="rlinf-iteration__actor-work">
        <span>actor–critic loss</span><i aria-hidden="true">→</i><strong>FSDP · θ<sub>k+1</sub></strong>
      </div>
      <p>The Runner waits for training, then triggers validation or a checkpoint only when its configured interval is due.</p>
      <span class="rlinf-iteration__barrier">Optimization barrier</span>
    </li>
  </ol>

  <div class="rlinf-iteration__return"><span aria-hidden="true">↶</span><strong>Next iteration</strong><p>θ<sub>k+1</sub> reaches rollout at the next weight-sync boundary; validation causes an immediate refresh first.</p></div>
  <figcaption id="rlinf-iteration-caption">During collection, EnvGroup and RolloutGroup run concurrently and exchange many observations and action chunks. Actor advantage computation starts only after trajectory receipt and rollout generation are complete.</figcaption>
</figure>

The critical engineering work is inside the collect phase. A tensor can have a valid shape and still carry the wrong meaning. Camera order, joint order, slice boundaries, time dimensions, and action chunking all need to agree.

## Running the reference path

On the current Isaac Lab `develop` documentation, the unified entry point looks like this after the RLinf extra and the documented pinned dependencies are installed:

```bash
uv run --extra rlinf isaaclab train --rl_library rlinf \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /path/to/base_model
```

Evaluation uses the same task contract. The base model describes the architecture; `--checkpoint` selects the RL-finetuned weights:

```bash
uv run --extra rlinf,video isaaclab play --rl_library rlinf \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /path/to/base_model \
  --checkpoint latest \
  --video
```

It is worth pinning commands to one Isaac Lab version when reproducing this workflow. Older beta documentation used a different checkpoint flag, while current `develop` uses `--checkpoint`. The integration is moving quickly enough that mixing snippets across versions is an avoidable source of confusion.

## Adapting it to another task

The useful order of operations is:

1. **Make the Isaac Lab task correct first.** Verify observations, actions, reward, reset behavior, and camera output without RLinf in the loop.
2. **Start from a pretrained VLA.** The documented workflow is demonstration collection, supervised base-model training, then RL post-training—not training a VLA from random initialization.
3. **Write the YAML contract.** Set the train and evaluation task IDs, camera keys, state slices, task description, GR00T mapping, action layout, and model configuration.
4. **Add an embodiment data config only when needed.** A robot whose modalities or action expert differ from the reference needs a matching model-side description.
5. **Validate the boundary before scaling.** Inspect one environment and one rollout. Confirm values and ordering, not only tensor shapes. Then increase parallel environments and worker placement.

That last point is deliberately unglamorous. Distributed training multiplies whatever is already true. It does not repair a camera mapped to the wrong key or an action shifted by one joint.

## What is still sharp-edged

As of September 2026, the Isaac Lab integration is explicitly experimental and Linux-only. The reference route is concrete—GR00T on the trocar assembly task—even though the surrounding interfaces are intended to support more models and algorithms. Installation currently pins an RLinf development build, compatible Transformers and tokenizers versions, and an exact Isaac-GR00T commit; it also documents dependency-conflict, FlashAttention, and aarch64 workarounds. Multi-GPU is recommended, and each FSDP checkpoint can occupy several gigabytes.

There is also an important scope distinction. RLinf's published throughput results demonstrate the flexibility of the system across embodied and reasoning workloads, but they are not Isaac Lab benchmark results. Likewise, the separate IsaacLab recipe in the RLinf repository is a different route with its own versions and commands. Treat the Isaac-Lab-first integration described here as one coherent stack rather than mixing the two recipes.

For a small MLP policy, one of Isaac Lab's established RL backends will usually be the simpler tool. RLinf becomes compelling when the policy is a foundation model and the simulator, inference engine, and trainer need to be treated as separate systems.

## Takeaway

The integration is not a rewrite of Isaac Lab inside RLinf. It is a narrow bridge:

- Isaac Lab continues to define what the robot sees, does, and earns.
- A YAML contract and extension translate that task into VLA semantics.
- RLinf turns the resulting interaction loop into distributed rollout and post-training workers.

That separation is the design win. Robotics researchers can keep task logic close to the simulator while changing how a large policy is placed, trained, and scaled around it.

## Sources and further reading

- [Isaac Lab: RL post-training for VLA models](https://isaac-sim.github.io/IsaacLab/develop/source/experimental-features/rlinf_vla_posttraining.html)
- [Isaac Lab: experimental feature status](https://isaac-sim.github.io/IsaacLab/develop/source/experimental-features/bleeding-edge.html)
- [Isaac Lab: unified reinforcement learning workflows](https://isaac-sim.github.io/IsaacLab/develop/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Isaac Lab's RLinf extension](https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab_contrib/isaaclab_contrib/rl/rlinf/extension.py)
- [Isaac Lab's reference trocar training configuration](https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config/isaaclab_ppo_gr00t_assemble_trocar.yaml)
- [RLinf repository and examples](https://github.com/RLinf/RLinf)
- [RLinf execution flow: Runner, WorkerGroups, and Channels](https://rlinf.readthedocs.io/en/latest/rst_source/concepts/execution_flow.html)
- [RLinf high-level programming and training flow](https://rlinf.readthedocs.io/en/latest/rst_source/concepts/flow.html)
- [RLinf execution and placement modes](https://rlinf.readthedocs.io/en/latest/rst_source/concepts/execution_modes.html)
- [RLinf `0.2.0dev2` embodied training sequence](https://github.com/RLinf/RLinf/blob/58066384e6cca81a69873f05d3cf3d5189e78a79/rlinf/runners/embodied_runner.py)
- [RLinf `0.2.0dev2` rollout worker and trajectory assembly](https://github.com/RLinf/RLinf/blob/58066384e6cca81a69873f05d3cf3d5189e78a79/rlinf/workers/rollout/hf/huggingface_worker.py)
- [RLinf systems paper](https://arxiv.org/abs/2509.15965)
