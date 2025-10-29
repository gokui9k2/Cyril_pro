# PETNN: Physics inspired Energy Transition Neural Network

This repository contains my implementation of the paper:

** Physics inspired Energy Transition Neural Network for Sequence Learning **

* **Paper Link:** [https://arxiv.org/abs/2505.03281](https://arxiv.org/abs/2505.03281)

The paper proposes a novel RNN architecture that can compete with **Transformer based** models.
This new algorithm is based on a **physics inspired** neural network, and this repository represents my personal implementation of that work.

## Core Concept: The Quantum Atom

The architecture is based on the physical behavior of a quantum atom, which can exist in two main states:

1.  **Ground State:** A stable, **low-energy** state.
2.  **Excited State:** An unstable, **high energy** state that an atom enters after receiving energy. It eventually releases this energy to return to its stable ground state.

This new architecture models this behavior using four key variables:

* **$C_t$ (Cell State):** Represents the "energy cell" where the model accumulates knowledge.
* **$T_t$ (Remaining Time):** Represents the time remaining before the next energy release.
* **$I_t$ (Ground State Level):** The fundamental (baseline) energy of the neuron, analogous to the atom's ground state.
* **$R_t$ (Time Decay Rate):** The decay variable that controls the "remaining time" $T_t$.

## Key Equations

For me, the most important part of this new system is the "energy release" mechanism, governed by the following equations:

**1. Time Update:** The "remaining time" $T_t$ is decreased at each step.
$$T_{t}=R_{t}\cdot\sigma(T_{t-1}+Z_{t})-1$$

**2. Release Switch:** A "hard" binary switch $m$ determines if the energy is released.
$$m = \begin{cases} 1 & \text{if } T_t \le 0 \text{ (Time to release)} \\ 0 & \text{if } T_t > 0 \text{ (Accumulate energy)} \end{cases}$$

**3. Cell State Update:** If $m=1$, the cell's energy $C_t$ is reset to its ground state $I_t$. Otherwise, it continues to accumulate new energy $Z_c$.
$$C_{t}=(1-m)\cdot C_{t-1}+m\cdot I_{t}+Z_{c}$$

## My Implementation: Notes & Experiments

During my implementation, I focused on a few key areas and experiments:

### 1. The Importance of Bias Initialization

I paid particular attention to the bias terms in these equations. I discovered a critical sensitivity:

* **Problem:** If the biases are initialized too close to zero, the model can get "stuck" in a state of constantly releasing energy (i.e., $m$ is always 1) or **constant accumulation** ($m$ is always 0).
* **Effect:** This prevents the model from learning properly, making it act more like a classic LSTM or just an RNN that **accumulates** all the information, and it never gets out of this state.
* **Solution:** I found a specific combination of bias initializations that allows the model to start in a stable state (not in "release mode" nor "accumulation mode") but is not so high that it *never* enters the energy release state.

### 2. Experiment: Custom Physics-Based Loss Function

I also experimented with a custom loss function based on the physics assumption.

* **Hypothesis:** Penalizing high levels of $C_t$ (the cell energy) might encourage the model to use the release mechanism more efficiently.
* **Result:** This custom loss *did* work in one way the model used the releasing method more often. However, this did not translate into better performance on the final task.

While it didn't improve my metrics, I still believe this could be a promising direction for future research to help the model better utilize its core process.

### 3. Experiment: Hard vs Smooth Switching

The paper uses a "hard" binary switch ($m=0$ or $m=1$), which is inspired by the discrete nature of a quantum jump.

* **Hypothesis:** Would a 'smooth' switch perform better? The smooth switch allows all the parameters of the model to receive gradients through the training.
* **Result:** I replaced the hard switch with a smooth function, and the performance was lower.
  
This strongly suggests that the **physics-inspired**, discrete "release" mechanism is a critical and important part of this architecture's success.

