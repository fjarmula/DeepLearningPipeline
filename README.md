# CLI Parameters

- `--config`  
  Sets the path to the configuration file (default: `config.yaml`).

- `--epochs`  
  Overrides the number of training epochs.

- `--logdir`  
  Sets the directory where logs will be saved.

- `--grid_search`  
  Enables grid search mode (runs all combinations of provided hyperparameters).
- `--transform`  
  Sets the data transformation pipeline to use.

---

## Hyperparameters

- `--lr`  
  Sets learning rate(s).  
  Accepts one or multiple values.

- `--batch_size`  
  Sets batch size(s).  
  Accepts one or multiple values.

- `--optimizer`  
  Sets optimizer(s).  
  Supported: `sgd`, `adam`.

- `--weight_decay`  
  Sets weight decay (L2 regularization).  
  Accepts one or multiple values.

- `--seed`  
  Sets random seed(s) for reproducibility.  
  Accepts one or multiple values.

---

## Model & Loss

- `--model`  
  Sets the model architecture to run  
  (e.g., `SimpleCNN`, `Baseline`, `Stabilized`, etc.).

- `--criterion`  
  Sets the loss function.  
  Supported:  
  - `cross_entropy` (PyTorch built-in)  
  - `manual` (custom implementation)

---

## Behavior Notes

- Without `--grid_search`:  
  - Only one value per parameter is used  
  - If multiple values are provided, only the **first** is taken  

- With `--grid_search`:  
  - All combinations of provided parameter values are evaluated  

# 1. Hyperparameter Analysis
### 1. Methodology
- **Training Subset:** 800 images
- **Validation Subset:** 100 images
- **Epochs per Trial:** 20
- **Hardware Acceleration:** Apple Metal Performance Shaders (MPS)
- **Search Space:**
  - **Learning Rates:** [0.0001, 0.001, 0.01, 0.1]
  - **Batch Sizes:** [8, 16, 32, 64, 128]
  - **Optimizer:** Adam
  - **Loss Function:** Cross-Entropy Loss
  - **Weight Decay:** 0.0

**directory**: *runs/mnist_experiment/grid_search_exp1*

### 2. Results Summary (Validation Accuracy)

| Learning Rate | BS: 8      | BS: 16 | BS: 32 | BS: 64 | BS: 128 |
|:--------------|:-----------|:-------|:-------|:-------|:--------|
| **0.0001**    | 92.00%     | 93.00% | 93.00% | 93.00% | 90.00%  |
| **0.001**     | **97.00%** | 94.00% | 95.00% | 94.00% | 93.00%  |
| **0.01**      | 95.00%     | 94.00% | 96.00% | 90.00% | 96.00%  |
| **0.1**       | 15.00%     | 15.00% | 15.00% | 14.00% | 15.00%  |


### 3. Best Configuration
The optimization identified the following top-performing parameters:

* **Learning Rate:** `0.001`
* **Batch Size:** `8`
* **Max Validation Accuracy:** **97.00%**

### 4. Key Observations from TensorBoard

#### The Best Zone (LR 0.001)
Configurations using $LR = 0.001$ showed the most stable convergence.
Loss decreased exponentially while accuracy scaled smoothly,
reaching 90%+ within the first 5 epochs. Similar behavior was observed for $LR = 0.0001$
but the loss curve was decaying slower.

#### Exploding Gradients (LR 0.1)
At the highest learning rate (0.1), the model immediately failed.
Initial training loss was recorded as high as **746.94**. 
Immediately loss dropped to around $2$ 
and because the optimizer overshot the local minima 
so drastically, the model stagnated at ~15% accuracy (near-random guessing).




#### Efficiency vs. Speed (LR 0.0001)
While $LR = 0.0001$ eventually reached high accuracies (93%), the "slope" of improvement was significantly flatter. This indicates that while the model is learning, it is being inefficient with the available compute compared to the $0.001$ trials.

### 5. Conclusion
For final deployment on the full MNIST dataset (60,000 images), **LR 0.001** is recommended as the primary learning rate. While **BS 8** provided the highest accuracy on this subset, a larger batch size (e.g., **BS 64**) may be preferred for full-scale training to utilize hardware parallelism more effectively, as it maintained a high 94% accuracy.

---

# 2. Optimizer & Weight Decay Analysis
**Dataset Subset:** 800 Train / 100 Validation

**Fixed Parameters:** Batch Size = 8, Epochs = 20

**directory**: *runs/mnist_experiment/grid_search_exp2*

---

### 1. Summary of Trials

| Trial | LR     | Optimizer | Weight Decay | Best Val Acc |
|:------|:-------|:----------|:-------------|:-------------|
| 1     | 0.01   | Adam      | 0.0          | 87.00%       |
| 2     | 0.01   | Adam      | 0.0001       | 91.00%       |
| 3     | 0.01   | SGD       | 0.0          | 93.00%       |
| 4     | 0.01   | SGD       | 0.0001       | **95.00%**   |
| 5     | 0.0001 | Adam      | 0.0          | 93.00%       |
| 6     | 0.0001 | Adam      | 0.0001       | 94.00%       |
| 7     | 0.0001 | SGD       | 0.0          | 84.00%       |
| 8     | 0.0001 | SGD       | 0.0001       | 84.00%       |

---

### 2. Best Configuration Found

The optimization process identified the following top-performing setup:

* **Optimizer:** `SGD` (with 0.9 Momentum)
* **Learning Rate:** `0.01`
* **Weight Decay:** `0.0001`
* **Validation Accuracy:** **95.00%**

##### Why this won:
While Adam is often the "default" choice, **SGD with Momentum** and a higher learning rate (0.01) found a better local minimum. The addition of **Weight Decay** acted as a crucial stabilizer, preventing the model from over-tuning to the noise in the small training subset.

### 3. Conclusion
For this CNN architecture on MNIST:
1.  **SGD (0.01)** is superior for peak accuracy but requires careful LR selection.
2.  **Adam (0.0001)** is the most "user-friendly" as it provides high accuracy (93-94%) across almost any configuration.
3.  **Weight Decay** is mandatory for small-batch training to prevent overfitting.

# 3. Architecture Performance Analysis
**Dataset Subset:** 6000 Train / 1000 Validation
**Fixed Parameters:** Learning Rate = 0.001,
Batch Size = 16,
Optimizer = SGD,
Weight Decay = 1e-05

**directory**: *runs/mnist_experiment/different_architectures*

### Executive Summary
The experiment successfully compared five different CNN architectures on the MNIST dataset using a consistent hyperparameter set:
* **Learning Rate:** 0.001
* **Batch Size:** 16
* **Optimizer:** SGD
* **Weight Decay:** 1e-05

| Model Name      | Type         | Activation | Batch Norm | Dropout | Kernel Size | MaxPool Layers |
|:----------------|:-------------|:-----------|:-----------|:--------|:------------|:---------------|
| **SimpleCNN**   | Standard     | ReLU       | No         | 0.0     | 3           | 1              |
| **Baseline**    | Experimental | ReLU       | No         | 0.0     | 3           | 2              |
| **Stabilized**  | Experimental | ReLU       | Yes        | 0.3     | 3           | 2              |
| **High-Vision** | Experimental | ReLU       | No         | 0.0     | 5           | 2              |
| **Modernist**   | Experimental | GELU       | No         | 0.0     | 3           | 2              |

The **Stabilized** model achieved the highest accuracy (**98.50%**), while the **SimpleCNN** proved to be the least efficient, requiring over 1.1 million parameters to achieve the lowest performance of the group.

---

### 1. Performance Comparison Table

| Architecture    | Accuracy (%) | Parameters | Time (s) | Convergence Epoch |
|:----------------|:-------------|:-----------|:---------|:------------------|
| **Stabilized**  | **98.50**    | 207,018    | 62.35    | 38                |
| **High-Vision** | 98.00        | 215,370    | 53.34    | 30                |
| **Baseline**    | 97.90        | 206,922    | 51.52    | 36                |
| **Modernist**   | 97.40        | 206,922    | 51.65    | 33                |
| **SimpleCNN**   | 97.30        | 1,199,882  | 57.30    | 27                |

---

### 2. Key Insights & Analysis

#### The "Stabilization" Advantage
The **Stabilized** model (incorporating BatchNorm and Dropout) reached the highest accuracy of **98.50%**. Although it had the longest training time (62.35s), the addition of Batch Normalization allowed the model to generalize better and achieve a higher performance ceiling than the "raw" architectures.

#### Efficiency: SimpleCNN vs. Experimental Models
The **SimpleCNN** is significantly over-parameterized for this task. It utilizes **1,199,882** parameters—nearly 6 times more than the **Baseline**—yet it produced the lowest accuracy. This confirms that modern CNN design (using Pooling and deeper, narrower layers) is far superior to legacy designs with massive Fully Connected layers.

#### Convergence and Stability
* **Earliest Convergence:** **SimpleCNN** (Epoch 27). Due to its high parameter count, it minimizes training loss rapidly but plateaus early on validation accuracy, suggesting a slight tendency toward overfitting.
* **Balanced Convergence:** **High-Vision** (Epoch 30). By utilizing a larger $5 \times 5$ kernel, this model captured more spatial information per layer, resulting in strong accuracy (98.00%) with a very competitive training time.

#### Modernist (GELU) vs. Baseline (ReLU)
In this specific experiment, the **Baseline (ReLU)** slightly outperformed the **Modernist (GELU)** (97.90% vs 97.40%). This indicates that for a relatively simple feature set like MNIST, the extra computational complexity of the GELU activation function does not necessarily translate to higher accuracy.

# 4. Impact of Random Seeds and Determinism
**directory**: *runs/mnist_experiment/different_seed*

### 1. Reproducibility Comparison
The experiment compared three runs with **unfixed seeds** (Stochastic) and three runs with a **fixed seed** (Deterministic).

| Run Type    | Seeds Used       | Accuracy Range  | Result Consistency |
|:------------|:-----------------|:----------------|:-------------------|
| **Unfixed** | None, None, None | 93.33% - 96.67% | **Variable**       |
| **Fixed**   | 42, 42, 42       | 96.67% - 96.67% | **Identical**      |

---

### 2. Key Conclusions

##### A. The Variance of "None" (Unfixed Seeds)
In Runs 1, 2, and 3, the model achieved different results (**96.67%, 93.33%, and 96.67%**) despite using identical hyperparameters.
* **Why?** Without a fixed seed, the weight initialization and data shuffling order change every time. 
* **Insight:** Run 2's lower performance (93.33%) shows that "bad luck" in initialization can lead a model to a suboptimal local minimum, even with the same architecture.

##### B. Success of Deep Determinism (Fixed Seeds)
In Runs 4, 5, and 6, the results were **mathematically identical**.
* **Loss & Accuracy:** If you look at Epoch 01 for all three runs, the Train Loss is exactly `2.2015`.
* **Insight:** By setting `torch.manual_seed(42)` and configuring deterministic backends, we have eliminated variance. This is critical for debugging and for scientific papers where results must be verified by others.

#### C. Convergence Behavior
* **Unfixed Seeds:** Convergence epochs varied slightly (up to Epoch 50).
* **Fixed Seeds:** All fixed-seed runs reached their peak at exactly the same time (Convergence Epoch 43). 
* **Stability:** The "Stabilized" architecture (BatchNorm + Dropout) shows resilience; even with the variance in unfixed seeds, it consistently stayed above 93%, proving the architecture is robust regardless of the starting point.

---

### 3. Technical Summary
* **Total Runs:** 6
* **Best Accuracy:** 96.67% 
* **Observation:** Fixing the seed does not necessarily give a *better* result than a random seed, but it gives a *predictable* one. Run 3 (unfixed) and Run 4 (fixed) both hit 96.67%, but only Run 4 can be perfectly recreated on another machine.

# 5. Data Augmentation Performance Analysis
**directory**: *runs/mnist_experiment/augmentation* (vs. *augmentation_none*)

### 1. Performance Comparison
| Configuration | Final Accuracy | Final Train Loss | Time (s) |
|:--------------|:---------------|:-----------------|:---------|
| **Standard**  | **93.33%**     | **0.0229**       | 3.99s    |
| **Augmented** | 86.67%         | 0.5607           | 4.94s    |

### 2. Key Conclusions

#### A. Data Augmentation is "Harder"
The augmented run showed a much higher training loss (0.56 vs 0.02). By rotating and shifting images, we prevent the model from memorizing pixel-specific locations, forcing it to learn more generalized features.

#### B. The Accuracy Gap
In a 50-epoch window, the **Standard** run achieved 93.33% while the **Augmented** run hit 86.67%. Standard training allows the model to "perfect" clean, centered digits quickly, while Augmented training requires more time (epochs) to generalize across the increased variety.

#### C. Generalization vs. Overfitting
The Standard run's very low loss (0.02) suggests potential overfitting to the specific orientation of the MNIST dataset. The Augmented model, while showing lower accuracy here, is likely much more robust to real-world variations (tilted or shifted handwriting).

---