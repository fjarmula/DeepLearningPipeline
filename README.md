# 1. Hyperparameter Optimization Report

### 1. Methodology
- **Training Subset:** 800 images
- **Validation Subset:** 100 images
- **Epochs per Trial:** 20
- **Hardware Acceleration:** Apple Metal Performance Shaders (MPS)
- **Search Space:**
  - **Learning Rates:** [0.0001, 0.001, 0.01, 0.1]
  - **Batch Sizes:** [8, 16, 32, 64, 128]

### 2. Results Summary (Validation Accuracy)

| Learning Rate | BS: 8 | BS: 16 | BS: 32 | BS: 64 | BS: 128 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.0001** | 92.00% | 93.00% | 93.00% | 93.00% | 90.00% |
| **0.001** | **97.00%** | 94.00% | 95.00% | 94.00% | 93.00% |
| **0.01** | 95.00% | 94.00% | 96.00% | 14.00%* | 96.00% |
| **0.1** | 15.00% | 15.00% | 15.00% | 14.00% | 15.00% |


### 3. Best Configuration
The optimization identified the following top-performing parameters:

* **Learning Rate:** `0.001`
* **Batch Size:** `8`
* **Max Validation Accuracy:** **97.00%**
* **Checkpoint:** `checkpoints/absolute_best_model.pth.tar`

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