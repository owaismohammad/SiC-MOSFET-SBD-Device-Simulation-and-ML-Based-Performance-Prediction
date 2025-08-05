# SiC MOSFET-SBD Device Simulation and ML-Based Performance Prediction

## 📌 Project Overview

This repository is my attempt to **reproduce the research paper**:

> **"SiC MOSFET with Integrated SBD Device Performance Prediction Method Based on Neural Network"**

The project demonstrates:
- **Device Simulation** of a SiC MOSFET integrated with a Schottky Barrier Diode (SBD) using Silvaco ATLAS.
- **Graphical Analysis** of device characteristics (Output, Input, Breakdown).
- **Generalized Simulation Automation** for 625 parameter variations (P-well doping, P-well depth, JFET width, SBD width).
- **Neural Network Model** to predict device performance parameters (Specific On-Resistance \(R_{on,sp}\), Threshold Voltage \(V_{th}\), Breakdown Voltage (BV)) from structural parameters.

⚠️ Due to **hardware limitations**, I could not execute all 625 simulations and train the ML model on the full dataset. However:
- The **entire pipeline is implemented**.
- The **neural network architecture is ready** and can be trained by anyone with sufficient computational resources.

---

## 📖 My Journey

I began this project **at the end of July**, with **zero prior knowledge of Silvaco ATLAS**.  
Here's how the implementation progressed:

1. **Learning Silvaco from Scratch:**  
   - With no proper tutorials or guides available online, I relied solely on the **1200-page Silvaco manual**.  
   - Progress was slow and debugging was extremely challenging due to limited online support.

2. **Single Device Simulation:**  
   - Implemented the given SiC MOSFET with integrated SBD structure as described in the paper.
   - Extracted **I-V characteristics** and threshold voltage \(V_{th}\).  
   - The paper reported \(V_{th} = 6.101V\), whereas my simulation resulted in **5.0889V** because:
     - Some device parameters were not specified in the paper.
     - Computational limitations forced me to use **less coarse meshing**, affecting accuracy.

3. **Graph Generation:**  
   - Reproduced **Output Characteristics**, **Input Characteristics**, and **Breakdown Characteristics** as shown in the paper.
   - Scripts were written to automatically plot results from Silvaco simulation outputs.

4. **Generalized Simulation:**  
   - Developed a **parameter sweep automation program** to simulate 625 devices with varying:
     - P-well doping  
     - P-well depth  
     - JFET width  
     - SBD width  
   - The code is fully functional but could not be fully executed due to resource constraints (one simulation takes ~40 min on my machine).

5. **Machine Learning Model:**  
   - Implemented the **neural network architecture** described in the paper to predict device performance metrics.
   - Ready to train once the full dataset is generated.

---

## 🚧 Challenges with Silvaco

Working with Silvaco ATLAS was one of the most challenging aspects of this project:

- **❌ Lack of Learning Resources:**  
  No proper tutorials or guides are available online for complex device simulations.  
  Most learning had to come from the **1200+ page manual**, making the process time-consuming.

- **⚠️ Debugging Issues:**  
  Errors are often cryptic and require multiple trial-and-error attempts to resolve.

- **⏳ Extremely Slow Simulations:**  
  Each device structure takes **30–40 minutes to simulate**, making large-scale parameter sweeps difficult on a personal machine.

---


---

## 📊 Results (Single Device)

- The following section compares the **results from the paper** with **my simulation results** for a single MOSFET.

### 1️⃣ Threshold Voltage \(V_{th}\)
- **Paper:** 6.101 V  
- **My Simulation:** 5.0889 V

⚠️ **Reason for Deviation:**  
- Missing some parameter definitions in the paper.  
- Computational limits forced me to use a **less fine mesh**, impacting accuracy.

---

### 2️⃣ Graphical Comparisons

### 2️⃣ Graphical Comparisons

### 2️⃣ Graphical Comparisons

#### 🔹 Device Structure
### 2️⃣ Graphical Comparisons

| **Characteristic**       | **Graph from Paper** | **My Implementation** |
|--------------------------|----------------------|-----------------------|
| **Device Structure**      | <img src="original_paper_results/sbdmosfet.png" width="350"/> | <img src="result_visualization/structure_sicsbdmosfet.png" width="350"/> |
| **Output Characteristic** | <img src="original_paper_results/output_characteristic.png" width="350"/> | <img src="result_visualization/output_characteristic.png" width="350"/> |
| **Input Characteristic**  | <img src="original_paper_results/input_characteristic.png" width="350"/> | <img src="result_visualization/transfer_characteristic.png" width="350"/> |
| **Breakdown Voltage**     | <img src="original_paper_results/breakdown_voltage.png" width="350"/> | <img src="result_visualization/breakdown_plot.png" width="350"/> |


## 🧠 Machine Learning Architecture – SiC-SBD MOSFET Performance Prediction

This module implements the **neural network architecture** inspired by the research paper:

> **"SiC MOSFET with Integrated SBD Device Performance Prediction Method Based on Neural Network"**

The model is designed to predict the following electrical characteristics of the SiC MOSFET-SBD device:
- Specific ON-Resistance (\( R_{on,sp} \))
- Threshold Voltage (\( V_{th} \))
- Breakdown Voltage (BV)
- (Optionally) Another device characteristic depending on dataset availability

---

## ⚙️ Architecture Overview

The implemented network goes **beyond a simple MLP**, integrating **fully connected layers, a transposed CNN, and a dual-branch convolutional block** to extract complex feature interactions between structural parameters and resulting device performance.

### **🔹 Input Features**
The model takes **4 structural parameters** as input:
1. P-well doping  
2. P-well depth  
3. JFET width  
4. SBD width  

---

### **🔹 Model Layers**

1️⃣ **Fully Connected (Feature Expansion)**  
- Expands the 4 input parameters into a high-dimensional representation.  
- Layers:
   - Linear(4 → 64) → BN → ReLU  
   - Linear(64 → 128) → BN → ReLU  
   - Linear(128 → 320) → BN → ReLU  
- Output reshaped to (64 channels × 5 timesteps) for CNN processing.

---

2️⃣ **Transposed Convolution**  
- Upsamples the dense features into a structured format for convolutional processing.  
- Layer:
   - ConvTranspose1d(64 → 32) → BN → ReLU

---

3️⃣ **Dual-Branch Convolution Block (Feature Extraction)**  
- Uses a **custom `DualConv` module** (imported from `dualbranchconvolution.py`) that:
  - Applies **parallel convolution branches** to capture multi-scale feature dependencies.
  - Merges feature maps for richer representations.
- Three consecutive `DualConv` blocks progressively extract patterns, resulting in 256 feature channels.

---

4️⃣ **Standard Convolution Layers (Refinement)**  
- Further processes the extracted features:
  - Conv1d(256 → 128) → BN → ReLU  
  - Conv1d(128 → 64) → BN → ReLU  
  - Conv1d(64 → 32) → BN → ReLU  

---

5️⃣ **Fully Connected Output Layer (Regression)**  
- Flattens the features and predicts four numerical outputs:
   - Linear(32×5 → 64) → BN → ReLU  
   - Linear(64 → 32) → BN → ReLU  
   - Linear(32 → 4) → Final predictions  

---

> The **model architecture** is implemented but not trained on the full dataset due to limited compute power.  
> Anyone with access to a high-performance machine and a Silvaco license can use this code to **generate the dataset and train the model**.

---

## 🚀 How This Repo Can Help You

- Provides **ready-to-use Silvaco scripts** for simulating a SiC MOSFET-SBD device.
- Shows how to **automate large-scale simulations** for parametric analysis.
- Offers a **template for building ML-based prediction models** for semiconductor devices.
- A good **starting point for beginners in Silvaco** to learn:
  - How to structure a device simulation project.
  - How to extract and visualize results.
  - How to integrate ML with TCAD simulations.

---

## 🛠️ Future Work

- Run all **625 simulations** on a high-performance machine.
- Train and validate the **neural network model** with the full dataset.
- Improve mesh refinement for higher accuracy results.
- Extend the pipeline to other **SiC-based power devices**.

---

## 📜 References

- **Original Paper:**  
*"SiC MOSFET with Integrated SBD Device Performance Prediction Method Based on Neural Network"*  
([Link to the original paper](https://www.mdpi.com/2072-666X/16/1/55))

---

> **Disclaimer:** This repository is for educational and research purposes only.  
> It assumes you have access to a **licensed version of Silvaco ATLAS/DeckBuild**.

---


