# SiC-MOSFET-SBD-Research-Reproduction
# Project: Automated Simulation & ML Modeling of MOSFET Devices with Silvaco

---

## Table of Contents

- [Project Motivation](#project-motivation)
- [Journey & Implementation Overview](#journey--implementation-overview)
- [Features & Capabilities](#features--capabilities)
- [Silvaco: Drawbacks & Challenges](#silvaco-drawbacks--challenges)
- [How to Use This Repository](#how-to-use-this-repository)
- [Results & Comparison](#results--comparison)
  - [Paper Graphs vs My Results](#paper-graphs-vs-my-results)
  - [Obtained MOSFET Vth Value](#obtained-mosfet-vth-value)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Potential & Next Steps](#potential--next-steps)
- [Gallery: Insert Your Graphs and Paper Graphs](#gallery-insert-your-graphs-and-paper-graphs)
- [Acknowledgements](#acknowledgements)

---

## Project Motivation

I started this project with zero prior knowledge in Silvaco at the end of July, determined to reproduce and extend the results of a referenced research paper. The goal was not just to validate a single device, but to **generalize the simulation process** for a broader set of devices, then apply ML on the complete dataset.

---

## Journey & Implementation Overview

- **July:** Began exploring Silvaco, starting with basic device simulations.
- **August:** Successfully implemented and validated a single MOSFET as described in the paper.
- **September:** Progressed to generating all simulation graphs (breakdown, output, input characteristics).
- **October:** Faced and solved the challenge of scaling the code.
- **November:** Developed the general framework to automate and run all 625 device simulations—fully parameterized code.
- **December:** Designed and implemented the entire ML architecture (per the paper’s approach).

**Note:** I could not extract all 625 device results or train the ML model because of computational resource limitations. However, both the simulation codebase and the ML architecture are complete; anyone with access to adequate computational power can directly use this workflow.

---

## Features & Capabilities

- **Automated Silvaco Simulation:** Reproducible setup for any MOSFET structure as described in the reference paper.
- **Device Characteristics:** Output graphs include:
  - Output characteristics
  - Input characteristics
  - Breakdown voltage
- **Parameter Sweep:** Generalized program structure, scalable to 625 simulation variations.
- **Machine Learning:** ML pipeline and model architecture ready for training and prediction.

---

## Silvaco: Drawbacks & Challenges

- **Lack of Documentation:** Minimal online resources, few tutorials, and a steep learning curve.
- **Support Materials:** Main reference is a 1,200+ page manual; if you're stuck, it's a slow process to debug or resolve errors.
- **Community & Forums:** Community support is sparse—solving issues often means exhaustive manual reading.
- **Resource Intensity:** Simulating hundreds of devices is computationally expensive; mesh coarseness and parameter granularity must be traded off for speed.
- **Parameter Visibility:** Many necessary device parameters are undefined in papers, requiring experimentation.

---

## How to Use This Repository

1. **Clone the Repo:**  
   `git clone <your-repo-url> && cd <repo-folder>`
2. **Install Silvaco:**  
   Obtain and install Silvaco TCAD software (not included).
3. **Configure Parameters:**  
   Adjust device structure, simulation ranges, and mesh settings as needed in the `.in` files and scripts.
4. **Run Simulations:**  
   Use the main script to:
   - Simulate a single device
   - Run batch simulations for parameter sweeps
5. **Machine Learning:**  
   Load the results (once generated) into the provided ML scripts to train and evaluate the predictive model.

---

## Results & Comparison

### Paper Graphs vs My Results

> _Add images below! Side-by-side comparison will be helpful; use the Markdown image syntax (see examples)._

| Characteristic | Graph from Paper | My Implementation |
|--|--|--|
| Output Characteristic | ![Paper Output](./images/paper_output.png) | ![My Output](./images/my_output.png) |
| Input Characteristic  | ![Paper Input](./images/paper_input.png)   | ![My Input](./images/my_input.png)   |
| Breakdown Voltage     | ![Paper Breakdown](./images/paper_breakdown.png) | ![My Breakdown](./images/my_breakdown.png) |

_Feel free to replace filenames above with your actual graphs._

### Obtained MOSFET Vth Value

- **Paper:** \( V_{th} = 6.101 \)
- **Mine:** \( V_{th} = 5.0889 \)  
  _This difference is due to undefined parameters in the paper, and because I had to use coarser meshes and less fine spacing due to computational limitations._

---

## Machine Learning Pipeline

Although I could not train the ML model on all 625 results due to resource constraints, the ML architecture is ready, following the flow described in the paper. If you wish to use the code for further prediction or analysis:

1. Input your own X-Y data (from Silvaco outputs).
2. Run the ML scripts in `/ml/` folder.
3. Adjust model hyperparameters as necessary.

---

## Potential & Next Steps

- **Community Resource:** This framework saves months of ramp-up for new Silvaco users, providing tested templates for both simulation and ML analysis.
- **Extendable:** The same logic and codebase can be adapted for other device structures or parameter sweeps.

---

## Gallery: Insert Your Graphs and Paper Graphs

> Paste your obtained results and the corresponding images from the paper in this section for a visual reference.

- _Output Characteristic Graphs_
  - Paper: ![Insert paper image here]()
  - Obtained: ![Insert my image here]()
- _Input Characteristic Graphs_
  - Paper: ![Insert paper image here]()
  - Obtained: ![Insert my image here]()
- _Breakdown Voltage_
  - Paper: ![Insert paper image here]()
  - Obtained: ![Insert my image here]()

---

## Acknowledgements

- Thanks to the authors of the reference paper for their detailed methodology.
- Silvaco documentation team (for the rare, clear sections in the 1,200-page manual).
- Early contributors and testers who improved the codebase and scripts.

---

## If you use, fork, or build upon this repository, please consider citing or referencing this README!

