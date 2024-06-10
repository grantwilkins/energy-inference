# Energy-Inference: A Project in Energy Usage of LLM Inference

This directory contains the results of the Energy-Inference project. The project was a thesis project on the energy usage of large language models (LLMs) in inference. The manuscript is available in this directory.

## Directory Structure

- `stats/`: Contains largely unorganized energy data used in the project.
- `mps/`: Contains the code for inference and energy monitoring inference on an Apple Silicon Mac.
- `cuda`: Contains the code for inference and energy monitoring inference on an NVIDIA GPU.
- `plots/`: Contains the plots, results, and simulations of the project.
- `job-scripts/`: Contains the job scripts used to run the inference and energy monitoring inference on the Palmetto and Swing supercomputers.

## Usage

The project is divided into two parts: inference and energy monitoring inference. The inference part is the process of running a model on a dataset and obtaining the predictions. The energy monitoring inference part is the process of running a model on a dataset, obtaining the predictions, and monitoring the energy usage of the model.

In the `mps/` and `cuda/` directories, there are Python scripts that run the inference and energy monitoring inference on an Apple Silicon Mac and an NVIDIA GPU, respectively.

## Installation

To run the code in this project, you can install the following dependencies:

```bash
conda env create -f environment.yml
```

## Thesis

The thesis for this project is available in this directory. The thesis is a comprehensive report on the energy usage of large language models in inference.
