# CEKT

A concise description of your project. Provide context and an overview of the purpose and functionality.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Execution Steps](#execution-steps)
5. [Notes](#notes)

---

## Introduction

This repository contains a series of Python scripts designed to accomplish specific tasks in a sequential manner. Each script serves a unique purpose within the pipeline. Follow the steps below to understand and execute the project efficiently.

## Requirements

Ensure the following are installed on your system:

- Python 3.8+
- Required Python libraries (install using the `requirements.txt` provided).
- CUDA-enabled GPU (for training and testing).

## Setup

1. Clone the repository:

   ```bash
    git clone https://github.com/xinjiesun-ustc/CEKT.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Execution Steps

Execute the scripts in the order provided below:

1. **Extract relevant files:**
   
   ```bash
   python Find_C_file_from_alldata.py
   ```

   This script processes and extracts relevant C files from the dataset.

2. **Generate embeddings for files:**

   ```bash
   python Num2Embedding_codenet_C.py
   ```

   Converts numerical data to embeddings for downstream processing.

3. **Preprocess data:**

   ```bash
   python data_CodeNet_C_preprocessing.py
   ```

   Handles preprocessing tasks to prepare the data for training.

4. **Train the model:**

   ```bash
   python train.py cuda:1 20
   ```

   Trains the model using the prepared data. Pass CUDA device and epoch count as arguments (e.g., `cuda:1` and `20`).

5. **Test the model:**

   ```bash
   python test.py cuda:1
   ```



