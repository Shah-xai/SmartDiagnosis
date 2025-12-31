# CV on SageMaker — Chest X-Ray Image Classification

End-to-end computer vision project demonstrating training and deployment of a **binary Chest X-Ray image classifier** using **transfer learning** with the **Amazon SageMaker built-in Image Classification** algorithm.

The project covers the full ML lifecycle: dataset preparation, S3 channel configuration, model training, evaluation via batch transform, and deployment to a real-time SageMaker endpoint.  
A **RESTful inference API** is developed on AWS using **API Gateway and Lambda**, and a lightweight **Streamlit web application** is provided for interactive inference.

---

## Project Overview

This repository implements a binary classification pipeline for detecting **Pneumonia vs. Normal** cases from Chest X-Ray images.

The workflow includes:

- Dataset download and preprocessing  
- Generation of SageMaker-compatible `.lst` label files  
- Upload of training and validation channels to Amazon S3  
- Training using SageMaker built-in Image Classification (pretrained ResNet)  
- Model evaluation using Batch Transform  
- Deployment of a real-time SageMaker inference endpoint  
- Development of a **REST API using AWS API Gateway + Lambda** to expose the model  
- A simple Streamlit UI that consumes the REST API for inference  

---

## Tech Stack

- **Amazon SageMaker** (built-in Image Classification)
- **Amazon S3**
- **AWS Lambda**
- **AWS API Gateway (REST API)**
- **Python**
- **SageMaker Python SDK**
- **Streamlit** (web app UI)

---

## Repository Contents

### `main.ipynb`
End-to-end notebook covering:
- Dataset download
- Data preprocessing and labeling
- S3 upload and channel configuration
- Model training
- Batch transform evaluation
- Endpoint deployment

### `tools/`
Utility modules used by the notebook for dataset handling and evaluation.

### `dataset/`
Local workspace created during preprocessing (not intended for long-term storage).

### `app.py`
Streamlit application for uploading images and running inference against the deployed model via the REST API.

---

## Model Details

- **Algorithm:** SageMaker built-in Image Classification  
- **Architecture:** Pretrained ResNet (101 layers)  
- **Number of classes:** 2  
- **Input:** Image data (binary)  
- **Output:** Class probabilities / predicted label  

---

## Running the Notebook

### Prerequisites
- AWS account with SageMaker permissions
- An S3 bucket for datasets and outputs
- Kaggle API credentials configured (for dataset download)
- SageMaker Studio or Jupyter environment with SageMaker SDK installed

### Steps
1. Open `main.ipynb`
2. Execute cells in order:
   - Data download and preparation
   - Dataset upload to S3
   - Model training
   - Batch transform evaluation
   - Endpoint deployment

>  Remember to delete deployed endpoints after testing to avoid unnecessary AWS charges.

---

## Inference Architecture

Inference is exposed through a **REST API**:

- **API Gateway (REST API)** receives binary image requests
- **Lambda** handles request processing and forwards data to the SageMaker endpoint
- **SageMaker Endpoint** performs inference and returns predictions
- **Streamlit app** acts as a client consuming the REST API

This design cleanly separates the frontend from the ML infrastructure and reflects standard production inference patterns.

---

## Streamlit Application

A simple Streamlit app is provided to demonstrate inference:

- Upload an image
- Send **binary image data** to the REST API
- Display prediction results returned by the backend

Run locally:

```bash
streamlit run app.py
