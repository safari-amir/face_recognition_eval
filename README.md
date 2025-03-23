
# face_recognition_eval

face_recognition_eval is a Python-based project for evaluating face recognition models using a client-server architecture powered by FastAPI. The repository is designed to work with standard datasets such as LFW, CALFW, and CPLFW.

## Overview

This project is divided into two main components:

- **Client:** Loads image pairs from the chosen dataset, sends requests to the server for face recognition prediction, and calculates evaluation metrics.
- **Server:** Implements a FastAPI server that handles requests for model initialization and face recognition predictions.

## Features

- **Model Initialization:** Initialize a face recognition model on the server with a specified model name.
- **Face Prediction:** Send pairs of images to the server and receive predictions indicating whether the images represent a match.
- **Dataset Support:** Built-in support for common datasets:
  - **LFW:** Labeled Faces in the Wild
  - **CALFW:** Cross-Age LFW
  - **CPLFW:** Cross-Pose LFW
- **Evaluation Metrics:** Calculate accuracy, False Match Rate (FMR), and False Non-Match Rate (FNMR) based on predictions.

## Project Structure

```
face_recognition_eval/
├── LICENSE
├── .gitignore
└── src
    ├── client
    │   ├── client.py         # Client code: Initializes model, loads datasets, and sends prediction requests
    │   └── utils.py          # Utility functions: Data loading and evaluation metrics calculation
    └── server
        ├── face_rec.py       # Face recognition related functions used by the server
        └── main.py           # FastAPI server entry point
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/safari-amir/face_recognition_eval.git
   cd face_recognition_eval
   ```

2. **Create a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   If you have a `requirements.txt` file, run:

   ```bash
   pip install -r requirements.txt
   ```

   Otherwise, ensure you have the following packages installed (and any other dependencies your project requires):

   - `fastapi`
   - `uvicorn`
   - `requests`
   - `opencv-python`
   - `numpy`
   - `insightface`

## Usage

### Running the Server

Start the FastAPI server by navigating to the `src/server` directory and running:

```bash
cd src/server
uvicorn main:app --reload
```

The server will be available at `http://127.0.0.1:8000`.

### Running the Client

In a separate terminal, navigate to the `src/client` directory and run the client script to evaluate the model:

```bash
cd src/client
python client.py --model <model_name> --dataset <dataset_name>
```

Replace `<model_name>` with your chosen model identifier (e.g., `buffalo_l`) and `<dataset_name>` with one of `lfw`, `calfw`, or `cplfw`.

#### Example:
```bash
python client.py --model buffalo_l --dataset lfw
```

## How It Works

### Client (client.py)

- **Model Initialization:**  
  The `initialize_model()` function sends a POST request to the `/initialize` endpoint on the server with the desired model name.

- **Prediction:**  
  The `predict()` function sends two image files (representing a face pair) to the `/predict` endpoint. The server returns a prediction (1 for match, 0 for non-match).

- **Dataset Loading & Evaluation:**  
  The client uses a `FacePairDataLoader` (from `utils.py`) to load image pairs and ground truth labels from the specified dataset. It then iterates through the pairs, sends each for prediction, and calculates evaluation metrics (accuracy, FMR, FNMR) using the `calculate_evaluation_metrics()` function.

### Server

The server (implemented using FastAPI in `main.py` and supporting functions in `face_rec.py`) exposes the following endpoints:

- **POST /initialize:**  
  Initializes the face recognition model based on the provided `model_name`.

- **POST /predict:**  
  Accepts two image files and returns a prediction indicating whether they match.

## Configuration

- **Server URL:**  
  The client script is configured to use `http://127.0.0.1:8000` as the server URL. Adjust the `SERVER_URL` variable in `client.py` if your server runs at a different address.

- **Dataset Paths:**  
  Make sure that your dataset files and corresponding images are placed in the expected directories (e.g., `data/lfw/`, `data/calfw/`, `data/cplfw/`). You may need to modify the paths in `FacePairDataLoader` if your directory structure differs.

