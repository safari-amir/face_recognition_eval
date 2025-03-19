import os
import argparse
import requests
from typing import List, Tuple
from utils import FacePairDataLoader, calculate_evaluation_metrics

# URL of the FastAPI server
SERVER_URL = 'http://127.0.0.1:8000'

def initialize_model(model_name: str) -> None:
    """
    Initialize the face recognition model on the FastAPI server.
    
    This function sends a POST request to the server's initialization endpoint.
    
    Args:
        model_name (str): The name of the model to initialize (e.g., 'buffalo_l').
    
    Raises:
        RuntimeError: If the server responds with an error.
    """
    url = f"{SERVER_URL}/initialize"
    payload = {"model_name": model_name}
    response = requests.post(url, params=payload)
    
    if response.status_code == 200:
        print(response.json().get("message", "Model initialized successfully."))
    else:
        raise RuntimeError(f"Error: {response.json().get('detail', 'Unknown error')}")

def predict(image_path1: str, image_path2: str) -> int:
    """
    Send two images to the FastAPI server for face recognition prediction.
    
    Args:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.
    
    Returns:
        int: 1 if the server indicates a match; 0 otherwise.
    
    Raises:
        RuntimeError: If the server responds with an error.
    """
    url = f"{SERVER_URL}/predict"
    with open(image_path1, "rb") as file1, open(image_path2, "rb") as file2:
        files = {"file1": file1, "file2": file2}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        return int(response.json().get('result', 0))
    else:
        raise RuntimeError(f"Error: {response.json().get('detail', 'Unknown error')}")

def main(model_name: str, dataset: str) -> None:
    """
    Main function to initialize the model, load image pairs from the chosen dataset,
    perform predictions, and evaluate the performance of the face recognition model.
    
    Args:
        model_name (str): The name of the model to be initialized.
        dataset (str): The dataset to use ('lfw', 'calfw', or 'cplfw').
    """
    # Initialize the model on the server
    try:
        initialize_model(model_name)
    except RuntimeError as e:
        print(e)
        return

    # Create an instance of the data loader class
    data_loader = FacePairDataLoader()
    
    # Load image pairs and ground truth labels based on the chosen dataset
    if dataset.lower() == "lfw":
        pairs, labels = data_loader.load_lfw_pairs('data/lfw/pairs.csv')
    elif dataset.lower() == "calfw":
        pairs, labels = data_loader.load_cacp_pairs('data/calfw/pairs_CALFW.txt', img_dir='data/calfw/aligned images')
    elif dataset.lower() == "cplfw":
        pairs, labels = data_loader.load_cacp_pairs('data/cplfw/pairs_CPLFW.txt', img_dir='data/cplfw/aligned images')
    else:
        print("Unsupported dataset. Please choose from 'lfw', 'calfw', or 'cplfw'.")
        return

    # Perform predictions for each image pair
    predicted_labels: List[int] = []
    for img1, img2 in pairs:
        try:
            pred_label = predict(img1, img2)
            predicted_labels.append(pred_label)
        except RuntimeError as e:
            print(f"Skipping pair ({img1}, {img2}) due to error: {e}")

    # Calculate evaluation metrics based on the true and predicted labels
    accuracy, fmr, fnmr = calculate_evaluation_metrics(labels, predicted_labels)
    print(f"Model Evaluation - Accuracy: {accuracy:.4f}, FMR: {fmr:.4f}, FNMR: {fnmr:.4f}")

if __name__ == "__main__":
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Face Recognition Model Evaluation")
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the face recognition model (e.g., 'buffalo_l')")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset to use: 'lfw', 'calfw', or 'cplfw'")
    
    args = parser.parse_args()
    main(args.model, args.dataset)
# Example usage:
# python client.py --model buffalo_l --dataset lfw
# python client.py --model buffalo_s --dataset calfw    