import argparse
import requests
from typing import List
from utils import FacePairDataLoader, calculate_evaluation_metrics

# URL of the FastAPI server
SERVER_URL = "http://127.0.0.1:8000"

def initialize_model(model_name: str) -> None:
    """
    Initialize the face recognition model on the FastAPI server.

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
        error_msg = response.json().get("detail", "Unknown error")
        raise RuntimeError(f"Error: {error_msg}")

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
        return int(response.json().get("result", 0))
    error_msg = response.json().get("detail", "Unknown error")
    raise RuntimeError(f"Error: {error_msg}")

def main(model_name: str, dataset: str) -> None:
    """
    Main function to initialize the model, load image pairs from the chosen dataset,
    perform predictions, and evaluate the performance of the face recognition model.

    Args:
        model_name (str): The name of the model to be initialized.
        dataset (str): The dataset to use ('lfw', 'calfw', or 'cplfw').
    """
    try:
        initialize_model(model_name)
    except RuntimeError as e:
        print(e)
        return

    data_loader = FacePairDataLoader()

    dataset_paths = {
        "lfw": ("data/lfw/pairs.csv", data_loader.load_lfw_pairs),
        "calfw": ("data/calfw/pairs_CALFW.txt", data_loader.load_cacp_pairs),
        "cplfw": ("data/cplfw/pairs_CPLFW.txt", data_loader.load_cacp_pairs),
    }

    if dataset.lower() not in dataset_paths:
        print("Unsupported dataset. Please choose from 'lfw', 'calfw', or 'cplfw'.")
        return

    file_path, loader_func = dataset_paths[dataset.lower()]
    img_dir = f"data/{dataset.lower()}/aligned images" if dataset.lower() != "lfw" else None
    pairs, labels = loader_func(file_path, img_dir=img_dir) if img_dir else loader_func(file_path)

    predicted_labels: List[int] = []
    for img1, img2 in pairs:
        try:
            predicted_labels.append(predict(img1, img2))
        except RuntimeError as e:
            print(f"Skipping pair ({img1}, {img2}) due to error: {e}")

    accuracy, fmr, fnmr = calculate_evaluation_metrics(labels, predicted_labels)
    print(f"Model Evaluation - Accuracy: {accuracy:.4f}, FMR: {fmr:.4f}, FNMR: {fnmr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition Model Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Face recognition model (e.g., 'buffalo_l')")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use: 'lfw', 'calfw', or 'cplfw'")
    args = parser.parse_args()
    main(args.model, args.dataset)
