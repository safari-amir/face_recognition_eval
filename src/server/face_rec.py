from typing import Optional
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceMatcher:
    """
    A class for face matching using the InsightFace model.

    This class performs face detection and embedding extraction, then compares embeddings
    using cosine similarity.
    """

    def __init__(self, model_name: str, ctx_id: int = 0, det_size: tuple = (480, 480)):
        """
        Initialize the face analysis model.

        Args:
            model_name (str): Model name (e.g., "buffalo_s" or "buffalo_l").
            ctx_id (int, optional): The context ID (GPU/CPU). Defaults to 0.
            det_size (tuple, optional): Detection size. Defaults to (480, 480).
        """
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.model = None
        self.prepare_model()

    def prepare_model(self) -> None:
        """Prepare the face analysis model."""
        self.model = FaceAnalysis(name=self.model_name)
        self.model.prepare(ctx_id=self.ctx_id, det_size=self.det_size)

    @staticmethod
    def preprocess_image(image: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess the input image by decoding and converting it from BGR to RGB.

        Args:
            image (np.ndarray): The input image as a byte buffer.

        Returns:
            Optional[np.ndarray]: The preprocessed image in RGB format, or None if decoding fails.
        """
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            return None  # Handle invalid images
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute the cosine similarity between two face embeddings.

        Args:
            embedding1 (np.ndarray): The first face embedding.
            embedding2 (np.ndarray): The second face embedding.

        Returns:
            float: Cosine similarity score.
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def get_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the face embedding from an image.

        Args:
            image (np.ndarray): The input image as a byte buffer.

        Returns:
            Optional[np.ndarray]: The face embedding, or None if no face is detected.

        Raises:
            ValueError: If no face is detected in the image.
        """
        image = self.preprocess_image(image)
        if image is None:
            raise ValueError("Invalid image format or decoding failure.")

        faces = self.model.get(image)
        return faces[0].embedding if faces else None

    def match_face(self, image1: np.ndarray, image2: np.ndarray, threshold: float = 0.25) -> Optional[bool]:
        """
        Compare two faces and determine if they match.

        Args:
            image1 (np.ndarray): The first image as a byte buffer.
            image2 (np.ndarray): The second image as a byte buffer.
            threshold (float, optional): Similarity threshold. Defaults to 0.25.

        Returns:
            Optional[bool]: True if the similarity is above the threshold, False otherwise, or None if faces are missing.
        """
        try:
            embedding1 = self.get_face_embedding(image1)
            embedding2 = self.get_face_embedding(image2)
        except ValueError:
            return None  # Handle cases where no faces are detected

        if embedding1 is None or embedding2 is None:
            return None  # Handle cases where face detection fails

        return self.cosine_similarity(embedding1, embedding2) > threshold
