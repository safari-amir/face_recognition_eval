from insightface.app import FaceAnalysis
import cv2
import numpy as np

class FaceMatcher:
    def __init__(self, model_name: str, ctx_id: int = 0, det_size: tuple = (480, 480)):
        """
        Initialize the face analysis model.

        Args:
            model_name (str): Model name (e.g., "buffalo_s" or "buffalo_l").
            ctx_id (int, optional): The context ID. Defaults to 0.
            det_size (tuple, optional): Detection size. Defaults to (640, 640).
        """
        self.model = FaceAnalysis(name=model_name)
        self.model.prepare(ctx_id=ctx_id, det_size=det_size)

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image by converting it from BGR to RGB format.
        
        Note: This function assumes the image is provided as a buffer.
        
        Args:
            image (np.ndarray): The input image in BGR format.
        
        Returns:
            np.ndarray: The preprocessed image in RGB format.
        """
        # Convert buffer to numpy array and decode the image
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # Convert from BGR to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        return image

    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute the cosine similarity between two face embeddings.

        Cosine similarity is defined as:
            similarity = (embedding1 dot embedding2) / (||embedding1|| * ||embedding2||)
        
        Args:
            embedding1 (np.ndarray): The first face embedding.
            embedding2 (np.ndarray): The second face embedding.
        
        Returns:
            float: The cosine similarity between the two embeddings.
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def get_face_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Get the face embedding from an image.

        Args:
            image (np.ndarray): The input image in BGR format (as a buffer).
        
        Returns:
            np.ndarray: The face embedding.
        
        Raises:
            ValueError: If no face is detected in the image.
        """
        # Preprocess the image (BGR to RGB)
        image = self.preprocess_image(image)
        # Get faces from the image
        faces = self.model.get(image)
        if not faces:
            return None
        else:
            return faces[0].embedding

    def match_face(self, image1: np.ndarray, image2: np.ndarray, threshold: float = 0.25) -> bool:
        """
        Match two faces by comparing their embeddings using cosine similarity.

        Args:
            image1 (np.ndarray): The first image (in BGR format as a buffer).
            image2 (np.ndarray): The second image (in BGR format as a buffer).
            threshold (float, optional): The similarity threshold. Defaults to 0.25.
        
        Returns:
            bool: True if the cosine similarity is above the threshold, False otherwise.
        """
        embedding1 = self.get_face_embedding(image1)
        embedding2 = self.get_face_embedding(image2)
        if embedding1 is None or embedding2 is None:
            return False
        else:
            similarity = self.cosine_similarity(embedding1, embedding2)
            return similarity > threshold
