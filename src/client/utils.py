import os
from typing import List, Tuple

class FacePairDataLoader:
    """
    A class for loading and pairing face images from different datasets.
    
    This class provides methods to load image pairs and their corresponding labels 
    from:
      - CALFW/CPLFW datasets.
      - LFW dataset.
    """

    def load_cacp_pairs(self, file_path: str, img_dir: str) -> Tuple[List[Tuple[str, str]], List[int]]:
        """
        Load and pair images from the CALFW/CPLFW datasets based on a text file.

        The text file should contain an even number of lines, where each pair of lines 
        corresponds to a pair of images. The first element in each line is the image filename,
        and the second element is a flag (a non-'0' value indicates a positive match).

        Args:
            file_path (str): Path to the text file with pairing information.
            img_dir (str): Directory where the images are stored.

        Returns:
            Tuple[List[Tuple[str, str]], List[int]]:
                - A list of tuples, each containing two image paths.
                - A list of labels (1 for a positive match, 0 for a negative match).
        """
        pairs = []
        labels = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                parts1 = lines[i].strip().split(' ')
                parts2 = lines[i + 1].strip().split(' ')
                img1 = os.path.join(img_dir, parts1[0])
                img2 = os.path.join(img_dir, parts2[0])
                pairs.append((img1, img2))
                label = 1 if parts1[1] != '0' else 0
                labels.append(label)
        return pairs, labels

    def load_lfw_pairs(self, pairs_file: str, lfw_dir: str = 'data/lfw/lfw-deepfunneled') -> Tuple[List[Tuple[str, str]], List[int]]:
        """
        Load and pair images from the LFW dataset based on a text file.

        The text file should have a header line followed by data lines. Each data line can contain either:
          - Three elements: a positive pair (same person, two image numbers).
          - Four elements: a negative pair (two different persons with one image each).

        Args:
            pairs_file (str): Path to the text file with LFW pairing information.
            lfw_dir (str, optional): Directory where the LFW images are stored. Defaults to 'lfw'.

        Returns:
            Tuple[List[Tuple[str, str]], List[int]]:
                - A list of tuples, each containing two image paths.
                - A list of labels (1 for a positive match, 0 for a negative match).
        """
        pairs = []
        labels = []
        with open(pairs_file, 'r') as f:
            # Skip header line
            for line in f.readlines()[1:]:
                elements = line.strip().split(',')
                if elements[-1] == '':
                    person, img_num1, img_num2 = elements[:3]
                    img1 = os.path.join(lfw_dir, person, f"{person}_{img_num1.zfill(4)}.jpg")
                    img2 = os.path.join(lfw_dir, person, f"{person}_{img_num2.zfill(4)}.jpg")
                    label = 1
                elif len(elements) == 4:
                    person1, img_num1, person2, img_num2 = elements
                    img1 = os.path.join(lfw_dir, person1, f"{person1}_{img_num1.zfill(4)}.jpg")
                    img2 = os.path.join(lfw_dir, person2, f"{person2}_{img_num2.zfill(4)}.jpg")
                    label = 0
                else:
                    continue  # Skip unexpected format lines
                pairs.append((img1, img2))
                labels.append(label)
        return pairs, labels


def calculate_evaluation_metrics(true_labels: List[int], predicted_labels: List[int]) -> Tuple[float, float, float]:
    """
    Calculate evaluation metrics for face recognition performance.

    Metrics include:
      - Accuracy: (TP + TN) / total samples.
      - False Match Rate (FMR): FP / (FP + TN) - rate of false positive matches.
      - False Non-Match Rate (FNMR): FN / (FN + TP) - rate of missed positive matches.

    Args:
        true_labels (List[int]): Ground truth labels (1 for positive, 0 for negative).
        predicted_labels (List[int]): Model's predicted labels (1 for positive, 0 for negative).

    Returns:
        Tuple[float, float, float]:
            A tuple containing accuracy, FMR, and FNMR.
    """
    tp = sum(1 for actual, pred in zip(true_labels, predicted_labels) if actual == 1 and pred == 1)
    tn = sum(1 for actual, pred in zip(true_labels, predicted_labels) if actual == 0 and pred == 0)
    fp = sum(1 for actual, pred in zip(true_labels, predicted_labels) if actual == 0 and pred == 1)
    fn = sum(1 for actual, pred in zip(true_labels, predicted_labels) if actual == 1 and pred == 0)

    total = len(true_labels)
    accuracy = (tp + tn) / total if total > 0 else 0
    fmr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnmr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return accuracy, fmr, fnmr
