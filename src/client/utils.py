import os
from typing import List, Tuple


class FacePairDataLoader:
    """
    A class for loading and pairing face images from different datasets.
    """

    @staticmethod
    def load_cacp_pairs(file_path: str, img_dir: str) -> Tuple[List[Tuple[str, str]], List[int]]:
        """
        Load and pair images from the CALFW/CPLFW datasets.
        """
        pairs, labels = [], []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            for i in range(0, len(lines), 2):
                parts1 = lines[i].strip().split()
                parts2 = lines[i + 1].strip().split()
                
                img1 = os.path.join(img_dir, parts1[0])
                img2 = os.path.join(img_dir, parts2[0])
                
                pairs.append((img1, img2))
                labels.append(1 if parts1[1] != '0' else 0)
        
        return pairs, labels

    @staticmethod
    def load_lfw_pairs(pairs_file: str, lfw_dir: str = 'data/lfw/lfw-deepfunneled') -> Tuple[List[Tuple[str, str]], List[int]]:
        """
        Load and pair images from the LFW dataset.
        """
        pairs, labels = [], []
        
        with open(pairs_file, 'r', encoding='utf-8') as file:
            for line in file.readlines()[1:]:  # Skip header line
                elements = line.strip().split(',')
                
                if not elements[-1]:  # Positive pair
                    person, img_num1, img_num2 = elements[:3]
                    img1 = os.path.join(lfw_dir, person, f"{person}_{img_num1.zfill(4)}.jpg")
                    img2 = os.path.join(lfw_dir, person, f"{person}_{img_num2.zfill(4)}.jpg")
                    label = 1
                elif len(elements) == 4:  # Negative pair
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
    """
    tp = sum(1 for actual, pred in zip(true_labels, predicted_labels) if actual == 1 and pred == 1)
    tn = sum(1 for actual, pred in zip(true_labels, predicted_labels) if actual == 0 and pred == 0)
    fp = sum(1 for actual, pred in zip(true_labels, predicted_labels) if actual == 0 and pred == 1)
    fn = sum(1 for actual, pred in zip(true_labels, predicted_labels) if actual == 1 and pred == 0)
    
    total = len(true_labels)
    accuracy = (tp + tn) / total if total else 0
    fmr = fp / (fp + tn) if (fp + tn) else 0
    fnmr = fn / (fn + tp) if (fn + tp) else 0
    
    return accuracy, fmr, fnmr