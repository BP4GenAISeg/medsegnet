import unittest
import torch
import numpy as np
from omegaconf import OmegaConf
from utils.metrics import dice_score

class TestDiceScore(unittest.TestCase):
    def setUp(self):
        cfg = OmegaConf.load('./conf/config.yaml')
        self.num_classes = cfg.training.num_classes
        self.epsilon = 1e-6

    def test_perfect_match(self):
        # Test case where predictions perfectly match the ground truth.
        y_pred = torch.tensor([[[[2, 1], [2, 2]]]])  # [B, D, H, W]
        y_true = torch.tensor([[[[2, 1], [2, 2]]]])  # [B, D, H, W]

        y_pred_logits = torch.nn.functional.one_hot(y_pred, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        score = dice_score(y_pred_logits, y_true, num_classes=self.num_classes, epsilon=self.epsilon)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_no_overlap(self):
        # Test case where predictions and ground truth have no overlap.
        y_pred = torch.tensor([[[[0, 1], [2, 2]]]])  # [B, D, H, W]
        y_true = torch.tensor([[[[2, 0], [1, 1]]]])  # Completely different

        y_pred_logits = torch.nn.functional.one_hot(y_pred, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        score = dice_score(y_pred_logits, y_true, num_classes=self.num_classes, epsilon=self.epsilon)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_partial_overlap(self):
        # Test case where predictions and ground truth partially overlap.
        y_pred = torch.tensor([[[[0, 1], [1, 2]]]])  # [B, D, H, W]
        y_true = torch.tensor([[[[0, 1], [2, 2]]]])  # Some overlap, some mismatch

        y_pred_logits = torch.nn.functional.one_hot(y_pred, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        score = dice_score(y_pred_logits, y_true, num_classes=self.num_classes, epsilon=self.epsilon)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

if __name__ == "__main__":
    unittest.main()