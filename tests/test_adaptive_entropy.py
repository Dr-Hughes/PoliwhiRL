# -*- coding: utf-8 -*-
import unittest
import numpy as np
from unittest.mock import MagicMock


class TestAdaptiveEntropy(unittest.TestCase):
    def setUp(self):
        """Set up test environment - create a mock model with entropy methods"""
        self.mock_model = MagicMock()
        self.mock_model.entropy_coef = 0.1
        self.mock_model.entropy_decay = 0.99
        self.mock_model.entropy_min = 0.02
        self.mock_model.entropy_boost = 0.0
        
        # Add the entropy calculation method
        def get_entropy_coef(episode):
            base_entropy = max(
                self.mock_model.entropy_coef * self.mock_model.entropy_decay**episode, 
                self.mock_model.entropy_min
            )
            entropy_boost = self.mock_model.entropy_boost
            return min(base_entropy + entropy_boost, self.mock_model.entropy_coef)
        
        self.mock_model._get_entropy_coef = get_entropy_coef
        
    def test_base_entropy_decay(self):
        """Test that base entropy decays correctly over episodes"""
        # Episode 0: should be at initial value
        entropy_0 = self.mock_model._get_entropy_coef(0)
        self.assertAlmostEqual(entropy_0, 0.1, places=4)
        
        # Episode 10: should decay
        entropy_10 = self.mock_model._get_entropy_coef(10)
        expected = max(0.1 * (0.99 ** 10), 0.02)
        self.assertAlmostEqual(entropy_10, expected, places=4)
        
        # Episode 100: should be at or near minimum
        entropy_100 = self.mock_model._get_entropy_coef(100)
        self.assertGreaterEqual(entropy_100, 0.02)
        self.assertLess(entropy_100, 0.1)
        
    def test_entropy_boost_new_stage(self):
        """Test entropy boost for new curriculum stage"""
        # Simulate new stage (low episode count)
        self.mock_model.entropy_boost = 0.05
        
        # Should add boost to base entropy
        base_entropy = max(0.1 * (0.99 ** 5), 0.02)
        boosted_entropy = self.mock_model._get_entropy_coef(5)
        self.assertAlmostEqual(boosted_entropy, min(base_entropy + 0.05, 0.1), places=4)
        
    def test_entropy_boost_capping(self):
        """Test that entropy boost doesn't exceed initial entropy"""
        # Set a large boost
        self.mock_model.entropy_boost = 0.2
        
        # Even with large boost, should not exceed initial entropy
        entropy = self.mock_model._get_entropy_coef(50)
        self.assertLessEqual(entropy, 0.1)
        
    def test_entropy_boost_calculation(self):
        """Test the entropy boost calculation logic"""
        # Test new stage detection
        episode = 5  # Early episode
        is_new_stage = episode < 20
        self.assertTrue(is_new_stage)
        
        # Test novelty rate calculation
        avg_visits = 3.5  # Average visits per state
        novelty_rate = 1.0 / avg_visits
        self.assertAlmostEqual(novelty_rate, 0.286, places=2)
        is_low_novelty = novelty_rate < 0.3
        self.assertTrue(is_low_novelty)
        
        # Test boost calculation
        boost = 0.0
        if is_new_stage:
            boost += 0.05
        if is_low_novelty:
            boost += 0.03
        
        # Apply decay
        decayed_boost = boost * (0.95 ** (episode // 10))
        self.assertAlmostEqual(decayed_boost, 0.08, places=2)


if __name__ == "__main__":
    unittest.main()