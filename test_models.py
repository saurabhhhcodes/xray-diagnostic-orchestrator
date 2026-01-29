
import unittest
import numpy as np
import cv2
import os
from xray_service import NIHPredictor, RSNAPredictor, PadChestPredictor

class TestXRayModels(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a dummy image for testing
        cls.dummy_img_path = 'test_xray_dummy.png'
        # Create a 512x512 random noise image (simulating X-Ray)
        dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(cls.dummy_img_path, dummy_img)
        print(f"Created dummy image: {cls.dummy_img_path}")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.dummy_img_path):
            os.remove(cls.dummy_img_path)

    def test_nih_model(self):
        print("\nTesting NIH Predictor...")
        if not os.path.exists('xray_model.h5'):
            print("Skipping NIH test (model not found)")
            return
        
        predictor = NIHPredictor('xray_model.h5')
        results = predictor.predict(self.dummy_img_path)
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 15)
        self.assertIn('Pneumonia', results)
        
        # Test GradCAM
        heatmap, label = predictor.make_gradcam(self.dummy_img_path)
        # Heatmap will be 7x7 (DenseNet feature map), resizing happens in frontend
        self.assertEqual(heatmap.shape, (7, 7))
        print("NIH Test Passed.")

    def test_rsna_model(self):
        print("\nTesting RSNA Predictor...")
        if not os.path.exists('rsna_pneumonia.h5'):
            print("Skipping RSNA test (model not found)")
            return

        predictor = RSNAPredictor('rsna_pneumonia.h5')
        results = predictor.predict(self.dummy_img_path)
        self.assertIsInstance(results, dict)
        self.assertIn('Pneumonia', results)
        self.assertIn('Normal', results)
        self.assertTrue(0 <= results['Pneumonia'] <= 1)
        print("RSNA Test Passed.")

    def test_padchest_model(self):
        print("\nTesting PadChest Predictor...")
        if not os.path.exists('padchest_sample_model.h5'):
            print("Skipping PadChest test (model not found)")
            return

        predictor = PadChestPredictor('padchest_sample_model.h5')
        results = predictor.predict(self.dummy_img_path)
        self.assertIsInstance(results, dict)
        print("PadChest Test Passed.")

if __name__ == '__main__':
    unittest.main()
