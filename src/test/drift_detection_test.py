import sys
import os
import pandas as pd
import numpy as np
import pytest

sys.path.append(os.getcwd())

from src.monitoring.drift_detection import DriftDetector

def test_no_drift():
    """
    Case 1: Distributions are the same. Should NOT detect drift.
    """
    print("\n--- TEST 1: Identical Distributions (Should PASS) ---")
    np.random.seed(42)
    
    ref_data = pd.DataFrame({'feature_a': np.random.normal(50, 5, 1000)})
    cur_data = pd.DataFrame({'feature_a': np.random.normal(50, 5, 1000)})
    
    detector = DriftDetector(threshold=0.05)
    report = detector.detect_drift(ref_data, cur_data)
    
    print(f"Report: {report['status']}")
    
    assert report['status'] == "PASSED"
    assert len(report['drifted_features']) == 0
    print("Test 1 Passed ✅")

def test_heavy_drift():
    """
    Case 2: Distributions are completely different. Should DETECT drift.
    """
    print("\n--- TEST 2: Shifted Mean (Should FAIL) ---")
    np.random.seed(42)
    
    # Reference: Mean=50
    ref_data = pd.DataFrame({'feature_a': np.random.normal(50, 5, 1000)})
    
    # Current: Mean=80 (Massive Shift)
    cur_data = pd.DataFrame({'feature_a': np.random.normal(80, 5, 1000)})
    
    detector = DriftDetector(threshold=0.05)
    report = detector.detect_drift(ref_data, cur_data)
    
    print(f"Report: {report['status']}")
    print(f"Drifted Features: {report['drifted_features']}")
    
    assert report['status'] == "FAILED"
    assert 'feature_a' in report['drifted_features']
    print("Test 2 Passed ✅")

if __name__ == "__main__":
    try:
        test_no_drift()
        test_heavy_drift()
        print("\nAll Drift Tests Passed Successfully! 🚀")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")