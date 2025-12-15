import argparse
import numpy as np
import sys
import pickle # Needed for the fix

# --- THE FIX STARTS HERE ---
import miniaturization

# 1. Import the class explicitly
from miniaturization import MLPFullDistributionClassifier

# 2. Map the class to the current '__main__' namespace
# This tells pickle: "When you look for __main__.MLPFullDistributionClassifier, look here!"
sys.modules['__main__'].MLPFullDistributionClassifier = MLPFullDistributionClassifier
# --- THE FIX ENDS HERE ---

# Import your main function
from miniaturization import main

def verify():
    TEST_FILE = "local_test.npz"
    MODEL_FILE = "miniaturization.model"
    
    print(f"Loading Ground Truth from {TEST_FILE}...")
    try:
        data = np.load(TEST_FILE)
        y_true = data['target']
    except FileNotFoundError:
        print("Error: Run create_split.py first!")
        return

    print("Running Prediction via your main function...")
    args = argparse.Namespace(
        predict=TEST_FILE,
        recodex=False,
        seed=42,
        model_path=MODEL_FILE
    )
    
    y_pred = main(args)
    
    if y_pred is None:
        print("Error: Your main function returned None.")
        return

    # Calculate Accuracy
    acc = np.mean(y_pred == y_true)
    print("-" * 30)
    print(f"Local Accuracy: {acc * 100:.4f}%")
    print("-" * 30)
    
    if acc > 0.99:
        print("✅ SUCCESS: You are ready to submit!")
    else:
        print("❌ KEEP GOING: You need more accuracy.")

if __name__ == "__main__":
    verify()