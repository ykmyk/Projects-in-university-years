import numpy as np
import os

# Configuration
ORIGINAL_FILE = "mnist.train.npz"
TRAIN_FILE = "local_train.npz"
TEST_FILE = "local_test.npz"
TEST_SIZE = 9999

def create_split():
    if not os.path.exists(ORIGINAL_FILE):
        print(f"Error: {ORIGINAL_FILE} not found. Run your main script once to download it.")
        return

    print(f"Loading {ORIGINAL_FILE}...")
    data = np.load(ORIGINAL_FILE)
    
    # Extract data
    keys = list(data.keys())
    # Assuming 'data' and 'target' are the keys based on your Dataset class
    X = data['data']
    y = data['target']
    
    # Shuffle indices
    indices = np.random.permutation(len(X))
    test_idx = indices[:TEST_SIZE]
    train_idx = indices[TEST_SIZE:]
    
    # Create splits
    print(f"Splitting: {len(train_idx)} Train, {len(test_idx)} Test")
    
    # Save Train
    np.savez(TRAIN_FILE, data=X[train_idx], target=y[train_idx])
    print(f"Saved {TRAIN_FILE}")
    
    # Save Test
    np.savez(TEST_FILE, data=X[test_idx], target=y[test_idx])
    print(f"Saved {TEST_FILE}")

if __name__ == "__main__":
    create_split()