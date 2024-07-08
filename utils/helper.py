import pickle
import os

def train_val_test_split(data, test_size, val_size):
    # Split data into train, test, and validation sets
    train_data, test_data = train_test_split(data, test_size=test_size)
    train_data, val_data = train_test_split(train_data, test_size=val_size)
    return train_data, val_data, test_data

def train_test_split(data, test_size):
    # Split data into train and test sets
    split_idx = int(len(data) * (1 - test_size))
    return data[:split_idx], data[split_idx:]

def save_model(path, model):
    pickle.dump(model, open(path, 'wb'))

def load_model(path):
    assert os.path.exists(path), "File does not exist"
    print('Loading parameters...')
    with open(path, 'rb') as f:
        model = pickle.load(f)
        print('Model loaded successfully!')
    return model
