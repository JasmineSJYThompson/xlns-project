import numpy as np
import gzip

# Took this code from ChatGPT but basically it removes the headers

def parse_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    # Convert first 16 bytes to integers (big-endian)
    magic_number, num_images, rows, cols = np.frombuffer(data[:16], dtype='>u4')
    
    # Reshape the remaining data into (num_images, rows, cols)
    return data[16:].reshape(num_images, rows, cols)

def parse_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    # Convert first 8 bytes to integers (big-endian)
    magic_number, num_labels = np.frombuffer(data[:8], dtype='>u4')
    
    # Extract labels
    return data[8:]

def get_mnist_data_numpy_format():
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    
    train_images = parse_images(files[0])
    test_images = parse_images(files[2])
    train_labels = parse_labels(files[1])
    test_labels = parse_labels(files[3])
    
    return train_images, train_labels, test_images, test_labels