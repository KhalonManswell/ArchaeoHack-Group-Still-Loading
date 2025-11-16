import ultralytics
from ultralytics import YOLO
import torch

DATA_PATH = r'/Users/khalonmanswell/Documents/GitHub/ArchaeoHack-Group-Still-Loading/archaeohack/processed_data'  # Update with your actual data path

def train_model(data_path: str, model_type: str = 'yolo11s-cls', epochs: int = 50, batch_size: int = 8, img_size: int = 200):
    """
    Train a YOLO model using the specified parameters.

    Parameters:
    - data_path (str): Path to the dataset configuration file.
    - model_type (str): Type of YOLO model to use (default is 'yolov11n').
    - epochs (int): Number of training epochs (default is 100).
    - batch_size (int): Size of each training batch (default is 16).
    - img_size (int): Size of input images (default is 640).
    """
    # Load the YOLO model
    model = YOLO(model_type)

    # Start training
    model.train(name = '1759', data=data_path, epochs=epochs, lrf = 0.001, batch=batch_size, imgsz=img_size, device="mps")
    model.export(imgsz = 200)

if __name__ == "__main__":
    
    # Example usage
    train_model(data_path=DATA_PATH, model_type='yolo11s-cls', epochs=20, batch_size=8, img_size=200)
    