import yaml
import gdown
import os
import sys
import torch
sys.path.append("src")
# Define the paths and filenames


def create_save_result_folder(path):
    """
        Create a folder at the specified path if it does not exist.

        Args:
            path (str): The path to the folder.
        Return: None
        """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created.")
    else:
        print(f"Folder '{path}' already exists.")


def load_model(url, output_path):
    """
        Download a model from the given URL and save it to the specified output path
        if the model does not exist at the output path.

        Args:
            url (str): The URL from which to download the model.
            output_path (str): The path to save the downloaded model.
        Return: None
        """
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
        print("Model loaded!")
    else:
        print(f"Model '{output_path}' already exists.")


# Access configuration values
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
SAVED_MODEL_FOLDER = config['saved_model_folder']
DATA_PATH = config['data_path']
MODEL_FILE = config['model_file']
MODEL_URL = config['model_url']
LR = config['lr']
EPOCHS = config['batch_size']
BATCH_SIZE = config['number_of_epochs']
IMAGE_SIZE = (config['image_height'], config['image_width'])
CONFIDENCE_THRESHOLD = config['confidence_threshold']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Create Examples folder
create_save_result_folder(SAVED_MODEL_FOLDER)
# load model
load_model(url=MODEL_URL, output_path=os.path.join(SAVED_MODEL_FOLDER, MODEL_FILE))