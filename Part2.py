import os
import shutil
import yaml

# Set up directory paths
project_path = 'C:/Users/sshak/Documents/GitHub/AER850_Project3'
dataset_path = 'C:/Users/sshak/Documents/GitHub/AER850_Project3/data/train/images'
yaml_path = 'C:/Users/sshak/Documents/GitHub/AER850_Project3/data/data.yaml'
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='C:/Users/sshak/Documents/GitHub/AER850_Project3/data/data.yaml', epochs=10, imgsz=900, batch=10)

# Install dependencies
os.system("pip install -U -r requirements.txt")

# Create a YAML configuration file
config_content = f"""
train: {dataset_path}/train.txt
val: {dataset_path}/val.txt

nc: 1  # Number of classes (adjust based on your dataset)
names: ['component']  # Class names

epochs: 50
batch_size: 16
img_size: 900
"""

with open('yolov8_config.yaml', 'w') as config_file:
    config_file.write(config_content)

# Copy YOLOv5 weights as a starting point
shutil.copy('yolov5s.pt', 'yolov5s_initial.pt')

# Training command
training_command = """
python train.py --img 900 --batch 16 --epochs 50 --data yolov8_config.yaml --weights yolov5s_initial.pt
"""

# Run training command
os.system(training_command)