import os
import shutil
import yaml
import torch

# Set up directory paths 
project_path = 'C:/Users/sshak/Documents/GitHub/AER850_Project3'
dataset_path = 'C:/Users/sshak/Documents/GitHub/AER850_Project3/data/train/images'
yaml_path = 'C:/Users/sshak/Documents/GitHub/AER850_Project3/data/data.yaml'
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt') 

# Train the model
results = model.train(data='C:/Users/sshak/Documents/GitHub/AER850_Project3/data/data.yaml', epochs=5, imgsz=900, batch=16)

metrics = model.val()

#-- Part 3
from PIL import Image
from ultralytics import YOLO
import time

results1 = model('C:/Users/sshak/Documents/GitHub/AER850_Project3/data/evaluation/ardmega.jpg')
results2 = model('C:/Users/sshak/Documents/GitHub/AER850_Project3/data/evaluation/arduno.jpg')
results3 = model('C:/Users/sshak/Documents/GitHub/AER850_Project3/data/evaluation/rasppi.jpg')
# Evaluate the model

for r1 in results1:
    im_array = r1.plot() 
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('results1.jpg')

for r2 in results2:
    im_array = r2.plot() 
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('results2.jpg')

for r3 in results3:
    im_array = r3.plot() 
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('results3.jpg')