from PIL import Image
from ultralytics import YOLO
from PIL import Image

 # Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# # Run inference on 'bus.jpg'
# results = model(r'C:\Users\sshak\Documents\GitHub\AER850_Project3\data')  # results list

# # Show the results
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()  # show image
#     im.save('results.jpg')  # save image

# Evaluate the model
for image_path in [r'C:/Users/sshak/Documents/GitHub/AER850_Project3/data/evaluation/ardmega.jpg',
                    r'C:/Users/sshak/Documents/GitHub/AER850_Project3/data/evaluation/arduino.jpg',
                    r'C:/Users/sshak/Documents/GitHub/AER850_Project3/data/evaluation/rasppi.jpg']:
    img = Image.open(image_path)
    results = model.predict(img, save=True)
    print(results)  # print results to stdout