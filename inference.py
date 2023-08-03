import numpy as np
import torch
import io
import cv2
import argparse
from train import IMAGE_SIZE, DEVICE
from PIL import Image
from model import faster_rcnn
from torchvision import transforms as T
from flask import Flask, request, jsonify

app = Flask(__name__)


def load_model():
    global model, device
    model = faster_rcnn(2)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()


def draw_boxes_on_image(image, boxes, labels):
    image_np = np.array(image)
    for box, label in zip(boxes, labels):
        if label == 1:  # Draw predictions only for the "Car" class (class 1)
            box = [int(coord) for coord in box]
            color = (0, 255, 0)  # Green for "Car" class
            image_np = cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Draw the label text next to the bounding box
            label_text = "Car"
            label_position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            label_color = (255, 255, 255)  # White text color
            cv2.putText(image_np, label_text, label_position, font, font_scale, label_color, font_thickness)

    return image_np


@app.route('/image_predict', methods=['POST'])
def image_predict():
    global model, device
    model.to(device)
    file = request.files['file']
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)
        filtered_predictions = []
        for i in range(len(predictions)):
            scores = predictions[i]['scores']
            keep = scores >= confidence_threshold
            predictions[i]['boxes'] = predictions[i]['boxes'][keep]
            predictions[i]['labels'] = predictions[i]['labels'][keep]
            predictions[i]['scores'] = predictions[i]['scores'][keep]
            filtered_predictions.append(predictions[i])
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

    # Draw boxes on the image
    image_with_boxes = draw_boxes_on_image(image, boxes, labels)
    return jsonify({'prediction': image_with_boxes.tolist()})


@app.route('/process_video', methods=['POST'])
def process_video():
    video = request.files['file']
    video.save(video.filename)
    cap = cv2.VideoCapture(video.filename)
    # Initialize output video writer
    fourcc = cv2.VideoWriter_fourcc('V','P','8','0')
    out = cv2.VideoWriter('output.webm', fourcc, 24, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        # Read a frame from the input video
        ret, image = cap.read()
        if not ret:
            break
        global model, device
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        model.to(device)
        with torch.no_grad():
            predictions = model(image_tensor)
            filtered_predictions = []
            for i in range(len(predictions)):
                scores = predictions[i]['scores']
                keep = scores >= confidence_threshold
                predictions[i]['boxes'] = predictions[i]['boxes'][keep]
                predictions[i]['labels'] = predictions[i]['labels'][keep]
                predictions[i]['scores'] = predictions[i]['scores'][keep]
                filtered_predictions.append(predictions[i])
            boxes = predictions[0]['boxes'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()

        # Draw boxes on the image
        image_with_boxes = draw_boxes_on_image(image, boxes, labels)

        out.write(image_with_boxes)

    # Release input and output video objects
    cap.release()
    out.release()
    with open('output.webm', 'rb') as f:
        result = f.read()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        help='Specify path to your model')
    parser.add_argument("--confidence_threshold", type=str,
                        help='Specify confidence threshold')
    args = parser.parse_args()
    args = vars(args)

transform = T.Compose([T.ToTensor(),
    T.Resize((IMAGE_SIZE[1], IMAGE_SIZE[0])),  # Replace (h, w)  with your desired size,
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model_path = args["model_path"] if args["model_path"] else r'./model.pth'
model = None
device = DEVICE
confidence_threshold = args["confidence_threshold"] if args["confidence_threshold"] else 0.5

load_model()

app.run(host='0.0.0.0', debug=False)
