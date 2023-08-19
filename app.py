import io
import json
import cv2
import numpy as np
import base64
from PIL import Image
import streamlit as st

import io
import json
import cv2
import numpy as np
import base64
from PIL import Image
import streamlit as st

class Detection:
    def __init__(self, model_path: str, classes: list):
        self.model_path = model_path
        self.classes = classes
        self.model = self.__load_model()

    def __load_model(self):
        net = cv2.dnn.readNet(self.model_path)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def __extract_output(self, preds, image_shape, input_shape, score=0.1, nms=0.0, confidence=0.0):
        class_ids, confs, boxes = [], [], []

        image_height, image_width = image_shape
        input_height, input_width = input_shape
        x_factor = image_width / input_width
        y_factor = image_height / input_height

        rows = preds[0].shape[0]
        for i in range(rows):
            row = preds[0][i]
            conf = row[4]

            classes_score = row[4:]
            _, _, _, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            if classes_score[class_id] > score:
                confs.append(conf)
                label = self.classes[int(class_id)]
                class_ids.append(label)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

        r_class_ids, r_confs, r_boxes = [], [], []
        indexes = cv2.dnn.NMSBoxes(boxes, confs, confidence, nms)
        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i] * 100)
            r_boxes.append(boxes[i].tolist())

        return {
            'boxes': r_boxes,
            'confidences': r_confs,
            'classes': r_class_ids
        }

    def __call__(self, image, width=640, height=640, score=0.1, nms=0.0, confidence=0.0):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (width, height), swapRB=True, crop=False)
        self.model.setInput(blob)
        preds = self.model.forward()
        preds = preds.transpose((0, 2, 1))

        results = self.__extract_output(
            preds=preds,
            image_shape=image.shape[:2],
            input_shape=(height, width),
            score=score,
            nms=nms,
            confidence=confidence
        )
        return results

# Rest of the code remains the same

CLASSES_YOLO = [
    'Pistol', 'Rifle', 'Machine Gun', 'Rocket Launcher', 'Sword', 'Pike', 'Knife'
]

detection = Detection(
    model_path='best.onnx',
    classes=CLASSES_YOLO
)

# Streamlit app starts here
st.title("Object Detection Streamlit App")

file = st.file_uploader("Upload an image", type=["jpg", "png"])

def draw_annotations(image, results):
    boxes = results['boxes']
    confidences = results['confidences']
    classes = results['classes']

    for box, confidence, class_name in zip(boxes, confidences, classes):
        left, top, width, height = box
        cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 255), 2)  # Purple bounding box
        label = f"{class_name}: {confidence:.2f}%"

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_left = max(left, 0)
        label_top = max(top - 10, label_size[1])

        cv2.putText(image, label, (label_left, label_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)  # Yellow font

    return image

# if file is not None:
#     image = Image.open(file).convert("RGB")
#     image = np.array(image)
#     image = image[:, :, ::-1].copy()
#     results = detection(image)

#     annotated_image = draw_annotations(image.copy(), results)

#     # Display annotated image
#     st.image(annotated_image, caption="Annotated Image", use_column_width=True)

#     # Display results
#     st.json(results)

if file is not None:
    image = Image.open(file).convert("RGB")
    image_np = np.array(image)
    results = detection(image_np)

    annotated_image = draw_annotations(image_np.copy(), results)

    # Display original and annotated images side by side
    col1, col2 = st.beta_columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)

    # Display detection results
    st.subheader("Detection Results")
    st.json(results)