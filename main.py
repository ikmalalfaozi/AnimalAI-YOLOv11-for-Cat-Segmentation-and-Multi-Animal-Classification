import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_image_comparison import image_comparison

# Load YOLOv11 model
detection_model = YOLO("./models/cat-detect.pt")
segmentation_model = YOLO("./models/cat-seg.pt")
classification_model = YOLO("./models/animal-cls.pt")


# Function to load and preprocess image
def load_image(image_file):
    image = Image.open(image_file)
    return np.array(image)


# Function for cat detection using YOLOv11
def detect_cats(image):
    results = detection_model.predict(image, save=False, conf=0.25)
    bboxes = []
    labels = []
    confidences = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        bboxes.append([x1, y1, x2, y2])
        labels.append('Cat')
        confidences.append(result.conf[0])
    return bboxes, labels, confidences


# Function for segmentation using YOLOv11
def segment_cats(image):
    results = segmentation_model(image)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for result in results:
        boxes = result.boxes.numpy()
        masks = result.masks.data.numpy()
        for box, mask in zip(boxes, masks):
            r = box.xyxy[0].astype(int)
            # Resize mask to match the original image size if necessary
            resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Apply the mask to the image
            img[resized_mask > 0.5] = [0, 255, 0]
            # Draw bounding box
            img = cv2.rectangle(img, tuple(r[:2]), tuple(r[2:]), (255, 0, 0), 8)

    return img


# Function for animal classification using YOLOv11
def classify_animals(image):
    results = classification_model.predict(image, save=False)
    classes = [name for name in results[0].names.values()]
    confidences = [prob for prob in results[0].probs.data]
    return classes, confidences


# Streamlit App
def main():
    st.title("Cat Detection, Segmentation, and Animal Classification")

    menu = ["Home", "Cat Detection", "Cat Segmentation", "Animal Classification"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Welcome to the Cat and Animal Recognition System")
        st.write("This app demonstrates cat detection, segmentation, and animal classification using YOLOv11.")

    elif choice == "Cat Detection":
        st.subheader("Cat Detection")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            bboxes, labels, confidences = detect_cats(image)
            for bbox, label, confidence in zip(bboxes, labels, confidences):
                x1, y1, x2, y2 = bbox
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 8)
                image = cv2.putText(image, f"{label} ({confidence * 100:.1f}%)", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 8)
            st.image(image, caption="Detection Result", use_container_width=True)

    elif choice == "Cat Segmentation":
        st.subheader("Cat Segmentation")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            segmented_image = segment_cats(image)
            image_comparison(
                img1=image,
                img2=segmented_image,
                label1="Original Image",
                label2="Segmented Image",
                width=700,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )

    elif choice == "Animal Classification":
        st.subheader("Animal Classification")
        animal_classes = [
            "Butterfly",
            "Cats",
            "Cow",
            "Dogs",
            "Elephant",
            "Hen",
            "Horse",
            "Monkey",
            "Panda",
            "Sheep",
            "Spider",
            "Squirrel"
        ]
        st.write(", ".join(animal_classes))

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            classes, probabilities = classify_animals(image)
            for cls, prob in zip(classes, probabilities):
                st.write(f"{cls}: {prob * 100:.1f}%")


if __name__ == "__main__":
    main()
