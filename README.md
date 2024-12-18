# Cat Detection, Segmentation, and Animal Classification

This repository demonstrates a comprehensive animal recognition system utilizing YOLOv11. The project showcases:

- **Cat Detection**: Identify and locate cats in images using bounding boxes.
- **Cat Segmentation**: Highlight cats in images using masks.
- **Animal Classification**: Classify images into one of the following animal classes:
  - Butterfly, Cats, Cow, Dogs, Elephant, Hen, Horse, Monkey, Panda, Sheep, Spider, Squirrel

---

## Features

1. **Cat Detection**:
   - Uses a YOLOv11 model trained on labeled cat datasets.
   - Outputs bounding boxes with confidence scores.

2. **Cat Segmentation**:
   - Leverages YOLOv11 segmentation capabilities.
   - Produces segmented images with cats highlighted.

3. **Animal Classification**:
   - Classifies images into one of 12 animal categories.
   - Includes detailed probabilities for each prediction.

---

## Dataset Details

The system was trained on a diverse dataset comprising:
- **[Cat Detection and Segmentation](https://www.kaggle.com/datasets/ikmalalfaozi/cat-detection-and-segmentation/data)**:   
  Custom-labeled images for segmentation tasks from Open Images Dataset consisting of 1000 train, 100 validation, 100 test.
- **[Animal Classification](https://www.kaggle.com/datasets/piyushkumar18/animal-image-classification-dataset)**:   
  12 animal classes with the following distribution:
  - Butterfly (1452 files)
  - Cats (1456 files)
  - Cow (1451 files)
  - Dogs (1456 files)
  - Elephant (1450 files)
  - Hen (1452 files)
  - Horse (1452 files)
  - Monkey (1452 files)
  - Panda (1201 files)
  - Sheep (1452 files)
  - Spider (1452 files)
  - Squirrel (1452 files)

---

## Training Results

The following table summarizes the model's performance:

| Task                   | Metric         | Value |
|------------------------|----------------|-------|
| Cat Detection          | mAP@50-95      | 79.6% |
| Cat Segmentation       | mAP@50-95      | 54.7% |
| Animal Classification  | Top-1 Accuracy | 96.9% |

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ikmalalfaozi/AnimalAI-YOLOv11-for-Cat-Segmentation-and-Multi-Animal-Classification.git
   cd AnimalAI-YOLOv11-for-Cat-Segmentation-and-Multi-Animal-Classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the Streamlit app:
```bash
streamlit run main.py
```

Navigate to the URL displayed in the terminal to interact with the application.

---

## Acknowledgements

- [YOLOv11](https://github.com/ultralytics/ultralytics): A state-of-the-art object detection framework.
- [OpenImages Dataset](https://storage.googleapis.com/openimages/web/index.html): Source for cat detection and segmentation data.
- [Kaggle Dataset](https://www.kaggle.com): Source for animal classification data

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

If you'd like to include additional information or refine any section, feel free to ask [ikmalalfaozi@gmail.com](mailto:ikmalalfaozi@gmail.com).