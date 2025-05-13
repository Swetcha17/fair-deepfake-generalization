# Reducing Demographic Bias in Open-Set Deepfake Detection

This project investigates how racial demographic bias affects the performance of deepfake detection systems in open-set conditions using **softmax thresholding**. Real images are sourced from the FairFace dataset, and synthetic fake images are generated using **FaceSwap** via the [SimSwap](https://github.com/neuralchen/SimSwap) framework. We compare models trained on single-race and multi-race datasets, evaluating their ability to generalize to unseen demographic groups.

---

## Dataset Overview

### Real Face Images
- Sourced from [FairFace Dataset](https://github.com/joojs/fairface)
- Cropped and organized by race:  
  - **White**
  - **Black**
  - **Indian**
  - **East Asian**
  - **Latino Hispanic**
  - **Southeast Asian**
  - **Middle Eastern**
- Preprocessing included resizing, cropping, and organizing into PyTorch-compatible folder structures.

### Fake Face Images (SimSwap FaceSwap)
- Generated fake faces using the **SimSwap** framework.
- Each fake was created by swapping one real face with another of the **same race and gender**.
- The number of fake images per race:
  - **White**: 1,000  
  - **Black**: 200  
  - **Indian**: 200  
  - **East Asian**: 200  
  - **Latino Hispanic**: 200
  - **Southeast Asian**: 200
  - **Middle Eastern**: 200  

- Face swapping was done using a CSV mapping:
  ```
  source, target, output
  real1.jpg, real2.jpg, fake_output.jpg
  ```

  With the following command:
  ```bash
  python test_one_image.py --isTrain false \
    --name people \
    --Arc_path arcface_model/arcface_checkpoint.tar \
    --pic_a_path path/to/source.jpg \
    --pic_b_path path/to/target.jpg \
    --output_path path/to/output.jpg
  ```

---

## Models Trained

Each model was trained for both **single-race** and **multi-race** settings:

| Model Name     | Description                      |
|----------------|----------------------------------|
| SimpleCNN      | Custom 3-layer convolutional net |
| ResNet18       | Lightweight residual network     |
| ResNet50       | Deeper residual model            |
| DenseNet121    | Dense connectivity between layers|
| EfficientNetV2 | Scalable, efficient CNN          |
| ViT            | Vision Transformer               |

---

## Evaluation & Metrics

We evaluated all models using:

- **Softmax thresholding** for open-set detection
- **Accuracy**
- **AUROC**
- **F1-score**
- **Diversity Score** (accuracy std. dev. across races)

Each model was tested on all 5 race groups to measure generalization.

---

## Folder Structure

```
FairFace_By_Race/
├── White/
│   ├── real/
│   └── fake/
├── Black/
├── Indian/
├── East_Asian/
├── Latino_Hispanic/
Trained_Models/
Evaluation/
Plots/
```

---

## Running the Code

1. Clone and install:
   ```bash
   git clone https://github.com/Swetcha17/deepfake-bias-openset.git
   cd deepfake-bias-openset
   pip install -r requirements.txt
   ```

2. Prepare the dataset as shown above.

3. Train a model:
   ```bash
   python train_resnet18.py
   ```

4. Evaluate open-set detection:
   ```bash
   python evaluate_openset.py
   ```

5. View results:
   - Plots: `Plots/`
   - Evaluation tables: `Evaluation/`

---

## Author

**Swetcha Reddy Tukkani**  
M.S. in Computer Science 
Lehigh University  
