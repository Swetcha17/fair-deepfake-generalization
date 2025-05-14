# Reducing Demographic Bias in Open-Set Deepfake Detection

This project explores the effect of racial demographic bias on the generalization of deepfake detection models under open-set conditions. Using **softmax thresholding**, we evaluate several CNN and transformer models trained on real faces from the **FairFace** dataset and fake images generated via **SimSwap** (FaceSwap). Both single-race and multi-race training setups are compared.

## Dataset Overview

### Real Images
- Sourced from [FairFace](https://github.com/joojs/fairface)
- Organized by race:
  - White, Black, Indian, East Asian, Latino Hispanic, Southeast Asian, Middle Eastern
- Preprocessed to align/crop and organized into PyTorch-compatible folders.

### Fake Images

- Generated using [SimSwap](https://github.com/neuralchen/SimSwap)
- **Intra-race, same-gender** face swaps
- Sample command used:
  ```bash
  python test_one_image.py --isTrain false \
    --name people \
    --Arc_path arcface_model/arcface_checkpoint.tar \
    --pic_a_path path/to/source.jpg \
    --pic_b_path path/to/target.jpg \
    --output_path path/to/output.jpg
  ```

| Race             | # Fake Images |
|------------------|---------------|
| White            | 1000          |
| Black            | 200           |
| Indian           | 200           |
| East Asian       | 200           |
| Latino Hispanic  | 200           |
| Southeast Asian  | 200           |

## Trained Models

Each model was trained in two settings:  
`*_Single.ipynb` → trained on a single race  
`*_Multi.ipynb` → trained on all races combined

| Model           | Notebook                          |
|----------------|------------------------------------|
| SimpleCNN       | `Simple_CNN_Single.ipynb`, `Simple_CNN_Multi.ipynb` |
| ResNet18        | `ResNet18_Single.ipynb`, `ResNet18_Multi.ipynb`     |
| ResNet50        | `ResNet50_Single.ipynb`, `ResNet50_Multi.ipynb`     |
| DenseNet121     | `DenseNet121_Single.ipynb`, `DenseNet121_Multi.ipynb` |
| EfficientNetV2  | `EfficientNetV2_Single.ipynb`, `EfficientNetV2_Multi.ipynb` |
| Vision Transformer (ViT) | `ViT_Single.ipynb`, `ViT_Multi.ipynb`       |

## Evaluation Metrics

- Softmax thresholding for open-set recognition
- Accuracy
- AUROC (Area Under ROC)
- F1-score
- Diversity Score (accuracy std. dev. across race groups)
- t-SNE Visualizations

## Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Swetcha17/fair-deepfake-generalization.git
   cd fair-deepfake-generalization
   ```
2. Run the Jupyter Notebook files

3. View results:
   - t-SNE plots → `Plots/`
   - Evaluation tables → `Evaluation/open_set_results.csv`

## Author
**Swetcha Reddy Tukkani**  
M.S. in Computer Science  
Lehigh University
