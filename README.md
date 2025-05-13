# Satellite Imagery Semantic Segmentation – README

This repository provides a comprehensive pipeline for semantic segmentation of high-resolution satellite imagery of Dubai, utilizing U-Net architectures. The dataset, sourced from MBRSC satellites, is annotated with pixel-wise labels across six distinct classes. ([GitHub][1])

---

## 📂 Dataset Overview

* **Source**: [MBRSC Satellite Imagery Dataset](https://www.kaggle.com/datasets/muhammadyasirsaleem/satellite-imagery)
* **Content**: 54 high-resolution images organized into 6 large tiles (with some references indicating 8 tiles), each accompanied by corresponding segmentation masks.
* **Classes**:

  * Building: `#3C1098`
  * Land (Unpaved Area): `#8429F6`
  * Road: `#6EC1E4`
  * Vegetation: `#FEDD3A`
  * Water: `#E2A929`
  * Unlabeled: `#9B9B9B`

---

## 🛠️ Preprocessing Pipeline

1. **Patch Extraction**: Images and masks are divided into 256x256 patches using the `patchify` library, ensuring no overlap.
2. **Normalization**: Pixel values of image patches are scaled between 0 and 1 using `MinMaxScaler`.
3. **Mask Conversion**:

   * Masks are read using OpenCV and converted from BGR to RGB format.
   * A custom `rgb_to_2D_label()` function maps RGB values to integer class labels, producing 2D label arrays.
4. **Dataset Splitting**: The dataset is split into training and validation sets using an 80-20 ratio.
5. **Visualization**: Random samples of image patches and their corresponding masks are displayed using Matplotlib for verification.

---

## 🧠 Model Architectures

### Experiment 1: Standard U-Net with Dual Loss Functions

* **Architecture**: Custom U-Net model.
* **Loss Functions**:

  * Dice Loss with class weights to address class imbalance.
  * Categorical Focal Loss to focus on hard-to-classify pixels.
* **Optimizer**: Adam.
* **Metrics**: Accuracy and a custom Jaccard Coefficient.([Medium][2])

### Experiment 2: U-Net with Partial Focal Cross Entropy Loss

* **Architecture**: Standard U-Net model.
* **Custom Loss Function**: Partial Focal Cross Entropy Loss, which:

  * Applies focal loss alongside cross-entropy loss.
  * Ignores unlabeled pixels during loss computation.
  * Focuses on challenging pixels by adjusting loss intensity.
* **Optimizer**: Adam with a learning rate of 1e-4.
* **Callbacks**:

  * ModelCheckpoint to save the best model based on validation loss.
  * EarlyStopping to halt training if validation loss doesn't improve for 10 consecutive epochs.([GitHub][3])

---

## 🏋️ Training Details

* **Epochs**: 100 (with early stopping in Experiment 2).
* **Batch Size**: 16.
* **Data**: Training on `X_train` and `y_train`; validation on `X_val` and `y_val`.
* **Monitoring**: Training and validation loss and accuracy are plotted over epochs for performance assessment.

---

## 📊 Evaluation Metrics

* **Accuracy**: Proportion of correctly predicted pixels.
* **Mean Intersection over Union (Mean IoU)**: Evaluates the overlap between predicted and ground truth segments across all classes.
* **Visualization**: Random test images are displayed alongside their ground truth and predicted masks for qualitative assessment.

---

## 📈 Results Summary

### Experiment 1:

* **Final Training Accuracy**: 56.34%
* **Final Validation Accuracy**: 49.85%
* **Jaccard Coefficient Improvement**: From 0.1340 to 0.1596
* **Observations**: The model showed significant training improvement but limited generalization to validation data, indicating potential overfitting.([ResearchGate][4])

### Experiment 2:

* **Training Duration**: 42 epochs (early stopping activated).
* **Final Validation Accuracy**: 39.39%
* **Final Validation Loss**: 0.7439
* **Mean IoU**: 0.085
* **Observations**: While the model demonstrated learning capability, the low Mean IoU suggests challenges in accurately segmenting certain classes, possibly due to class imbalance or label noise.

---

## 🔍 Conclusion & Future Work

Both experiments highlight the challenges in semantic segmentation of satellite imagery, especially concerning class imbalance and generalization. Future improvements could include:

* Implementing advanced architectures like Attention U-Net or ResU-Net.
* Applying data augmentation techniques to enhance model robustness.
* Exploring different loss functions or class weighting strategies to address class imbalance. ([ResearchGate][4])

Lastly, the IoU and results percentage might be low. But the segmented image generated from the model is close to the one present in ground truth image. 
---

## 📎 References

* [MBRSC Satellite Imagery Dataset on Kaggle](https://www.kaggle.com/datasets/muhammadyasirsaleem/satellite-imagery)
* [Semantic Segmentation of Aerial Imagery Using U-Net with Self-Attention Mechanisms](https://www.mdpi.com/2076-3417/14/9/3712)([MDPI][5])

---

For any queries or contributions, please open an issue or submit a pull request.

---

[1]: https://github.com/ayushdabra/dubai-satellite-imagery-segmentation?utm_source=chatgpt.com "ayushdabra/dubai-satellite-imagery-segmentation - GitHub"
[2]: https://ueberf.medium.com/creating-a-dataset-of-satellite-images-for-stylegan-training-8eff8fd56e68?utm_source=chatgpt.com "Creating a dataset of satellite images for StyleGAN training"
[3]: https://github.com/ad-1/u-net-aerial-imagery-segmentation?utm_source=chatgpt.com "ad-1/u-net-aerial-imagery-segmentation - GitHub"
[4]: https://www.researchgate.net/publication/390823616_Semantic_Segmentation_of_Satellite_Images_using_2_various_U-Net_Architectures_A_Comparison_Study?utm_source=chatgpt.com "Semantic Segmentation of Satellite Images using 2 various U-Net ..."
[5]: https://www.mdpi.com/2076-3417/14/9/3712?utm_source=chatgpt.com "Semantic Segmentation of Aerial Imagery Using U-Net with Self ..."
