# Brain Tumor Volume Prediction

This project extends the nnU-Net framework from the BraTS 2021 competition winner to predict the whole tumor volume (in cm³) from multi-parametric MRI scans. We use a pre-trained nnU-Net segmentation model as a backbone and add a scalar regression head to estimate tumor volumes directly. The approach builds on the paper "Extending nn-U-Net for brain tumor segmentation" by Luu et al., incorporating modifications like a larger encoder, group normalization, and axial attention in the decoder for improved performance.

## Training Details

For training this model, we utilized the pre-trained weights from the nnU-Net model described in the paper by [Isensee et al. (2021)](https://www.nature.com/articles/s41592-020-01008-z). The nnU-Net framework is available on GitHub at [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet). If you want to download the model weights from the paper and run the code, you can find them in the same repository: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet). We trained our model on approximately 300 cases from the BraTS 2021 dataset, which can be downloaded from [https://kaggle.com/datasets/dschettler8845/brats-2021-task1](https://kaggle.com/datasets/dschettler8845/brats-2021-task1).

## Model Architecture

The model leverages a modified 3D U-Net architecture with an asymmetric encoder (doubled filters up to 512) for increased capacity, group normalization for better training stability with small batch sizes, and axial attention in the decoder to enhance feature integration across dimensions. A custom scalar head is attached to the backbone's output features to regress the normalized log-tumor volume.

## Results

The following table summarizes the performance metrics across train, validation, and test sets. Metrics include Mean Absolute Error (MAE) in cm³, Mean Absolute Percentage Error (MAPE), R-squared (R²) for explained variance, and Concordance Correlation Coefficient (CCC) for agreement between predictions and ground truth. These were evaluated after training with Huber loss and ensemble folding.

| Set   | MAE (cm³) | MAPE (%) | R²    | CCC   |
|-------|-----------|----------|-------|-------|
| Test  | [26.37]       | [68.14]      | [0.6340]   | [0.7065]   |

## Team Members
Zahra Mirzaei (zahraa.mirzaee.1999@gmail.com)
MahtaFetrat (http://77fetrat@gmail.com)
Negin Rahimi (http://neginrahimiyzd@gmail.com)
Maryam Borzo (http://m.borzoo1289@gmail.com)

## References
- Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211. [Link](https://www.nature.com/articles/s41592-020-01008-z)
- [Insert other references here, e.g., the base paper by Luu et al. and any others]

## Quick Inference
To compute the whole tumor volume for a single patient's MRI data and predict using the model:

```bash
python src/inference.py --patient_dir path/to/patient_dir --config config.yaml
```
