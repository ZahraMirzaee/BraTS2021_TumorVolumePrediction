# Brain Tumor Volume Prediction

This project extends the nnU-Net framework from the BraTS 2021 competition winner to predict the whole tumor volume (in cm³) from multi-parametric MRI scans. We use a pre-trained nnU-Net segmentation model as a backbone and add a scalar regression head to estimate tumor volumes directly. The approach builds on the paper "Extending nn-U-Net for brain tumor segmentation" by Luu et al., incorporating modifications like a larger encoder, group normalization, and axial attention in the decoder for improved performance.
You can download the trained model weights (.pth file) from this Google Drive link:  
[Download Model Weights](https://drive.google.com/file/d/1TH80Ak-JLEtRBAiZpYH-yUljEpkAVRN5/view?usp=sharing)

## Training Details

For training this model, we utilized the pre-trained weights from the nnU-Net model described in the paper by [Isensee et al. (2021)](https://www.nature.com/articles/s41592-020-01008-z). The nnU-Net framework is available on GitHub at [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet). If you want to download the model weights from the paper and run the code, you can find them in the same repository: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet). We trained our model on approximately 300 cases from the BraTS 2021 dataset, which can be downloaded from [https://kaggle.com/datasets/dschettler8845/brats-2021-task1](https://kaggle.com/datasets/dschettler8845/brats-2021-task1).

## Model Architecture

The model leverages a modified 3D U-Net architecture with an asymmetric encoder (doubled filters up to 512) for increased capacity, group normalization for better training stability with small batch sizes, and axial attention in the decoder to enhance feature integration across dimensions. A custom scalar head is attached to the backbone's output features to regress the normalized log-tumor volume.
<img width="624" height="446" alt="Model_Arch" src="https://github.com/user-attachments/assets/8318da5e-3125-4dc3-9dc9-b344e7ead859" />


## Results

The following table summarizes the performance metrics across the test set. Metrics include Mean Absolute Error (MAE) in cm³, Mean Absolute Percentage Error (MAPE), R-squared (R²) for explained variance, and Concordance Correlation Coefficient (CCC) for agreement between predictions and ground truth. These were evaluated after training with Huber loss .

| Set   | MAE (cm³) | MAPE (%) | R²    | CCC   |
|-------|-----------|----------|-------|-------|
| Test  | [8.88]       | [15.68]      | [0.9601]   | [0.9789]   |

## Team Members
- Zahra Mirzaei [zahraa.mirzaee.1999@gmail.com]
- MahtaFetrat [77fetrat@gmail.com]
- Negin Rahimi [neginrahimiyzd@gmail.com]
- Maryam Borzo [m.borzoo1289@gmail.com]
For any questions or collaboration, please contact us.

## References

[1] G. K. Reddy, Bh. A. Sahitya, and A. Ghosh, “Size Analysis of Brain Tumor from MRI Images Using MATLAB,” 2021.

[2] , O. Erogul, Z. Telatar, and , “Automatic Brain Tumor Detection and Volume Estimation in Multimodal MRI Scans via a Symmetry Analysis,” 2023.

[3] J. M. Cameron, E. Gray, P. Brennan, M. Baker, P. Hall, A. Lishman, P. Karunaratne, G. Tramonti, and M. Vallet, “Brain Tumor Diagnostic Interval and Tumor Size at Detection: Impact on Survival, Recurrence, Inpatient Length of Stay and Neurological Deficit,” 2025.

[4] E. Gray, J. Cameron, A. Lishman, P. Hall, P. Karunaratne, G. Tramonti, M. Vallet, L. Pike, M. Baker, and P. M. Brennan, “Does Earlier Diagnosis and Treatment of Brain Tumours Matter? Time-to-Treatment Intervals and Tumour Size at Detection; Impact on Survival, Recurrence, Inpatient Length of Stay and Neurological Deficit,” 2025.

[5] H. M. Luu and S.-H. Park, “Extending nn-UNet for brain tumor segmentation,” arXiv preprint arXiv:2112.04653, 2021. [Online]. Available: https://arxiv.org/abs/2112.04653

[6] U. Baid et al., “The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification,” arXiv preprint arXiv:2107.02314, 2021. [Online]. Available: https://arxiv.org/abs/2107.02314

[7] F. Isensee, P. F. Jaeger, S. A. A. Kohl, K. H. Petersen, and K. H. Maier‑Hein, “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature Methods, vol. 18, pp. 203–211, 2021, doi: 10.1038/s41592-020-01008-z. [Online]. Available: https://doi.org/10.1038/s41592-020-01008-z

[8] E. S. Biratu, F. Schwenker, Y. M. Ayano, and T. G. Debelee, “A Survey of Brain Tumor Segmentation and Classification Algorithms,” Journal of Imaging, vol. 7, no. 9, article 179, 2021. [Online]. Available: https://www.mdpi.com/2313-433X/7/9/179

[9] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” arXiv preprint arXiv:1505.04597, 2015. [Online]. Available: https://arxiv.org/abs/1505.04597

[10] Y. Xu, R. Quan, W. Xu, Y. Huang, X. Chen, and F. Liu, “Advances in Medical Image Segmentation: A Comprehensive Review of Traditional, Deep Learning and Hybrid Approaches,” Bioengineering, vol. 11, no. 10, article 1034, 2024. [Online]. Available: https://www.mdpi.com/2306-5354/11/10/1034

[11] D. Leung, X. Han, T. Mikkelsen, and L. B. Nabors, “Role of MRI in primary brain tumor evaluation,” Journal of the National Comprehensive Cancer Network, vol. 12, no. 11, pp. 1561–1568, Nov. 2014, doi: 10.6004/jnccn.2014.0156. [Online]. Available: https://pubmed.ncbi.nlm.nih.gov/25361803/

[12] M. Basthikodi, M. Chaithrashree, B. M. A. Shafeeq, and A. P. Gurpur, “Enhancing multiclass brain tumor diagnosis using SVM and innovative feature extraction techniques,” Scientific Reports, vol. 14, p. 26023, 2024, doi: 10.1038/s41598-024-26023-0.

[13] N.-H. Lu, Y.-H. Huang, K.-Y. Liu, and T.-B. Chen, “Deep learning-driven brain tumor classification and segmentation using non-contrast MRI,” Scientific Reports, vol. 15, p. 27831, 2025, doi: 10.1038/s41598-025-13591-2.

[14] S. Aresta et al., “Advancing Brain Tumor Diagnosis Using Deep Learning: A Systematic Review on Glioma Segmentation and Classification on Multiparametric MRI,” medRxiv, preprint 2026.01.13.26344038, 2026. [Online]. Available: https://www.medrxiv.org/content/early/2026/01/15/2026.01.13.26344038

[15] A. Yeafi, M. Islam, and M. S. U. Yusuf, “A deep learning framework for 3D brain tumor segmentation and survival prediction,” Healthcare Analytics, vol. 8, p. 100418, 2025, doi: 10.1016/j.health.2025.100418. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2772442525000371

[16] W.-J. Zhang, W.-T. Chen, C.-H. Liu, S.-W. Chen, Y.-H. Lai, and S. D. You, “Feasibility Study of Detecting and Segmenting Small Brain Tumors in a Small MRI Dataset with Self-Supervised Learning,” Diagnostics, vol. 15, no. 3, article 249, 2025. [Online]. Available: https://www.mdpi.com/2075-4418/15/3/249

[17] A. A. Pravitasari et al., “Gaussian Mixture Model for MRI Image Segmentation to Build a Three-Dimensional Image on Brain Tumor Area,” Journal of Imaging, vol. 6, no. 12, p. 149, 2020, doi: 10.3390/jimaging6120149.

[18] Y. Barhoumi, A. H. Fattah, N. Bouaynaya, F. Moron, J. Kim, H. M. Fathallah-Shaykh, R. A. Chahine, and H. Sotoudeh, “Robust AI-Driven Segmentation of Glioblastoma T1c and FLAIR MRI Series and the Low Variability of the MRIMath® Smart Manual Contouring Platform,” Cancers, vol. 17, no. 2, p. 1456, 2025, doi: 10.3390/cancers17021456.

[19] W.-J. Zhang et al., “Feasibility Study of Detecting and Segmenting Small Brain Tumors in a Small MRI Dataset with Self-Supervised Learning,” Journal of Imaging, vol. 11, no. 2, p. 45, 2025. doi: 10.3390/jimaging11020045.

[20] A. R. Kumar and H. Kuttiappan, “Detection of brain tumor size using modified deep learning and multilevel thresholding utilizing modified dragonfly optimization algorithm,” \textit{Concurrency and Computation: Practice and Experience}, vol. 34, no. 10, May 2022, Art. no. e7016, doi: 10.1002/cpe.7016.

## Quick Inference
To compute the whole tumor volume for a single patient's MRI data and predict using the model:

```bash
python src/inference.py --patient_dir path/to/patient_dir --config config.yaml
```
