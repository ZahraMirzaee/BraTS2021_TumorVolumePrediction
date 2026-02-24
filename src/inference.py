# src/inference.py
# This module handles inference on test data or a single patient, computing volume and predictions.

import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import BratsScalarDataset, Sample
from model import NNUNetScalarHead
from train import evaluate_metrics, whole_tumor_volume_mm3, load_config
from nnunet.training.model_restore import load_model_and_checkpoint_files

def main():
    parser = argparse.ArgumentParser(description="Inference for brain tumor volume prediction.")
    parser.add_argument('--patient_dir', type=str, help='Path to single patient directory (optional)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    device = config['device']

    # Load model (example for one fold)
    trainer, _ = load_model_and_checkpoint_files(config['nnunet_fold_folder'], folds=[2], checkpoint_name="model_final_checkpoint")
    trainer.initialize(False)
    patch_size = tuple(trainer.plans['plans_per_stage'][trainer.stage]['patch_size'])
    num_input_channels = trainer.plans['num_modalities']

    trainer_fold, _ = load_model_and_checkpoint_files(config['nnunet_fold_folder'], folds=[2], checkpoint_name="model_final_checkpoint")
    net_fold = trainer_fold.network
    net_fold.do_ds = False
    net_fold.eval()
    net_fold.to(device)
    with torch.no_grad():
        dummy = torch.zeros((1, num_input_channels, *patch_size), device=device)
        out = net_fold(dummy)
        n_backbone_channels = out[0].shape[1] if isinstance(out, tuple) else out.shape[1]
    model = NNUNetScalarHead(net_fold, n_backbone_channels).to(device)
    model.load_state_dict(torch.load("../output/full_model0.pth"))  # Load best model
    model.eval()

    if args.patient_dir:
        # Single patient inference
        case_id = Path(args.patient_dir).name
        image_paths = [
            str(Path(args.patient_dir) / f"{case_id}_flair.nii.gz"),
            str(Path(args.patient_dir) / f"{case_id}_t1.nii.gz"),
            str(Path(args.patient_dir) / f"{case_id}_t1ce.nii.gz"),
            str(Path(args.patient_dir) / f"{case_id}_t2.nii.gz")
        ]
        seg_path = str(Path(args.patient_dir) / f"{case_id}_seg.nii.gz")
        y = whole_tumor_volume_mm3(seg_path) / 1000.0 if Path(seg_path).exists() else 0.0  # cm3
        sample = Sample(image_paths=image_paths, seg_path=seg_path, y=y)
        ds = BratsScalarDataset([sample], patch_size, config['mask_with_seg'])
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device)
                pred_norm = model(xb)
                pred = np.expm1(pred_norm.item() * 1.0 + 0.0)  # Adjust with actual y_mean/std if needed
                true_y = y
                print(f"True volume: {true_y:.2f} cm³, Predicted: {pred:.2f} cm³")
                metrics = evaluate_metrics([true_y], [pred])
                print(metrics)
    else:
        # Full test set (load from config)
        # Similar to train.py test loop
        print("Running on full test set...")

if __name__ == "__main__":
    main()