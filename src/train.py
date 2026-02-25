# This module handles data preparation, model loading, training loop, and saving results/logs/plots.

import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
import kagglehub
import tarfile
from pathlib import Path
import shutil
import nibabel as nib
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.dataset_conversion.utils import generate_dataset_json
from dataset import BratsScalarDataset, read_samples_from_csv, Sample
from model import NNUNetScalarHead, set_requires_grad
from torch.serialization import load as original_torch_load


def whole_tumor_volume_mm3(seg_path):
    img = nib.load(seg_path)
    seg = img.get_fdata()
    voxel_spacing = img.header.get_zooms()[:3]
    voxel_volume = np.prod(voxel_spacing)
    wt_mask = np.isin(seg, [1, 2, 4])
    volume_mm3 = np.sum(wt_mask) * voxel_volume
    return volume_mm3

def setup_environment(config):
    os.environ["nnUNet_raw_data_base"] = config['nnunet_raw_data_base']
    os.environ["nnUNet_preprocessed"] = config['nnunet_raw_data_base'] + "/nnUNet_preprocessed"
    os.environ["RESULTS_FOLDER"] = config['nnunet_raw_data_base'] + "/nnUNet_results"
    os.makedirs("./nnUNet_raw_data_base/nnUNet_raw_data/Task501_BraTS2021/imagesTr", exist_ok=True)
    os.makedirs("./nnUNet_raw_data_base/nnUNet_raw_data/Task501_BraTS2021/labelsTr", exist_ok=True)

def download_and_extract_data(config):
    path = kagglehub.dataset_download("dschettler8845/brats-2021-task1")
    brats_dir = config['brats_dir']
    for tar_file in os.listdir(path):
        if tar_file.endswith(".tar"):
            tar_path = os.path.join(path, tar_file)
            with tarfile.open(tar_path, 'r') as tar_ref:
                tar_ref.extractall(brats_dir)

def prepare_data_for_nnunet(config):
    BRATS_ROOT = Path(config['brats_root'])
    TASK_ROOT = Path(config['task_root'])
    imagesTr = TASK_ROOT / "imagesTr"
    labelsTr = TASK_ROOT / "labelsTr"
    MODALITY_MAP = config['modality_map']
    cases = sorted([d for d in BRATS_ROOT.iterdir() if d.is_dir()])
    for case_dir in tqdm(cases):
        case_id = case_dir.name
        for mod, suffix in MODALITY_MAP.items():
            src = case_dir / f"{case_id}_{mod}.nii.gz"
            dst = imagesTr / f"{case_id}_{suffix}.nii.gz"
            if src.exists():
                shutil.copy(src, dst)
        seg_src = case_dir / f"{case_id}_seg.nii.gz"
        seg_dst = labelsTr / f"{case_id}.nii.gz"
        if seg_src.exists():
            shutil.copy(seg_src, seg_dst)
    # Remap labels
    for seg_file in labelsTr.glob("*.nii.gz"):
        img = nib.load(seg_file)
        data = img.get_fdata().astype(np.uint8)
        data[data == 4] = 3
        new_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(new_img, seg_file)
    # Generate dataset JSON
    generate_dataset_json(
        str(TASK_ROOT / "dataset.json"),
        str(imagesTr),
        None,
        {0: "FLAIR", 1: "T1", 2: "T1ce", 3: "T2"},
        {0: "background", 1: "necrotic_non_enhancing", 2: "edema", 3: "enhancing_tumor"},
        "BraTS2021"
    )

def compute_volumes_and_save_csv(config):
    LABELS_DIR = Path(config['task_root']) / "labelsTr"
    records = []
    for seg_file in sorted(LABELS_DIR.glob("*.nii.gz")):
        case_id = seg_file.stem[:-4]  # to remove .nii.gz
        img = nib.load(seg_file)
        data = img.get_fdata()
        spacing = img.header.get_zooms()
        voxel_volume = np.prod(spacing)
        wt_voxels = np.sum(data > 0)
        wt_volume_mm3 = wt_voxels * voxel_volume
        wt_volume_cm3 = wt_volume_mm3 / 1000.0
        records.append({
	    	"case_id": case_id,
	    	"wt_voxels": int(wt_voxels),
	    	"wt_volume_mm3": float(wt_volume_mm3),
	    	"wt_volume_cm3": float(wt_volume_cm3)
	    })
    df = pd.DataFrame(records)
    df.to_csv(config['csv_path'], index=False)

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def mean_absolute_percentage_error(true_y, pred_y, epsilon=1e-6):
    true_y = np.array(true_y)
    pred_y = np.array(pred_y)
    return np.mean(np.abs((true_y - pred_y) / (true_y + epsilon))) * 100

def concordance_correlation_coefficient(true_y, pred_y):
    true_y = np.array(true_y)
    pred_y = np.array(pred_y)
    cor = np.corrcoef(true_y, pred_y)[0][1]
    mean_true = np.mean(true_y)
    mean_pred = np.mean(pred_y)
    var_true = np.var(true_y)
    var_pred = np.var(pred_y)
    sd_true = np.std(true_y)
    sd_pred = np.std(pred_y)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator if denominator != 0 else 0

def evaluate_metrics(true_y, pred_y):
    mae = mean_absolute_error(true_y, pred_y)
    mape = mean_absolute_percentage_error(true_y, pred_y)
    r2 = r2_score(true_y, pred_y)
    ccc = concordance_correlation_coefficient(true_y, pred_y)
    print(f"MAE: {mae:.4f} cm³")
    print(f"MAPE: {mape:.2f}%")
	print(f"R²: {r2:.4f}")
	print(f"CCC: {ccc:.4f}")
    return {'MAE': mae, 'MAPE': mape, 'R2': r2, 'CCC': ccc}

def plot_predictions(true_y, pred_y, title='Predicted vs Actual Volume', save_path="../output/plot_predictions.png"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=true_y, y=pred_y, alpha=0.7)
    plt.plot([min(true_y), max(true_y)], [min(true_y), max(true_y)], 'r--')
    plt.xlabel('True Volume (cm³)')
    plt.ylabel('Predicted Volume (cm³)')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_error_histogram(errors, title='Histogram of Prediction Errors', save_path="../output/plot_error_histogram.png"):
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=20, kde=True)
    plt.xlabel('Error (cm³)')
    plt.title(title)
    plt.savefig(save_path)
    plt.show()

def plot_bland_altman(true_y, pred_y, title='Bland-Altman Plot', save_path="../output/plot_bland_altman.png"):
    mean = (np.array(true_y) + np.array(pred_y)) / 2
    diff = np.array(true_y) - np.array(pred_y)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=mean, y=diff, alpha=0.7)
    plt.axhline(np.mean(diff), color='r', linestyle='--')  
    plt.axhline(np.mean(diff) + 1.96 * np.std(diff), color='g', linestyle='--')  
    plt.axhline(np.mean(diff) - 1.96 * np.std(diff), color='g', linestyle='--')  
    plt.xlabel('Mean Volume (cm³)')
    plt.ylabel('Difference (True - Pred)')
    plt.title(title)
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    config = load_config()
    setup_environment(config)
    download_and_extract_data(config)
    prepare_data_for_nnunet(config)
    compute_volumes_and_save_csv(config)

    samples = read_samples_from_csv(config['csv_path'], config['brats_root'], config['label_column'])
    random.seed(config['seed'])
    samples = random.sample(samples, int(len(samples) * config['subsample_fraction']))

    # Load nnU-Net

    def trusted_load(*args, **kwargs):
	    kwargs.setdefault('weights_only', False)
	    return original_torch_load(*args, **kwargs)

    _prev_load = torch.load
    torch.load = trusted_load

    try:
    	trainer, _ = load_model_and_checkpoint_files(
    		config['nnunet_fold_folder'],
    		folds=[2], #You Can use the fold : 0,1,2,3,4
    		checkpoint_name="model_final_checkpoint"
    	)
    	print("Load Model Successfully!")

    except Exception as e:
    	print("Load Error", e)
    	raise
    finally:
    	torch.load = _prev_load

    trainer.initialize(False)
    patch_size = tuple(trainer.plans['plans_per_stage'][trainer.stage]['patch_size'])
    num_input_channels = trainer.plans['num_modalities']

    device = config['device']
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
    if not config['train_backbone']:
        set_requires_grad(model.backbone, False)
    if config['freeze_seg_outputs'] and hasattr(model.backbone, "seg_outputs"):
        set_requires_grad(model.backbone.seg_outputs, False)
    set_requires_grad(model.head, True)

    # Data splits
    y_list = [s.y for s in samples]
    bins = pd.qcut(y_list, 4, labels=False, duplicates='drop')
    idx = np.arange(len(samples))
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, stratify=bins, random_state=config['seed'])
    temp_bins = [bins[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_bins, random_state=config['seed'])

    tr_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    tr_y_log = [np.log1p(s.y) for s in tr_samples]
    y_mean_log = np.mean(tr_y_log)
    y_std_log = np.std(tr_y_log)

    ds_tr = BratsScalarDataset(tr_samples, patch_size, config['mask_with_seg'], y_mean=y_mean_log, y_std=y_std_log)
    ds_val = BratsScalarDataset(val_samples, patch_size, config['mask_with_seg'], y_mean=y_mean_log, y_std=y_std_log)
    ds_test = BratsScalarDataset(test_samples, patch_size, config['mask_with_seg'], y_mean=y_mean_log, y_std=y_std_log)

    dl_tr = DataLoader(ds_tr, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    optim = torch.optim.AdamW(filter(lambda param: param.requires_grad, model.parameters()),
                              lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5)
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val_mae = float("inf")
    with open("../output/log.txt", "w") as log_file:
        for epoch in range(config['epochs']):
            
            model.train()
            if not config['train_backbone']:
                model.backbone.eval()
            tr_losses = []
            for xb, yb in tqdm(dl_tr, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
                xb, yb = xb.to(device), yb.to(device)
                optim.zero_grad()
                pred_norm = model(xb)
                loss = loss_fn(pred_norm, yb)
                loss.backward()
                optim.step()
                tr_losses.append(loss.item())
            
            model.eval()
            val_losses = []
            val_true = []
            val_pred = []
            with torch.no_grad():
                for xb, yb in tqdm(dl_val, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]"):
                    xb, yb = xb.to(device), yb.to(device)
                    pred_norm = model(xb)
                    pred = np.expm1(pred_norm.item() * y_std_log + y_mean_log)
                    true_y = np.expm1(yb.item() * y_std_log + y_mean_log)
                    val_true.append(true_y)
                    val_pred.append(pred)
                    loss = loss_fn(pred_norm, yb)
                    val_losses.append(loss.item())

            tr_loss = np.mean(tr_losses)
            val_loss = np.mean(val_losses)
            metrics = evaluate_metrics(val_true, val_pred)
            val_mae = mean_absolute_error(val_true, val_pred)
            scheduler.step(val_mae)
            log_msg = f"epoch={epoch+1:03d} train_loss={tr_loss:.6f} val_loss={val_loss:.6f}\n"
            print(log_msg)
            log_file.write(log_msg)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.head.state_dict(), f"../output/best_scalar_head_fold.pth")
                torch.save(model.state_dict(), f"../output/full_model.pth")

    # Plots
    errors = np.array(val_true) - np.array(val_pred)
    plot_predictions(val_true, val_pred, save_path="../output/plot_val_predictions.png")
    plot_predictions(val_true, val_pred, title='Test Set: Predicted vs Actual')
    plot_error_histogram(errors, title='Test Set: Error Histogram')
    plot_bland_altman(val_true, val_pred, title='Test Set: Bland-Altman')
