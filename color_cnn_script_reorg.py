#!/usr/bin/env python3
import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--layer",
    type=int,
    default=None,
    help="Which vgg16.features index to process"
)
args = parser.parse_args()

if args.layer is None:
    raise ValueError("You must pass --layer when running this script under Slurm.")
# ───────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────
TARGET_LAYER = args.layer
BASE_DIR       = "./tiny-imagenet-200/val/images_1000"
OUTPUT_DIR     = "./results"
TOP_K          = 250        # number of top patches per neuron
BATCH_SIZE     = 16
IMAGE_SIZE     = (224, 224)
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ───────────────────────────────────────────────────────────


# ───────────────────────────────────────────────────────────
# Dataset
# ───────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

class TinyImageNetValDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted(
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir) if f.endswith(".JPEG")
        )
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.loader(self.image_paths[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.image_paths[idx]


# ───────────────────────────────────────────────────────────
# Activation capture + patch extraction
# ───────────────────────────────────────────────────────────
def get_activation_hook(name, store):
    def hook(_, __, output):
        store[name] = output.detach().cpu()
    return hook

def generate_activation_list(model, layer_name, dataset, batch_size, device):
    """Returns (activations, image_paths) for every image."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"Layer '{layer_name}' not found.")
    activations = {}
    handle = modules[layer_name].register_forward_hook(
        get_activation_hook(layer_name, activations)
    )

    model.to(device).eval()
    activation_list, paths = [], []
    with torch.no_grad():
        for imgs, pths in loader:
            imgs = imgs.to(device)
            _ = model(imgs)
            batch_act = activations[layer_name]  # [B,C,H,W]
            activation_list.extend(batch_act)
            paths.extend(pths)

    handle.remove()
    return activation_list, paths

def get_top_patches(activation_list, image_paths, top_k):
    """Return dict: neuron_idx → list of patch Tensors."""
    C = activation_list[0].shape[0]
    patches = {}
    for n in range(C):
        vals = [(act[n].max().item(), i) for i, act in enumerate(activation_list)]
        top_idxs = sorted(vals, key=lambda x: x[0], reverse=True)[:top_k]
        patch_list = []
        for _, img_i in top_idxs:
            img = Image.open(image_paths[img_i]).convert('RGB')
            t   = transform(img)
            amap = activation_list[img_i][n]
            y,x = np.unravel_index(torch.argmax(amap).item(), amap.shape)
            # map back to pixel coords
            scale = IMAGE_SIZE[0] / amap.shape[0]
            x1, y1 = int(x*scale), int(y*scale)
            x2, y2 = min(x1+int(scale), IMAGE_SIZE[0]), min(y1+int(scale), IMAGE_SIZE[1])
            patch = TF.crop(t, y1, x1, y2-y1, x2-x1)
            patch_list.append(patch)
        patches[n] = patch_list
    return patches


# ───────────────────────────────────────────────────────────
# Neuron-feature & CSI
# ───────────────────────────────────────────────────────────
def compute_neuron_features(patches, activation_list, top_k):
    nf = {}
    for n, p_list in patches.items():
        if not p_list:
            continue
        T = torch.stack([p if isinstance(p, torch.Tensor) else ToTensor()(p)
                         for p in p_list])  # [K,C,H,W]
        acts = torch.tensor([activation_list[i][n].max().item() for i in range(top_k)])
        wts  = acts / (acts.sum() + 1e-8)
        nf[n] = (T * wts[:,None,None,None]).sum(0)  # [C,H,W]
    return nf

def rgb2opp(img_np):
    R,G,B = img_np[...,0], img_np[...,1], img_np[...,2]
    O1 = (R-G)/np.sqrt(2)
    O2 = (R+G-2*B)/np.sqrt(6)
    O3 = (R+G+B)/np.sqrt(3)
    return np.stack([O1,O2,O3],-1)

def compute_csi(neuron_idx, activation_list, image_paths, top_k, eps=1e-8):
    rgb, gray = [], []
    for i, act in enumerate(activation_list):
        rgb.append(act[neuron_idx].max().item())
        img_np = (np.array(Image.open(image_paths[i]).resize(IMAGE_SIZE))
                  .astype(np.float32)/255.0)
        opp   = rgb2opp(img_np)
        gray.append(np.max(np.abs(opp)))
    rgb = np.array(rgb); gray = np.array(gray)
    m = rgb.max()
    if m < eps: 
        return 0.0
    nr = rgb / (m+eps); ng = gray / (m+eps)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(nr>eps, ng/nr, 0.0)
    return float(np.clip(1-r,0,1).mean())


# ───────────────────────────────────────────────────────────
# I/O helpers
# ───────────────────────────────────────────────────────────
def save_features(nf_dict, out_dir, max_save=100):
    os.makedirs(out_dir, exist_ok=True)
    for i,(n,tensor) in enumerate(nf_dict.items()):
        if i>=max_save: break
        img = ToPILImage()(tensor.clamp(0,1).cpu())
        img.save(os.path.join(out_dir, f"neuron_{n}.png"))
        gc.collect()

def save_csv(records, path):
    pd.DataFrame(records).to_csv(path, index=False)


# assume all the helper functions (TinyImageNetValDataset, transform,
# generate_activation_list, get_top_patches, compute_neuron_features,
# compute_csi, save_features, save_csv) are already defined as above

def process_one_folder(
    image_folder: str,
    layer: int,
    output_dir: str,
    k: int = 128,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu")
):
    os.makedirs(output_dir, exist_ok=True)

    # 1) dataset & model
    ds    = TinyImageNetValDataset(image_folder, transform=transform)
    model = models.vgg16(pretrained=True).eval().to(device)
    layer_str = f"features.{layer}"

    # 2) collect activations
    acts, paths = generate_activation_list(model, layer_str, ds, batch_size, device)
    C = acts[0].shape[0]
    print(f"[INFO] Folder '{os.path.basename(image_folder)}' → {C} channels at layer {layer_str}")

    # 3) get patches & neuron features
    patches = get_top_patches(acts, paths, k)
    nf      = compute_neuron_features(patches, acts, k)

    # 4) compute metrics per neuron
    records = []
    for n in range(C):
        if n not in nf:
            continue
        max_act = max(a[n].max().item() for a in acts)
        csi_val = compute_csi(n, acts, paths, k)
        records.append({
            "neuron_idx":     n,
            "max_activation": max_act,
            "CSI":            csi_val
        })

    # 5) save outputs
    #   a) neuron-feature images
    save_features(nf, output_dir, max_save=k)
    #   b) metrics CSV
    csv_path = os.path.join(output_dir, f"metrics_layer{layer}.csv")
    save_csv(records, csv_path)

    print(f"[DONE] Results in {output_dir} (images + {csv_path})")

if __name__ == "__main__":
    folder = "./tiny-imagenet-200/val/images_1000"  # one folder
    process_one_folder(
        image_folder=folder,
        layer=TARGET_LAYER,
        k=TOP_K,
        batch_size=16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        output_dir=f"./results/image_1000/layer_{TARGET_LAYER}"
    )


