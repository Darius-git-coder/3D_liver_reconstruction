# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 07:55:48 2026
Filename: preprocessing_data_pipeline.py
@author: Darius
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom, affine_transform
from tqdm import tqdm
import gc

# --- DEINE KONSTANTEN ---
THRESH = 0.05
DIM = 64  # Ziel-Dimension für das U-Net

# ==========================================
# MODULARE FUNKTIONEN 
# ==========================================

def load_nifti_volume(file_path):
    """Lädt NIfTI und bringt es in Standard-Orientierung (RAS)."""
    nii = nib.load(file_path)
    nii = nib.as_closest_canonical(nii)
    volume = nii.get_fdata().astype(np.float32)
    spacing = np.array(nii.header.get_zooms()[:3])
    return volume, spacing, nii.affine, nii.header

def normalize_ct_volume(volume, mask=None, min_hu=-100, max_hu=400):
    """Windowing und Min-Max Normalisierung auf [0, 1]."""
    img = np.clip(volume, min_hu, max_hu)
    img = (img - min_hu) / (max_hu - min_hu)
    if mask is not None:
        img[mask == 0] = 0
    return img.astype(np.float32)

def pca_align_volume(volume, threshold=0.1):
    """Richtet das Volumen an den Hauptachsen aus."""
    coords = np.argwhere(volume > threshold)
    if len(coords) < 10: return volume
    center = coords.mean(axis=0)
    coords_centered = coords - center
    cov = np.cov(coords_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]
    if np.linalg.det(R) < 0: R[:, -1] *= -1
    offset = center - R @ center
    return affine_transform(volume, R, offset=offset, order=1, mode='constant', cval=0.0)

def crop_and_pad_volume(volume, threshold=0.1, margin=5, padding=10):
    """Schneidet auf Objekt zu und fügt Sicherheitsrand hinzu."""
    mask = volume > threshold
    coords = np.argwhere(mask)
    if coords.size == 0: return volume
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    
    z_min, z_max = max(0, min_c[0]-margin), min(volume.shape[0], max_c[0]+1+margin)
    y_min, y_max = max(0, min_c[1]-margin), min(volume.shape[1], max_c[1]+1+margin)
    x_min, x_max = max(0, min_c[2]-margin), min(volume.shape[2], max_c[2]+1+margin)

    cropped = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    if padding > 0:
        cropped = np.pad(cropped, pad_width=padding, mode='constant', constant_values=0)
    cropped[cropped < threshold] = 0
    return cropped

def resample_isometric(img, target_dim=256):
    """Skaliert isometrisch und bettet in Zielwürfel ein."""
    target_shape = (target_dim, target_dim, target_dim)
    scale_factor = min(t / s for t, s in zip(target_shape, img.shape))
    img_rescaled = zoom(img, scale_factor, order=1)
    
    new_img = np.zeros(target_shape, dtype=img.dtype)
    offsets = [(t - s) // 2 for t, s in zip(target_shape, img_rescaled.shape)]
    new_img[offsets[0]:offsets[0] + img_rescaled.shape[0],
            offsets[1]:offsets[1] + img_rescaled.shape[1],
            offsets[2]:offsets[2] + img_rescaled.shape[2]] = img_rescaled
    
    params = {'scale_factor': scale_factor, 'offsets': offsets, 'orig_crop_shape': img.shape}
    return new_img, params

# ==========================================
# BATCH PROCESSING PIPELINE
# ==========================================

def process_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = list(input_path.glob("*.nii.gz"))
    
    # tqdm umschließt die Liste 'files'
    # unit="img" zeigt an, dass wir Bilder zählen
    progress_bar = tqdm(files, desc="Preprocessing Liver Data", unit="img")

    for f in progress_bar:
        # Den aktuellen Dateinamen links im Balken anzeigen
        progress_bar.set_postfix(file=f.name[:15]) 
        
        try:
            # 1. Laden (Speicherschonend mit 32-bit Float)
            nii = nib.load(f)
            nii = nib.as_closest_canonical(nii)
            img = np.asanyarray(nii.dataobj).astype(np.float32)
            
            # 2. Normalisieren
            img = normalize_ct_volume(img, mask=(img > 0))

            # 3. PCA Alignment (mit Sicherheits-Padding)
            img = np.pad(img, pad_width=50, mode='constant', constant_values=0)
            img = pca_align_volume(img, threshold=THRESH)

            # 4. Crop & Pad
            img = crop_and_pad_volume(img, threshold=THRESH, margin=5, padding=10)

            # 5. Resize auf Zielgröße
            final_img, params = resample_isometric(img, target_dim=DIM)

            # 6. Speichern
            new_nii = nib.Nifti1Image(final_img, np.eye(4))
            nib.save(new_nii, str(output_path / f.name))
            np.save(output_path / f.name.replace(".nii.gz", "_params.npy"), params)

            # --- RAM-REINIGUNG ---
            # Wir löschen die großen Arrays explizit aus dem Speicher
            del img, final_img, nii, new_nii
            gc.collect()

        except Exception as e:
            # tqdm.write verhindert, dass Fehlermeldungen den Balken zerschießen
            tqdm.write(f"\nFehler bei {f.name}: {e}")

    print("\n Fertig! Alle Bilder wurden verarbeitet.")

# ==========================================
# START
# ==========================================
if __name__ == "__main__":
    SOURCE = r"C:\Users\dariu\OneDrive\Dokumente\Uni\Bachelor_Arbeit\Python\data\Task03_Liver\masked_liver_images" 
    TARGET = r"E:\Bachelorarbeit_Daten\pipeline_preprocessed_data"
    
    if not os.path.exists(SOURCE):
        print(f"PFAD NICHT GEFUNDEN: {SOURCE}")
    else:
        print(f"Pfad existiert. Inhalt: {os.listdir(SOURCE)[:5]}") # Zeigt die ersten 5 Dateien
        
    process_dataset(SOURCE, TARGET)