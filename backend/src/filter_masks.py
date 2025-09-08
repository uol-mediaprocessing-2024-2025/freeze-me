#!/usr/bin/env python3
"""
Entfernt kleine Pixelflecke im transparenten Bereich von PNGs mittels
morphologischer Filter auf der Alpha-Ebene.

Vorgehen:
1) Binärmaske aus Alpha (sichtbar = alpha >= threshold).
2) Morphologische Öffnung entfernt kleine "weiße" Inseln.
3) Entfernte Inseln werden identifiziert (Original - Geöffnet) und
   deren Alpha auf 0 gesetzt (optional auch RGB auf 0).
4) Optional: Komponenten kleiner als min_area per Connected Components entfernen.

Nutzung:
    python clean_png_transparent_noise.py /pfad/zum/ordner \
        --out out --alpha-threshold 10 --kernel 3 --iterations 1 --min-area 0

Abhängigkeiten:
    pip install opencv-python numpy
"""

import argparse
import glob
import os
from typing import Tuple

import cv2
import numpy as np


def clean_image_alpha_based(
    img_bgra: np.ndarray,
    alpha_threshold: int = 10,
    kernel_size: int = 3,
    iterations: int = 1,
    min_area: int = 0,
    zero_rgb_on_remove: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Wendet morphologische Öffnung auf der Sichtbarkeitsmaske (Alpha) an und
    entfernt kleine Inseln. Optional entfernt zusätzlich Komponenten < min_area.

    Returns:
        (geändertes Bild (BGRA), Anzahl entfernte Pixel)
    """
    if img_bgra.ndim != 3 or img_bgra.shape[2] < 4:
        # Kein Alpha -> unverändert zurück
        return img_bgra, 0

    b, g, r, a = cv2.split(img_bgra)
    # Sichtbarkeitsmaske (True = sichtbar)
    visible = a >= alpha_threshold  # bool, Form HxW

    # Morphologische Öffnung (auf 0/255-Maske)
    mask_u8 = (visible.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # Alles, was in der Öffnung „weggefallen“ ist, werten wir als Fleck
    opened_bool = opened > 0
    specks_open = visible & (~opened_bool)

    # Optional: Komponenten kleiner als min_area löschen
    small_comps = np.zeros_like(visible, dtype=bool)
    if min_area > 0:
        # Connected Components auf Original-Visible-Maske
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            visible.astype(np.uint8), connectivity=8
        )
        # Label 0 ist Hintergrund
        for lab in range(1, num_labels):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area:
                small_comps |= (labels == lab)

    # Gesamt-Entfernungsmaske
    remove_mask = specks_open | small_comps
    removed_pixels = int(remove_mask.sum())

    if removed_pixels > 0:
        a = a.copy()
        a[remove_mask] = 0
        if zero_rgb_on_remove:
            b = b.copy(); g = g.copy(); r = r.copy()
            b[remove_mask] = 0
            g[remove_mask] = 0
            r[remove_mask] = 0

    cleaned = cv2.merge([b, g, r, a])
    return cleaned, removed_pixels


def main():
    ap = argparse.ArgumentParser(description="PNG-Flecken im transparenten Bereich entfernen (morphologische Filter).")
    ap.add_argument("indir", help="Eingabeordner mit .png-Dateien")
    ap.add_argument("--out", default="cleaned", help="Ausgabeunterordner (wird im Eingabeordner erstellt, default: cleaned)")
    ap.add_argument("--alpha-threshold", type=int, default=10,
                    help="Alpha-Schwelle (0..255), ab der ein Pixel als 'sichtbar' gilt (default: 10)")
    ap.add_argument("--kernel", type=int, default=3, help="Kernelgröße für die morphologische Öffnung (odd, default: 3)")
    ap.add_argument("--iterations", type=int, default=1, help="Iterationen für die Öffnung (default: 1)")
    ap.add_argument("--min-area", type=int, default=0,
                    help="Optionale Minimalfläche: Komponenten < min_area werden entfernt (0 zum Deaktivieren, default: 0)")
    ap.add_argument("--keep-rgb", action="store_true",
                    help="RGB-Werte der entfernten Pixel NICHT auf 0 setzen (nur Alpha wird 0).")
    args = ap.parse_args()

    indir = os.path.abspath(args.indir)
    outdir = os.path.join(indir, args.out)
    os.makedirs(outdir, exist_ok=True)

    print(indir)
    paths = sorted(glob.glob(os.path.join(indir, "*.png")))
    if not paths:
        print("Keine .png-Dateien gefunden.")
        return

    total_removed = 0
    print(f"Finde {len(paths)} PNGs in: {indir}")
    for i, p in enumerate(paths, 1):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[{i}/{len(paths)}] Konnte nicht laden: {os.path.basename(p)}")
            continue
        if img.ndim == 2:
            # Graubild -> ohne Alpha, ignorieren
            print(f"[{i}/{len(paths)}] Kein Alpha-Kanal: {os.path.basename(p)} (übersprungen)")
            continue
        if img.shape[2] == 3:
            print(f"[{i}/{len(paths)}] Kein Alpha-Kanal: {os.path.basename(p)} (übersprungen)")
            continue

        cleaned, removed = clean_image_alpha_based(
            img,
            alpha_threshold=args.alpha_threshold,
            kernel_size=args.kernel,
            iterations=args.iterations,
            min_area=args.min_area,
            zero_rgb_on_remove=not args.keep_rgb,
        )
        total_removed += removed

        out_path = os.path.join(outdir, os.path.basename(p))
        ok = cv2.imwrite(out_path, cleaned)
        status = "OK" if ok else "FEHLER"
        print(f"[{i}/{len(paths)}] {os.path.basename(p)} -> {status}, entfernt: {removed} Pixel")

    print(f"Fertig. Gesamt entfernte Pixel: {total_removed}")
    print(f"Ausgabeordner: {outdir}")


if __name__ == "__main__":
    main()
