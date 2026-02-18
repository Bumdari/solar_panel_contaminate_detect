"""
=============================================================================
 Solar Panel Contamination Detection — Test & Inference Script
 Phase 2: solar_panel, snow, dust, bird_droppings detection
 4 class segmentation model — contamination = snow + dust + bird_droppings
=============================================================================
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test_results.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class Config:
    # --- PATHS (raw string — Windows backslash-д зориулсан) ---
    PROJECT_ROOT = r"C:\Users\bumdari.b\Develop\Solar_panel_monitoring_system\contamination_detect\test_pt"
    DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

    # Training дуусаад best.pt-ийн зам
    MODEL_PATH = os.path.join(
        PROJECT_ROOT, "contam_phase2_20260212_1123", "weights", "best.pt"
    )

    # Test зургуудын зам
    TEST_ROOT = os.path.join(PROJECT_ROOT, "test_images")
    TEST_FOLDERS = ["cctv", "edge_cases", "other_panels"]

    # Үр дүн хадгалах
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_results")

    # --- INFERENCE ---
    IMAGE_SIZE = 640
    CONFIDENCE = 0.25       # Бүдэг тоосыг алдахгүй байх
    IOU_THRESHOLD = 0.45    # Ижил төрлийн давхардлыг арилгах
    DEVICE = "cpu"

    # --- CLASSES (YOLO class ID дарааллаар) ---
    # 0: solar_panel, 1: snow, 2: dust, 3: bird_droppings
    CLASS_NAMES = ["solar_panel", "snow", "dust", "bird_droppings"]
    NUM_CLASSES = 4

    # Бохирдлын class-ууд (solar_panel-ийг ОРУУЛАХГҮЙ)
    CONTAM_CLASSES = ["snow", "dust", "bird_droppings"]

    # Өнгө (BGR — OpenCV format, бүгд 3 утгатай)
    CLASS_COLORS = {
        "solar_panel": (255, 255, 255),     # White
        "snow": (255, 150, 0),              # Blue (BGR)
        "dust": (0, 165, 255),              # Orange (BGR)
        "bird_droppings": (0, 0, 255),      # Red (BGR)
    }

    # Panel талбайн хэдэн %-г бохирдол эзэлбэл цэвэрлэгээ шаардлагатай
    ALERT_THRESHOLD_PCT = 25.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SINGLE IMAGE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
def predict_single(model, image_path: str, cfg: Config) -> dict:
    """
    Нэг зураг дээр inference хийж, бохирдлын мэдээлэл буцаана.

    Чухал: total_contamination нь зөвхөн snow + dust + bird_droppings.
    solar_panel-ийн area-г тусад нь тооцоолж, бохирдлын хувийг
    panel-ийн талбайтай харьцуулж тооцоолно.
    """
    results = model.predict(
        source=image_path,
        imgsz=cfg.IMAGE_SIZE,
        conf=cfg.CONFIDENCE,
        iou=cfg.IOU_THRESHOLD,
        device=cfg.DEVICE,
        verbose=False,
        retina_masks=True,
    )

    result = results[0]
    img_h, img_w = result.orig_shape
    total_pixels = img_h * img_w

    detections = []
    class_area = {name: 0.0 for name in cfg.CLASS_NAMES}

    if result.masks is not None and len(result.masks) > 0:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            bbox = boxes.xyxy[i].cpu().numpy().tolist()

            if cls_id >= len(cfg.CLASS_NAMES):
                logger.warning(f"Unknown class_id={cls_id}, skipping")
                continue

            cls_name = cfg.CLASS_NAMES[cls_id]

            # Mask area тооцоолол
            mask = masks[i]
            if mask.shape != (img_h, img_w):
                import cv2
                mask = cv2.resize(
                    mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR
                )
            mask_binary = (mask > 0.5).astype(np.uint8)
            mask_area_px = int(np.sum(mask_binary))
            mask_area_pct = (mask_area_px / total_pixels) * 100

            class_area[cls_name] += mask_area_pct

            detections.append({
                "class": cls_name,
                "confidence": round(conf, 4),
                "bbox": [round(x, 1) for x in bbox],
                "mask_area_px": mask_area_px,
                "mask_area_pct": round(mask_area_pct, 2),
            })

    # ── Contamination тооцоолол ──
    # solar_panel-ийг ОРУУЛАХГҮЙ — зөвхөн бохирдлын class-ууд
    total_contamination = sum(class_area[c] for c in cfg.CONTAM_CLASSES)
    panel_area = class_area.get("solar_panel", 0.0)

    # Хэрэв panel илэрсэн бол: бохирдлыг panel-ийн талбайтай харьцуулах
    # Хэрэв panel илрээгүй бол: бүх зургийн талбайтай харьцуулах
    if panel_area > 0:
        contam_vs_panel_pct = (total_contamination / panel_area) * 100
    else:
        contam_vs_panel_pct = total_contamination  # fallback: % of image

    alert = contam_vs_panel_pct >= cfg.ALERT_THRESHOLD_PCT

    return {
        "image": os.path.basename(image_path),
        "image_path": image_path,
        "image_size": [img_w, img_h],
        "num_detections": len(detections),
        "detections": detections,
        "summary": {
            "panel_area_pct": round(panel_area, 2),
            "total_contamination_pct": round(total_contamination, 2),
            "contam_vs_panel_pct": round(contam_vs_panel_pct, 2),
            "by_class": {k: round(class_area[k], 2) for k in cfg.CONTAM_CLASSES},
            "alert": alert,
            "alert_message": (
                f"⚠️ ЦЭВЭРЛЭГЭЭ ШААРДЛАГАТАЙ! "
                f"Бохирдол: {contam_vs_panel_pct:.1f}% (panel-ийн {total_contamination:.1f}%)"
                if alert
                else f"✅ Хэвийн. Бохирдол: {contam_vs_panel_pct:.1f}%"
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def visualize_result(model, image_path: str, output_path: str, cfg: Config):
    """Зураг дээр mask, bbox, label бүгдийг зурж хадгалах."""
    try:
        import cv2
    except ImportError:
        logger.warning("opencv суулгаагүй: pip install opencv-python-headless")
        return None

    results = model.predict(
        source=image_path,
        imgsz=cfg.IMAGE_SIZE,
        conf=cfg.CONFIDENCE,
        iou=cfg.IOU_THRESHOLD,
        device=cfg.DEVICE,
        verbose=False,
        retina_masks=True,
    )

    result = results[0]
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Зураг уншиж чадсангүй: {image_path}")
        return None

    img_h, img_w = img.shape[:2]
    overlay = img.copy()

    if result.masks is not None and len(result.masks) > 0:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])

            if cls_id >= len(cfg.CLASS_NAMES):
                continue

            cls_name = cfg.CLASS_NAMES[cls_id]
            color = cfg.CLASS_COLORS[cls_name]
            bbox = boxes.xyxy[i].cpu().numpy().astype(int)

            # Mask overlay (semi-transparent)
            mask = masks[i]
            if mask.shape != (img_h, img_w):
                mask = cv2.resize(mask, (img_w, img_h))
            mask_binary = (mask > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(img)
            colored_mask[mask_binary == 1] = color

            # solar_panel-д бага opacity, бохирдолд илүү тод
            alpha = 0.2 if cls_name == "solar_panel" else 0.4
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)

            # Bounding box (solar_panel-д нимгэн шугам)
            thickness = 1 if cls_name == "solar_panel" else 2
            cv2.rectangle(
                overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness
            )

            # Label
            label = f"{cls_name} {conf:.2f}"
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                overlay,
                (bbox[0], bbox[1] - label_size[1] - baseline - 4),
                (bbox[0] + label_size[0], bbox[1]),
                color, -1,
            )
            cv2.putText(
                overlay, label,
                (bbox[0], bbox[1] - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

    # Folder label (зүүн дээд буланд)
    folder_name = Path(image_path).parent.name
    cv2.putText(
        overlay, f"[{folder_name}]", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )

    cv2.imwrite(output_path, overlay)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FOLDER STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_folder_stats(results: list, folder_name: str, cfg: Config) -> dict:
    """Нэг folder-ийн бүх inference үр дүнг нэгтгэх."""
    if not results:
        return {"folder": folder_name, "total_images": 0}

    contam_pcts = [r["summary"]["contam_vs_panel_pct"] for r in results]
    alert_count = sum(1 for r in results if r["summary"]["alert"])

    class_pcts = {name: [] for name in cfg.CONTAM_CLASSES}
    panel_pcts = []
    confidences = []

    for r in results:
        panel_pcts.append(r["summary"]["panel_area_pct"])
        for cls_name in cfg.CONTAM_CLASSES:
            class_pcts[cls_name].append(r["summary"]["by_class"].get(cls_name, 0.0))
        for d in r["detections"]:
            confidences.append(d["confidence"])

    stats = {
        "folder": folder_name,
        "total_images": len(results),
        "alerts": alert_count,
        "clean": len(results) - alert_count,
        "alert_rate_pct": round(alert_count / len(results) * 100, 1),
        "panel_detection": {
            "mean_area_pct": round(float(np.mean(panel_pcts)), 2),
            "detected_in": sum(1 for p in panel_pcts if p > 0),
        },
        "contamination": {
            "mean_pct": round(float(np.mean(contam_pcts)), 2),
            "max_pct": round(float(np.max(contam_pcts)), 2),
            "min_pct": round(float(np.min(contam_pcts)), 2),
            "std_pct": round(float(np.std(contam_pcts)), 2),
        },
        "per_class": {},
        "confidence": {},
    }

    for cls_name in cfg.CONTAM_CLASSES:
        pcts = class_pcts[cls_name]
        detected = [p for p in pcts if p > 0]
        stats["per_class"][cls_name] = {
            "detected_in": len(detected),
            "mean_pct": round(float(np.mean(detected)), 2) if detected else 0.0,
            "max_pct": round(float(np.max(detected)), 2) if detected else 0.0,
        }

    if confidences:
        stats["confidence"] = {
            "mean": round(float(np.mean(confidences)), 4),
            "min": round(float(np.min(confidences)), 4),
            "max": round(float(np.max(confidences)), 4),
        }

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PRINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def print_folder_report(stats: dict, cfg: Config):
    """Folder-ийн статистикийг хэвлэх."""
    f = stats["folder"]
    total = stats["total_images"]
    if total == 0:
        logger.info(f"  📂 {f}: зураг байхгүй")
        return

    logger.info(f"\n{'─'*60}")
    logger.info(f"  📂 {f.upper()}")
    logger.info(f"{'─'*60}")
    logger.info(f"  Images:        {total}")
    logger.info(
        f"  Alerts:        {stats['alerts']} / {total} "
        f"({stats['alert_rate_pct']}%)"
    )
    logger.info(f"  Clean:         {stats['clean']}")

    pd = stats["panel_detection"]
    logger.info(
        f"\n  Panel detection: {pd['detected_in']}/{total} images, "
        f"mean area={pd['mean_area_pct']:.1f}%"
    )

    c = stats["contamination"]
    logger.info(f"\n  Contamination (vs panel):")
    logger.info(f"    Mean: {c['mean_pct']:6.1f}%  ±{c['std_pct']:.1f}%")
    logger.info(f"    Max:  {c['max_pct']:6.1f}%")
    logger.info(f"    Min:  {c['min_pct']:6.1f}%")

    logger.info(f"\n  Per-class breakdown:")
    for cls_name in cfg.CONTAM_CLASSES:
        pc = stats["per_class"][cls_name]
        bar = "█" * min(int(pc["mean_pct"] / 2), 40)
        logger.info(
            f"    {cls_name:20s}: found in {pc['detected_in']:3d} images, "
            f"mean={pc['mean_pct']:5.1f}%, max={pc['max_pct']:5.1f}%  {bar}"
        )

    if stats["confidence"]:
        conf = stats["confidence"]
        logger.info(
            f"\n  Confidence: mean={conf['mean']:.3f}, "
            f"min={conf['min']:.3f}, max={conf['max']:.3f}"
        )


def print_comparison_table(all_stats: list, cfg: Config):
    """Бүх folder-уудын харьцуулсан хүснэгт."""
    logger.info(f"\n{'═'*75}")
    logger.info("  📊 FOLDER COMPARISON TABLE")
    logger.info(f"{'═'*75}")

    header = (
        f"  {'Folder':<18} {'Images':>6} {'Alerts':>7} {'Rate':>6} "
        f"{'Panel%':>7} {'Contam%':>8} {'Max%':>6} {'AvgConf':>8}"
    )
    logger.info(header)
    logger.info(f"  {'─'*70}")

    for s in all_stats:
        if s["total_images"] == 0:
            logger.info(f"  {s['folder']:<18} {'—':>6}")
            continue

        avg_conf = s["confidence"].get("mean", 0) if s["confidence"] else 0
        panel_pct = s["panel_detection"]["mean_area_pct"]
        logger.info(
            f"  {s['folder']:<18} "
            f"{s['total_images']:>6} "
            f"{s['alerts']:>7} "
            f"{s['alert_rate_pct']:>5.1f}% "
            f"{panel_pct:>6.1f}% "
            f"{s['contamination']['mean_pct']:>7.1f}% "
            f"{s['contamination']['max_pct']:>5.1f}% "
            f"{avg_conf:>8.3f}"
        )

    # Insights
    logger.info(f"\n  💡 INSIGHTS:")
    valid = [s for s in all_stats if s["total_images"] > 0]
    if not valid:
        return

    best_conf = max(valid, key=lambda s: s["confidence"].get("mean", 0))
    worst_conf = min(valid, key=lambda s: s["confidence"].get("mean", 0))
    logger.info(
        f"    Хамгийн итгэлтэй:      {best_conf['folder']} "
        f"(avg conf={best_conf['confidence'].get('mean', 0):.3f})"
    )
    logger.info(
        f"    Хамгийн бага итгэлтэй:  {worst_conf['folder']} "
        f"(avg conf={worst_conf['confidence'].get('mean', 0):.3f})"
    )

    for s in valid:
        if s["confidence"].get("mean", 1) < 0.4:
            logger.warning(
                f"    ⚠ {s['folder']}: confidence бага (mean={s['confidence']['mean']:.3f}). "
                f"Модель энэ төрлийн зургуудад сул байж магадгүй."
            )
        if s["panel_detection"]["detected_in"] < s["total_images"] * 0.5:
            logger.warning(
                f"    ⚠ {s['folder']}: panel зургуудын {s['panel_detection']['detected_in']}"
                f"/{s['total_images']}-д л илэрсэн. Panel detection сул байж магадгүй."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TEST RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_images(folder_path: str) -> list:
    """Folder дотор зургийн файлуудыг олох."""
    if not os.path.isdir(folder_path):
        return []
    return sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if Path(f).suffix.lower() in IMG_EXTS
    ])


def test_folder(model, folder_path: str, output_dir: str, cfg: Config) -> list:
    """Нэг folder дотор бүх зургуудыг тест хийх."""
    images = get_images(folder_path)
    if not images:
        logger.warning(f"  Зураг олдсонгүй: {folder_path}")
        return []

    folder_name = Path(folder_path).name
    vis_dir = os.path.join(output_dir, folder_name)
    os.makedirs(vis_dir, exist_ok=True)

    results = []
    for idx, img_path in enumerate(images, 1):
        img_name = os.path.basename(img_path)
        pred = predict_single(model, img_path, cfg)
        results.append(pred)

        status = "🔴" if pred["summary"]["alert"] else "🟢"
        s = pred["summary"]
        logger.info(
            f"  [{idx}/{len(images)}] {status} {img_name} — "
            f"{pred['num_detections']} det, "
            f"panel={s['panel_area_pct']:.1f}%, "
            f"contam={s['contam_vs_panel_pct']:.1f}% "
            f"(snow:{s['by_class']['snow']:.1f} "
            f"dust:{s['by_class']['dust']:.1f} "
            f"bird:{s['by_class']['bird_droppings']:.1f})"
        )

        vis_path = os.path.join(vis_dir, f"pred_{img_name}")
        visualize_result(model, img_path, vis_path, cfg)

    return results


def run_all_folders(cfg: Config):
    """Бүх test folder-уудыг тест хийж харьцуулсан report гаргах."""
    from ultralytics import YOLO

    if not os.path.exists(cfg.MODEL_PATH):
        logger.error(f"Model олдсонгүй: {cfg.MODEL_PATH}")
        logger.info("  best.pt-ийн зам олох:")
        logger.info(f"  dir /s /b \"{cfg.PROJECT_ROOT}\\runs\\*best.pt\"")
        sys.exit(1)

    logger.info(f"\n{'═'*60}")
    logger.info("  MULTI-FOLDER TEST — SOLAR PANEL CONTAMINATION")
    logger.info(f"  Model:     {cfg.MODEL_PATH}")
    logger.info(f"  Conf:      {cfg.CONFIDENCE}")
    logger.info(f"  IoU:       {cfg.IOU_THRESHOLD}")
    logger.info(f"  Alert at:  {cfg.ALERT_THRESHOLD_PCT}% of panel area")
    logger.info(f"{'═'*60}")

    model = YOLO(cfg.MODEL_PATH)

    # Discover folders
    if cfg.TEST_FOLDERS:
        folders = [
            os.path.join(cfg.TEST_ROOT, f)
            for f in cfg.TEST_FOLDERS
            if os.path.isdir(os.path.join(cfg.TEST_ROOT, f))
        ]
    else:
        folders = sorted([
            os.path.join(cfg.TEST_ROOT, d)
            for d in os.listdir(cfg.TEST_ROOT)
            if os.path.isdir(os.path.join(cfg.TEST_ROOT, d))
        ])

    if not folders:
        logger.error(f"Test folder олдсонгүй: {cfg.TEST_ROOT}")
        sys.exit(1)

    logger.info(f"\n  Found {len(folders)} test folders:")
    for f in folders:
        img_count = len(get_images(f))
        logger.info(f"    📂 {Path(f).name}: {img_count} images")

    # Run tests
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    all_folder_results = {}
    all_stats = []

    for folder_path in folders:
        folder_name = Path(folder_path).name
        logger.info(f"\n{'─'*60}")
        logger.info(f"  📂 Testing: {folder_name}")
        logger.info(f"{'─'*60}")

        results = test_folder(model, folder_path, cfg.OUTPUT_DIR, cfg)
        all_folder_results[folder_name] = results

        stats = compute_folder_stats(results, folder_name, cfg)
        all_stats.append(stats)
        print_folder_report(stats, cfg)

    # Comparison
    print_comparison_table(all_stats, cfg)

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": cfg.MODEL_PATH,
        "config": {
            "confidence": cfg.CONFIDENCE,
            "iou_threshold": cfg.IOU_THRESHOLD,
            "alert_threshold_pct": cfg.ALERT_THRESHOLD_PCT,
            "image_size": cfg.IMAGE_SIZE,
            "classes": cfg.CLASS_NAMES,
            "contam_classes": cfg.CONTAM_CLASSES,
        },
        "folder_stats": all_stats,
        "detailed_results": all_folder_results,
    }

    report_path = os.path.join(cfg.OUTPUT_DIR, "multi_folder_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'═'*60}")
    logger.info(f"  ✓ ТЕСТ ДУУСЛАА")
    logger.info(f"  Report:         {report_path}")
    logger.info(f"  Visualizations: {cfg.OUTPUT_DIR}\\[folder_name]\\")
    logger.info(f"{'═'*60}\n")

    return report


def run_single_folder(folder_name: str, cfg: Config):
    """Зөвхөн нэг folder тест хийх."""
    from ultralytics import YOLO

    if not os.path.exists(cfg.MODEL_PATH):
        logger.error(f"Model олдсонгүй: {cfg.MODEL_PATH}")
        sys.exit(1)

    folder_path = os.path.join(cfg.TEST_ROOT, folder_name)
    if not os.path.isdir(folder_path):
        logger.error(f"Folder олдсонгүй: {folder_path}")
        logger.info(f"  Боломжит folder-ууд: {cfg.TEST_FOLDERS}")
        sys.exit(1)

    model = YOLO(cfg.MODEL_PATH)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    logger.info(f"\n  Testing folder: {folder_name}")
    results = test_folder(model, folder_path, cfg.OUTPUT_DIR, cfg)
    stats = compute_folder_stats(results, folder_name, cfg)
    print_folder_report(stats, cfg)


def test_single_image(image_path: str, cfg: Config):
    """Нэг зураг тест."""
    from ultralytics import YOLO

    if not os.path.exists(cfg.MODEL_PATH):
        logger.error(f"Model олдсонгүй: {cfg.MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(image_path):
        logger.error(f"Зураг олдсонгүй: {image_path}")
        sys.exit(1)

    model = YOLO(cfg.MODEL_PATH)
    pred = predict_single(model, image_path, cfg)

    print(f"\n{'='*55}")
    print(f"  Image:  {pred['image']}")
    print(f"  Size:   {pred['image_size']}")
    print(f"  Folder: {Path(image_path).parent.name}")
    print(f"{'='*55}")

    s = pred["summary"]
    print(f"  Panel area:    {s['panel_area_pct']:.1f}% of image")
    print(f"  Detections:    {pred['num_detections']}")

    if pred["num_detections"] == 0:
        print("  ✅ Бохирдол илрээгүй")
    else:
        # Detection тус бүрийг хэвлэх
        for d in pred["detections"]:
            marker = "🟦" if d["class"] == "solar_panel" else "🔶"
            print(
                f"    {marker} {d['class']:20s} conf={d['confidence']:.2f}  "
                f"area={d['mask_area_pct']:.1f}%"
            )

        print(f"\n  {s['alert_message']}")
        print(f"\n  Contamination breakdown:")
        for cls in cfg.CONTAM_CLASSES:
            pct = s["by_class"][cls]
            if pct > 0:
                bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                print(f"    {cls:20s}: {bar} {pct:.1f}%")

    # Visualization
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    vis_path = os.path.join(cfg.OUTPUT_DIR, f"pred_{os.path.basename(image_path)}")
    visualize_result(model, image_path, vis_path, cfg)
    print(f"\n  Visualization: {vis_path}")


def run_validation(cfg: Config):
    """Validation set дээр mAP metrics."""
    from ultralytics import YOLO

    if not os.path.exists(cfg.MODEL_PATH):
        logger.error(f"Model олдсонгүй: {cfg.MODEL_PATH}")
        sys.exit(1)

    yaml_path = os.path.join(cfg.PROJECT_ROOT, "dataset.yaml")
    if not os.path.exists(yaml_path):
        logger.error(f"dataset.yaml олдсонгүй: {yaml_path}")
        sys.exit(1)

    model = YOLO(cfg.MODEL_PATH)

    logger.info("Validation metrics тооцоолж байна...")
    metrics = model.val(
        data=yaml_path,
        imgsz=cfg.IMAGE_SIZE,
        batch=8,
        device=cfg.DEVICE,
        plots=True,
        save_json=True,
        project=cfg.OUTPUT_DIR,
        name="validation",
    )

    print(f"\n{'='*55}")
    print("  📊 VALIDATION RESULTS")
    print(f"{'='*55}")
    print(f"  Box  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  Box  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Mask mAP@50:    {metrics.seg.map50:.4f}")
    print(f"  Mask mAP@50-95: {metrics.seg.map:.4f}")

    print(f"\n  Per-class Mask mAP@50:")
    for i, name in enumerate(cfg.CLASS_NAMES):
        if i < len(metrics.seg.maps):
            score = metrics.seg.maps[i]
            bar = "█" * int(score * 40) + "░" * (40 - int(score * 40))
            print(f"    {name:20s}: {bar} {score:.4f}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLI
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Solar Panel Contamination Detection — Test Script v2"
    )
    parser.add_argument(
        "mode",
        choices=["all", "folder", "single", "val", "batch"],
        help=(
            "all: бүх test folder | "
            "folder: нэг folder | "
            "batch: TEST_IMAGES дахь бүх зураг | "
            "single: нэг зураг | "
            "val: validation metrics"
        ),
    )
    parser.add_argument("--image", "-i", type=str, help="Зургийн зам (single mode)")
    parser.add_argument("--folder", "-f", type=str, help="Folder нэр (folder mode)")
    parser.add_argument("--model", "-m", type=str, help="Model path")
    parser.add_argument(
        "--conf", "-c", type=float, default=0.25, help="Confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="IoU threshold"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=25.0, help="Alert threshold %%"
    )

    args = parser.parse_args()
    cfg = Config()

    if args.model:
        cfg.MODEL_PATH = args.model
    cfg.CONFIDENCE = args.conf
    cfg.IOU_THRESHOLD = args.iou
    cfg.ALERT_THRESHOLD_PCT = args.threshold

    if args.mode == "all":
        run_all_folders(cfg)

    elif args.mode == "folder":
        if not args.folder:
            logger.error("--folder нэр өгөөрэй! (cctv, edge_cases, other_panels)")
            sys.exit(1)
        run_single_folder(args.folder, cfg)

    elif args.mode == "batch":
        # Legacy batch mode — TEST_IMAGES дээр шууд ажиллана
        from ultralytics import YOLO
        if not os.path.exists(cfg.MODEL_PATH):
            logger.error(f"Model олдсонгүй: {cfg.MODEL_PATH}")
            sys.exit(1)
        model = YOLO(cfg.MODEL_PATH)
        test_images_dir = os.path.join(cfg.DATASET_DIR, "test_images", "cctv")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        results = test_folder(model, test_images_dir, cfg.OUTPUT_DIR, cfg)
        stats = compute_folder_stats(results, "cctv", cfg)
        print_folder_report(stats, cfg)

    elif args.mode == "single":
        if not args.image:
            logger.error("--image зам өгөөрэй!")
            sys.exit(1)
        test_single_image(args.image, cfg)

    elif args.mode == "val":
        run_validation(cfg)


if __name__ == "__main__":
    main()