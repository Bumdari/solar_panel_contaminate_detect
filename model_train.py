import os
import sys
import yaml
import shutil
import logging
from pathlib import Path
from datetime import datetime

class Config:
    DATASET_ROOT = "/path/to/your/dataset"   
    PROJECT_DIR  = "./runs/contamination_seg"    
    MODEL_SIZE   = "yolov8m-seg.pt"  
    CLASS_NAMES  = {0: "dust", 1: "bird_droppings", 2: "snow"}
    NUM_CLASSES  = 3
    
    EPOCHS       = 150          
    BATCH_SIZE   = 8             
    IMG_SIZE     = 640           
    WORKERS      = 8             
    
    OPTIMIZER    = "AdamW"       
    LR0          = 0.001       
    LRF          = 0.01          
    MOMENTUM     = 0.937
    WEIGHT_DECAY = 0.0005
    WARMUP_EPOCHS = 5
    WARMUP_MOMENTUM = 0.8
    
    HSV_H        = 0.015        
    HSV_S        = 0.5           
    HSV_V        = 0.3           
    DEGREES      = 10.0         
    TRANSLATE    = 0.1          
    SCALE        = 0.4           
    FLIPLR       = 0.5           
    FLIPUD       = 0.2          
    MOSAIC       = 0.8           
    MIXUP        = 0.1           
    COPY_PASTE   = 0.15         
    ERASING      = 0.2          
    
    PATIENCE     = 30           
    SAVE_PERIOD  = 10           
    
    SEED         = 42
    AMP          = True         
    CACHE        = "ram"         

    CLOSE_MOSAIC = 20            
    RECT         = False         # Rectangular training (seg-д False байх)
    RESUME       = False         # Сүүлийн checkpoint-оос үргэлжлүүлэх


# ── Setup Logging ────────────────────────────────────────────────────────────

def setup_logging(project_dir: str) -> logging.Logger:
    """Configure logging to both console and file."""
    log_dir = Path(project_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ── Dataset Validation ───────────────────────────────────────────────────────

def validate_dataset(dataset_root: str, logger: logging.Logger) -> dict:
    """
    Dataset бүтцийг шалгах:
    dataset_root/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml  (автоматаар үүсгэнэ)
    """
    root = Path(dataset_root)
    
    required_dirs = [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]
    
    for d in required_dirs:
        if not d.exists():
            logger.error(f"Directory олдсонгүй: {d}")
            raise FileNotFoundError(f"Dataset directory олдсонгүй: {d}")
    
    # Count files
    stats = {}
    for split in ["train", "val"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]
        labels = [f for f in lbl_dir.iterdir() if f.suffix == ".txt"]
        
        # Image-label matching шалгах
        img_stems = {f.stem for f in images}
        lbl_stems = {f.stem for f in labels}
        
        missing_labels = img_stems - lbl_stems
        orphan_labels = lbl_stems - img_stems
        
        stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "matched": len(img_stems & lbl_stems),
            "missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
        }
        
        logger.info(f"  {split}: {len(images)} images, {len(labels)} labels, "
                     f"{len(img_stems & lbl_stems)} matched")
        
        if missing_labels:
            logger.warning(f"  ⚠ {split}: {len(missing_labels)} images have no labels")
            for m in list(missing_labels)[:5]:
                logger.warning(f"    - {m}")
                
        if orphan_labels:
            logger.warning(f"  ⚠ {split}: {len(orphan_labels)} orphan labels (no image)")
    
    # Label content шалгах (class index validation)
    logger.info("  Label content шалгаж байна...")
    invalid_count = 0
    class_distribution = {i: 0 for i in range(Config.NUM_CLASSES)}
    
    for split in ["train", "val"]:
        lbl_dir = root / "labels" / split
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file, "r") as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) < 7:  # class_id + min 3 x,y pairs = 7 values
                        invalid_count += 1
                        continue
                    class_id = int(parts[0])
                    if class_id not in Config.CLASS_NAMES:
                        logger.warning(
                            f"  ⚠ Invalid class {class_id} in {lbl_file.name}:{line_num}"
                        )
                        invalid_count += 1
                    else:
                        class_distribution[class_id] += 1
    
    logger.info("  Class тархалт:")
    total_annotations = sum(class_distribution.values())
    for cls_id, count in class_distribution.items():
        pct = (count / total_annotations * 100) if total_annotations > 0 else 0
        logger.info(f"    {Config.CLASS_NAMES[cls_id]:>15s}: {count:,} ({pct:.1f}%)")
    
    if invalid_count > 0:
        logger.warning(f"  ⚠ {invalid_count} invalid annotation(s) олдлоо")
    
    stats["class_distribution"] = class_distribution
    stats["total_annotations"] = total_annotations
    stats["invalid_annotations"] = invalid_count
    
    return stats


# ── Create data.yaml ─────────────────────────────────────────────────────────

def create_data_yaml(dataset_root: str, logger: logging.Logger) -> str:
    """YOLOv8-д зориулсан data.yaml үүсгэх."""
    root = Path(dataset_root)
    yaml_path = root / "data.yaml"
    
    data_config = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": Config.NUM_CLASSES,
        "names": Config.CLASS_NAMES,
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"  data.yaml үүсгэлээ: {yaml_path}")
    return str(yaml_path)


# ── Training ─────────────────────────────────────────────────────────────────

def train(data_yaml: str, logger: logging.Logger):
    """YOLOv8 segmentation training."""
    from ultralytics import YOLO
    
    # Load model
    logger.info(f"Model ачаалж байна: {Config.MODEL_SIZE}")
    model = YOLO(Config.MODEL_SIZE)
    
    # Run name with timestamp
    run_name = f"contam_seg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("=" * 60)
    logger.info("TRAINING ЭХЭЛЖ БАЙНА")
    logger.info("=" * 60)
    logger.info(f"  Model:      {Config.MODEL_SIZE}")
    logger.info(f"  Epochs:     {Config.EPOCHS}")
    logger.info(f"  Batch:      {Config.BATCH_SIZE}")
    logger.info(f"  Image size: {Config.IMG_SIZE}")
    logger.info(f"  Optimizer:  {Config.OPTIMIZER}")
    logger.info(f"  LR:         {Config.LR0} → {Config.LR0 * Config.LRF}")
    logger.info(f"  AMP:        {Config.AMP}")
    logger.info(f"  Cache:      {Config.CACHE}")
    logger.info("=" * 60)
    
    # Train
    results = model.train(
        # ── Data ──
        data=data_yaml,
        task="segment",
        
        # ── Training ──
        epochs=Config.EPOCHS,
        batch=Config.BATCH_SIZE,
        imgsz=Config.IMG_SIZE,
        workers=Config.WORKERS,
        seed=Config.SEED,
        deterministic=True,
        
        # ── Optimizer ──
        optimizer=Config.OPTIMIZER,
        lr0=Config.LR0,
        lrf=Config.LRF,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_epochs=Config.WARMUP_EPOCHS,
        warmup_momentum=Config.WARMUP_MOMENTUM,
        
        # ── Augmentation ──
        hsv_h=Config.HSV_H,
        hsv_s=Config.HSV_S,
        hsv_v=Config.HSV_V,
        degrees=Config.DEGREES,
        translate=Config.TRANSLATE,
        scale=Config.SCALE,
        fliplr=Config.FLIPLR,
        flipud=Config.FLIPUD,
        mosaic=Config.MOSAIC,
        mixup=Config.MIXUP,
        copy_paste=Config.COPY_PASTE,
        erasing=Config.ERASING,
        
        # ── Checkpointing & Early Stop ──
        patience=Config.PATIENCE,
        save_period=Config.SAVE_PERIOD,
        close_mosaic=Config.CLOSE_MOSAIC,
        
        # ── Performance ──
        amp=Config.AMP,
        cache=Config.CACHE,
        rect=Config.RECT,
        
        # ── Output ──
        project=Config.PROJECT_DIR,
        name=run_name,
        exist_ok=False,
        plots=True,
        save=True,
        save_json=True,         # COCO format үр дүн хадгалах
        
        # ── Resume ──
        resume=Config.RESUME,
        
        # ── Logging ──
        verbose=True,
    )
    
    return results, run_name


# ── Post-training Evaluation ─────────────────────────────────────────────────

def evaluate_model(run_name: str, data_yaml: str, logger: logging.Logger):
    """Train дууссаны дараа best model-ийг нарийвчлан шалгах."""
    from ultralytics import YOLO
    
    best_model_path = Path(Config.PROJECT_DIR) / run_name / "weights" / "best.pt"
    
    if not best_model_path.exists():
        logger.error(f"Best model олдсонгүй: {best_model_path}")
        return
    
    logger.info("=" * 60)
    logger.info("POST-TRAINING EVALUATION")
    logger.info("=" * 60)
    
    model = YOLO(str(best_model_path))
    
    # Validation set дээр evaluate
    metrics = model.val(
        data=data_yaml,
        imgsz=Config.IMG_SIZE,
        batch=Config.BATCH_SIZE,
        split="val",
        save_json=True,
        plots=True,
        verbose=True,
    )
    
    # Results summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    
    # Box metrics
    logger.info("\n📦 Bounding Box Metrics:")
    logger.info(f"  mAP50:      {metrics.box.map50:.4f}")
    logger.info(f"  mAP50-95:   {metrics.box.map:.4f}")
    
    # Segmentation metrics
    logger.info("\n🎭 Segmentation Metrics:")
    logger.info(f"  mAP50:      {metrics.seg.map50:.4f}")
    logger.info(f"  mAP50-95:   {metrics.seg.map:.4f}")
    
    # Per-class metrics
    logger.info("\n📊 Per-class (Seg mAP50):")
    if hasattr(metrics.seg, 'maps') and metrics.seg.maps is not None:
        for i, cls_map in enumerate(metrics.seg.maps):
            cls_name = Config.CLASS_NAMES.get(i, f"class_{i}")
            logger.info(f"  {cls_name:>15s}: {cls_map:.4f}")
    
    # Speed
    logger.info(f"\n⚡ Inference speed: {metrics.speed.get('inference', 'N/A')} ms/image")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Best model: {best_model_path}")
    logger.info(f"Results:    {Path(Config.PROJECT_DIR) / run_name}")
    logger.info("=" * 60)
    
    return metrics


# ── Export for Production ────────────────────────────────────────────────────

def export_model(run_name: str, logger: logging.Logger):
    """Production deploy-д зориулж export хийх."""
    from ultralytics import YOLO
    
    best_model_path = Path(Config.PROJECT_DIR) / run_name / "weights" / "best.pt"
    model = YOLO(str(best_model_path))
    
    # ONNX export (хамгийн түгээмэл)
    logger.info("ONNX format руу export хийж байна...")
    model.export(
        format="onnx",
        imgsz=Config.IMG_SIZE,
        simplify=True,
        dynamic=False,
        half=False,             # ONNX-д FP32 ашиглах
    )
    
    # TensorRT export (NVIDIA GPU дээр хамгийн хурдан)
    try:
        logger.info("TensorRT format руу export хийж байна...")
        model.export(
            format="engine",
            imgsz=Config.IMG_SIZE,
            half=True,          # TensorRT FP16
            device=0,
        )
        logger.info("  TensorRT export амжилттай!")
    except Exception as e:
        logger.warning(f"  TensorRT export алдаа (TensorRT суугаагүй байж магадгүй): {e}")
    
    logger.info("Export дууслаа!")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Full training pipeline."""
    
    # 1. Setup
    logger = setup_logging(Config.PROJECT_DIR)
    logger.info("🔆 Solar Panel Contamination Detection - Phase 2 Training")
    logger.info(f"   Dataset: {Config.DATASET_ROOT}")
    logger.info(f"   Output:  {Config.PROJECT_DIR}")
    
    # 2. Validate dataset
    logger.info("\n📁 Dataset шалгаж байна...")
    try:
        stats = validate_dataset(Config.DATASET_ROOT, logger)
    except FileNotFoundError as e:
        logger.error(f"\n❌ {e}")
        logger.error("DATASET_ROOT зөв тохируулсан эсэхийг шалгана уу!")
        sys.exit(1)
    
    # 3. Create data.yaml
    logger.info("\n📝 data.yaml үүсгэж байна...")
    data_yaml = create_data_yaml(Config.DATASET_ROOT, logger)
    
    # 4. Train
    logger.info("\n🚀 Training эхэлж байна...")
    results, run_name = train(data_yaml, logger)
    
    # 5. Evaluate
    logger.info("\n📊 Evaluation хийж байна...")
    metrics = evaluate_model(run_name, data_yaml, logger)
    
    # 6. Export (optional - uncomment if needed)
    # logger.info("\n📦 Model export хийж байна...")
    # export_model(run_name, logger)
    
    logger.info("\n✅ Бүх процесс амжилттай дууслаа!")
    logger.info(f"   Best model: {Config.PROJECT_DIR}/{run_name}/weights/best.pt")


if __name__ == "__main__":
    main()