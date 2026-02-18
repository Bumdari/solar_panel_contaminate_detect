import json

with open('contamination_segment.json', 'r') as f:
    coco = json.load(f)

clean_annotations = []
for ann in coco['annotations']:
    valid_segs = [seg for seg in ann['segmentation'] if len(seg) // 2 >= 3]
    if valid_segs:
        ann['segmentation'] = valid_segs
        clean_annotations.append(ann)

coco['annotations'] = clean_annotations

with open('contamination_segment_clean.json', 'w') as f:
    json.dump(coco, f)

print(f"Цэвэрлэсний дараа: {len(clean_annotations)} annotation")