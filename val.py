from ultralytics import YOLO

model = YOLO('best.pt')
metrics = model.val(data='./ultralytics/cfg/datasets/coco.yaml', save_json=True)
print(metrics.box.map)