from ultralytics.models import YOLO
 
if __name__ == '__main__':
    model = YOLO(model='./ultralytics/cfg/models/pst/yolo11-cls-pst.yaml')
    model.train(data='/path/ImageNet1K', 
                optimizer='SGD',
                lr0=0.2,
                lrf=0.01,
                epochs=200, 
                imgsz=224, 
                batch=256, 
                device=[0,1,2,3,4,5,6,7], 
                workers=16, 
                cache=False,
                amp=True, 
                weight_decay=1e-4,
                warmup_epochs=0,
                nbs=256,
                hsv_h = 0.015,
                hsv_s = 0.4,
                hsv_v = 0.4,
                erasing=0.4,
                momentum=0.9,
                auto_augment='randaugment',
                cos_lr=True,
                project='run/train-cls', 
                name='expyolo11n-cls-pst')
 