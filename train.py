from ultralytics.models import YOLO
 
if __name__ == '__main__':
    model = YOLO(model='./ultralytics/cfg/models/pst/yolo11-pst.yaml')
    model.train(data='./ultralytics/cfg/datasets/coco.yaml', 
                epochs=600, 
                optimizer='SGD',
                imgsz=640, 
                batch=128, 
                device=[0,1,2,3,4,5,6,7], 
                workers=8, 
                cache=False,
                amp=True, 
                hsv_h = 0.015,
                hsv_s = 0.7,
                hsv_v = 0.4,
                translate = 0.1,
                scale=0.5,  # N:0.5; S:0.9; M:0.9;
                mosaic=1.0,
                close_mosaic = 10,
                mixup=0.0,  # N:0.0; S:0.05; M:0.15;
                copy_paste=0.1,  # N:0.1; S:0.15; M:0.4;
                project='run/train', 
                name='expyolo11n-pst')
 