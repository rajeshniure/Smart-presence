import os
from ultralytics import YOLO

def main():
    # Paths
    data_yaml = 'face_data.yaml'
    project_dir = 'yolo_runs'
    epochs = 20
    imgsz = 224
    batch = 32

    # Create a data.yaml file for YOLO
    with open(data_yaml, 'w') as f:
        f.write(f"""
train: datasets/face detection/images/train
val: datasets/face detection/images/val
nc: 1
names: ['face']
""")

    # Train YOLOv8
    model = YOLO('yolov8n.pt') 
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, project=project_dir, name='face_yolo')

    # Validate and export metrics
    metrics = model.val()
    print('Validation metrics:', metrics)
    # Save confusion matrix and other results
    if hasattr(metrics, 'confusion_matrix'):
        metrics.confusion_matrix.plot(save_dir=project_dir)

if __name__ == '__main__':
    main() 