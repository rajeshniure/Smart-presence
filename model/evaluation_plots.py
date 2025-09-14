import os
from face_recognition_plots import FaceRecognitionPlotter
from face_detection_plots import FaceDetectionPlotter

def main():
    """Main function to run the evaluation"""
    print("🚀 Starting Model Evaluation...")
    print("="*60)
    
    # Initialize plotters
    recognition_plotter = FaceRecognitionPlotter()
    detection_plotter = FaceDetectionPlotter()
    
    # Face Recognition Evaluation
    recognition_history_path = "face_recognition/recognition_training_history.pkl"
    if os.path.exists(recognition_history_path):
        recognition_plotter.load_training_history(recognition_history_path)
        print("\n Generating Face Recognition Model Plots...")
        recognition_plotter.plot_training_metrics("Face Recognition")
        recognition_plotter.plot_confusion_matrix("Face Recognition")
        recognition_plotter.generate_evaluation_report()
        print("\n Face Recognition evaluation complete!")
    else:
        print(f"Warning: {recognition_history_path} not found!")
    
    # Face Detection Evaluation
    detection_model_path = "face detection/mobilenet_1_0_224_tf.h5"
    if os.path.exists(detection_model_path):
        detection_plotter.load_face_detection_model(detection_model_path)
        print("\n Generating Face Detection Model Plots...")
        detection_plotter.plot_training_metrics("Face Detection")
        detection_plotter.plot_confusion_matrix("Face Detection")
        detection_plotter.generate_evaluation_report()
        print("\n Face Detection evaluation complete!")
    else:
        print(f"Warning: {detection_model_path} not found!")
    
    print("\n" + "="*60)
    print("🎉 All evaluations complete! All plots have been generated.")

if __name__ == "__main__":
    main()