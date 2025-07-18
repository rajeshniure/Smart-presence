import os
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm

# Paths
DATASET_DIR = 'datasets/face_recognition'
EMBEDDINGS_PATH = 'facenet_embeddings.npy'
LABELS_PATH = 'facenet_labels.npy'
CLASSIFIER_PATH = 'facenet_svm.joblib'
ENCODER_PATH = 'facenet_label_encoder.joblib'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=0, device=device)

# Prepare data
embeddings = []
labels = []
people = os.listdir(DATASET_DIR)
for person in tqdm(people, desc='People'):
    person_dir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_dir):
        continue
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            face = mtcnn(img)
            if face is not None:
                face = face.unsqueeze(0).to(device)
                emb = resnet(face).detach().cpu().numpy()[0]
                embeddings.append(emb)
                labels.append(person)
        except Exception as e:
            print(f'Error processing {img_path}: {e}')

embeddings = np.array(embeddings)
labels = np.array(labels)
np.save(EMBEDDINGS_PATH, embeddings)
np.save(LABELS_PATH, labels)

# Encode labels
encoder = LabelEncoder()
labels_num = encoder.fit_transform(labels)
joblib.dump(encoder, ENCODER_PATH)

# Train SVM classifier
clf = SVC(kernel='linear', probability=True)
clf.fit(embeddings, labels_num)
joblib.dump(clf, CLASSIFIER_PATH)

print('FaceNet embeddings, label encoder, and SVM classifier saved.') 