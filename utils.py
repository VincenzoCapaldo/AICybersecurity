import torch
from torchvision import transforms
import os
import csv
import numpy as np
from PIL import Image
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# pre-processa le immagini in modo compatibile con la rete.
def load_images(filepath):
    img = Image.open(filepath).convert("RGB")
    rsz = img.resize((160, 160))
    tns = transforms.ToTensor()(rsz)
    return tns, rsz


def evaluate_performance(device, net, test_set=".\\dataset\\test_set", labels=".\\dataset\\test_set.csv"):
    y_true = []
    y_pred = []
    # carica la label.
    LABELS = np.load(".\\dataset\\rcmalli_vggface_labels_v2.npy")

    with open(labels, "r", encoding="utf-8") as csvfile: # apri con la codifica corretta.
        reader = csv.reader(csvfile)

        for row in reader:
            filename, name = row[0], row[1].strip(' "') # togli spazio e doppi apici.
            img_path = os.path.join(test_set, filename)
            for img in os.listdir(img_path):
                try:
                    img_tensor, _ = load_images(img_path + "/" + img) # usa il path completo.
                except Exception as e:
                    print(f"Errore nel caricamento di {img_path}: {e}")
                    continue

                img_tensor = img_tensor.unsqueeze(0)  # aggiunge dimensione batch.
                with torch.no_grad():
                    output = net(img_tensor.to(device))
                    # dato l'output ottieni la stringa corrispondente.
                    predicted_class = LABELS[np.array(output[0].detach().cpu().numpy()).argmax()]

                y_true.append(0)
                y_pred.append(predicted_class)
                print(name, predicted_class)

    print("Report di classificazione:")
    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    resnet.classify = True
    evaluate_performance(device, resnet.to(device))