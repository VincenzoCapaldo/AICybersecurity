import torch
from torchvision import transforms
import os
import csv
from PIL import Image
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from sklearn.metrics import classification_report, accuracy_score

def load_images(filepath):
    img = Image.open(filepath).convert("RGB")
    rsz = img.resize((160, 160))
    tns = transforms.ToTensor()(rsz)
    return tns, rsz


def evaluate_performance(net, test_set=".\\dataset\\test_set", labels=".\\dataset\\test_set.csv"):
    y_true = []
    y_pred = []

    with open(labels, "r") as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            filename, name = row[0], row[1]
            img_path = os.path.join(test_set, filename)
            for img in os.listdir(img_path):
                try:
                    img_tensor, _ = load_images(img)
                except Exception as e:
                    print(f"Errore nel caricamento di {img_path}: {e}")
                    continue

                img_tensor = img_tensor.unsqueeze(0)  # Aggiunge dimensione batch
                with torch.no_grad():
                    output = net(img_tensor)
                    predicted_class = torch.argmax(output, dim=1).item()

                y_true.append(0)
                y_pred.append(predicted_class)
                print(name, predicted_class)

    print("Report di classificazione:")
    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    resnet.classify = True
    evaluate_performance(resnet.to(device))