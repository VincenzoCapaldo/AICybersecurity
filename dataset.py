import os
import csv
import shutil
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Classe per la gestione del dataset
class FaceDataset(Dataset):
    def __init__(self, images_dir="./dataset/test_set", csv_path="./dataset/test_set.csv", 
                 label_map_path="./dataset/rcmalli_vggface_labels_v2.npy"):
        self.images_dir = images_dir
        self.samples = []

        # Carica le etichette vere (str -> int)
        LABELS = np.load(label_map_path)
        self.true_labels = {str(name).strip(): idx for idx, name in enumerate(LABELS)}

        # Legge il CSV e costruisce i path delle immagini e le label
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                person_dir, name = row[0], row[1].strip(' "')
                full_dir = os.path.join(images_dir, person_dir)
                if os.path.isdir(full_dir):
                    for img_file in os.listdir(full_dir):
                        img_path = os.path.join(full_dir, img_file)
                        if os.path.isfile(img_path):
                            label = self.true_labels.get(name)
                            if label is not None:
                                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        image = transforms.Resize((160, 160))(image)
        image = np.array(image, dtype=np.uint8)
        return transforms.ToTensor()(image), label

    def get_true_label(self, name):
        return self.true_labels.get(name)

    def get_used_labels(self):
        return sorted({label for _, label in self.samples})
    
    def get_images(self):
        dataloader = DataLoader(self, batch_size=32, shuffle=False)
        test_images, test_labels = [], []

        for images, labels in dataloader:
            test_images.append(images.numpy())
            test_labels.append(labels)

        test_images = np.concatenate(test_images, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        return dataloader, test_images, test_labels


def get_test_set():
    # Istanza del test_set
    return FaceDataset()

# Funzione per creare il dataset di test a partire da un file CSV
def create_test_set(csv_file, dataset_directory_origin, dataset_directory_destination, number_img):
    
    # Lettura del file CSV contenente gli ID delle persone
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')  # i campi sono separati da virgole
        for row in reader:
            if len(row) < 1:
                continue  # salta righe vuote

            person_id = row[0].strip()  # es: "n000016"
            person_name = row[1].strip(' "')

            # Costruzione dei percorsi di origine e destinazione per ogni persona
            origin_path = os.path.join(dataset_directory_origin, person_id)
            destination_path = os.path.join(dataset_directory_destination, person_id)

            if not os.path.exists(origin_path):
                print(f"Cartella non trovata: {origin_path}")
                continue

            # Lista dei file immagine nella cartella della persona
            images = [f for f in os.listdir(origin_path) if os.path.isfile(os.path.join(origin_path, f))]
            if not images:
                print(f"Nessuna immagine in: {origin_path}")
                continue

            # Seleziona un numero massimo di immagini in modo casuale
            random.seed(2025)
            selected_images = random.sample(images, min(number_img, len(images)))

            # Crea la cartella di destinazione se non esiste
            os.makedirs(destination_path, exist_ok=True)

            # Copia e rinomina le immagini nella cartella di destinazione
            for idx, image in enumerate(selected_images):
                src = os.path.join(origin_path, image)
                ext = os.path.splitext(image)[1]  # estensione del file (es. .jpg)
                dst_filename = f"{person_id}_{idx + 1:02d}{ext}"  # es: n000016_01.jpg
                dst = os.path.join(destination_path, dst_filename)
                shutil.copy2(src, dst)

            print(f"Copiate {len(selected_images)} immagini per {person_name}")

if __name__ == "__main__":
    # Parametri
    csv_file = '.\\dataset\\test_set.csv' # Percorso al file CSV contenente gli ID delle persone da inserire nel dataset finale (es. 'n000016'). Ogni riga deve contenere almeno un ID
    dataset_directory_origin = '.\\dataset\\vggface2_train\\train' # Directory di origine contenente le sottocartelle per ogni ID persona con le immagini
    dataset_directory_destination = '.\\dataset\\test_set' # Directory in cui salvare le immagini selezionate
    number_img = 10 # Numero massimo di immagini da copiare per ciascun ID
    
    create_test_set(csv_file, dataset_directory_origin, dataset_directory_destination, number_img)