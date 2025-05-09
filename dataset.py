from glob import glob
import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import fixed_image_standardization

# trasforma l'immagine in float, in tensore e poi la rappresenta da un intervallo [0, 255] a [-1, 1]
trans = transforms.Compose([
    transforms.ToTensor(),  # converte da HWC uint8 [0, 255] numpy → CHW float32 [0.0, 1.0] tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0,1] → [-1,1] 
])

# processed utilizza immagini con resize + centercrop
def get_test_set(images_dir="./dataset/test_set/clean/processed", csv_path="./dataset/test_set/test_set.csv", 
                 label_map_path="./dataset/rcmalli_vggface_labels_v2.npy"):
    return TestSet(images_dir, csv_path, label_map_path)


def get_train_set(images_dir="./dataset/detectors_train_set/clean/processed"):
    return TrainSet(images_dir)


# Classe per la gestione del test set
class TestSet(Dataset):
    def __init__(self, images_dir, csv_path, label_map_path):
        self.images_dir = images_dir
        self.n_max_person = 100 # Numero massimo di immagini del dataset (per fare prove più veloci)
        self.n_max_images_person = 10 # Numero massimo di immagini del dataset (per fare prove più veloci)
        
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"La directory {self.images_dir} non esiste.")
        self.samples = []

        # Carica le etichette vere (str -> int)
        LABELS = np.load(label_map_path)
        self.true_labels = {str(name).strip(): idx for idx, name in enumerate(LABELS)}

        # Legge il CSV e costruisce i path delle immagini e le label
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < self.n_max_person: # Limita il numero di persone
                    person_dir, name = row[0], row[1].strip(' "')
                    full_dir = os.path.join(images_dir, person_dir)
                    if os.path.isdir(full_dir):

                        # Se la directory esiste, cerca le immagini della persona selezionata
                        for i, img_file in enumerate(os.listdir(full_dir)):
                            img_path = os.path.join(full_dir, img_file)
                            if os.path.isfile(img_path):
                                if i < self.n_max_images_person: # Limita il numero di immagini per persona
                                    #print(len(self.samples))
                                    label = self.true_labels.get(name)
                                    if label is not None:
                                        self.samples.append((img_path, label))
        
        if len(self.samples) == 0:
            raise FileNotFoundError(f"Nessuna immagine trovata nella directory {self.images_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        return trans(image), label

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
        return test_images, test_labels


# Classe per la gestione del train set
class TrainSet(Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.n_max_images = 1000 # Numero massimo di immagini del dataset (per fare prove più veloci)
        self.samples = []

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"La directory {self.images_dir} non esiste.")
        
        for i, fname in enumerate(os.listdir(self.images_dir)):
            if fname.endswith('.jpg') and i < self.n_max_images:
                self.samples.append(os.path.join(self.images_dir, fname))
            if i >= self.n_max_images:
                break

        if len(self.samples) == 0:
            raise FileNotFoundError(f"Nessuna immagine .jpg trovata nella directory {self.images_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path)
        return trans(image)
    
    def get_images(self):
        dataloader = DataLoader(self, batch_size=32, shuffle=False)
        train_images = []

        for images in dataloader:
            train_images.append(images.numpy())

        train_images = np.concatenate(train_images, axis=0)
        return train_images