from glob import glob
import os
import csv
import shutil
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_test_set(images_dir="./dataset/test_set/clean", csv_path="./dataset/test_set.csv", 
                 label_map_path="./dataset/rcmalli_vggface_labels_v2.npy"):
    # Istanza del dataset
    return TestSet(images_dir, csv_path, label_map_path)

# Funzione per creare il dataset di test clean a partire da un file CSV
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


def get_train_set(images_dir="./dataset/detectors_train_set/clean"):
    return TrainSet(images_dir)


# Funzione per creare il dataset di train clean a partire da un file CSV
def create_detectors_training_set(dataset_directory_origin, dataset_directory_destination, number_img):
    # Crea la directory di destinazione se non esiste
    os.makedirs(dataset_directory_destination, exist_ok=True)

    # Lista di tutte le sottocartelle (classi) nella directory origine
    class_folders = [os.path.join(dataset_directory_origin, d) for d in os.listdir(dataset_directory_origin) if os.path.isdir(os.path.join(dataset_directory_origin, d))]

    # Lista di tutte le immagini in tutte le sottocartelle
    all_images = []
    for folder in class_folders:
        images = glob(os.path.join(folder, '*.jpg'))
        all_images.extend(images)

    # Controllo se ci sono abbastanza immagini
    if number_img > len(all_images):
        raise ValueError(f"Numero richiesto di immagini ({number_img}) maggiore del numero disponibile ({len(all_images)}).")

    # Seleziona immagini randomicamente
    selected_images = random.sample(all_images, number_img)

    # Copia le immagini nella directory di destinazione
    for i, image_path in enumerate(selected_images):
        filename = f"img_{i:05d}.jpg"
        destination_path = os.path.join(dataset_directory_destination, filename)
        shutil.copy(image_path, destination_path)


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
        if image.size != (224, 224):
            image = transforms.Resize(256)(image)
            image = transforms.CenterCrop(224)(image)
        image = np.array(image, dtype=np.float32)/255
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
        return test_images, test_labels


# Classe per la gestione del test set
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
        if image.size != (224, 224):
            #image = transforms.Resize((160,160))(image) # Si puo utilizzare anche questo, ma calano le prestazioni di entrambi
            image = transforms.Resize(256)(image)
            image = transforms.CenterCrop(224)(image)
        image = np.array(image, dtype=np.float32)/255
        return transforms.ToTensor()(image)
    
    def get_images(self):
        dataloader = DataLoader(self, batch_size=32, shuffle=False)
        train_images = []

        for images in dataloader:
            train_images.append(images.numpy())

        train_images = np.concatenate(train_images, axis=0)
        return train_images
    

if __name__ == "__main__":
    # Parametri
    random.seed(2025) # Imposta il seed per la riproducibilità
    Train_set = True # True se si vuole creare il training set, False se si vuole creare il test set
    if (Train_set):
        number_img_train = 1000 # Numero di immagini totale del training set
        dataset_directory_origin = '.\\dataset\\vggface2_train\\train' # Directory di origine contenente le sottocartelle per ogni ID persona con le immagini
        dataset_directory_destination = '.\\dataset\\detectors_train_set\\clean' # Directory in cui salvare le immagini selezionate
        create_detectors_training_set(dataset_directory_origin, dataset_directory_destination, number_img_train)
    else:
        csv_file = '.\\dataset\\test_set.csv' # Percorso al file CSV contenente gli ID delle persone da inserire nel dataset finale (es. 'n000016'). Ogni riga deve contenere almeno un ID
        dataset_directory_origin = '.\\dataset\\vggface2_train\\train' # Directory di origine contenente le sottocartelle per ogni ID persona con le immagini
        dataset_directory_destination = '.\\dataset\\test_set\\clean' # Directory in cui salvare le immagini selezionate
        number_img_test = 10 # Numero massimo di immagini da copiare per ciascun ID (10 per 100 persone = 1000 img)
        create_test_set(csv_file, dataset_directory_origin, dataset_directory_destination, number_img_test)