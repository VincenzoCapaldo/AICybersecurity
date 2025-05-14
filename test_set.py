from glob import glob
import os
import csv
import random
import shutil
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tqdm

# Trasforma l'immagine in float, in tensore e poi la rappresenta da un intervallo [0, 255] a [-1, 1]
trans = transforms.Compose([
    transforms.ToTensor(),  # converte da HWC uint8 [0, 255] numpy → CHW float32 [0.0, 1.0] tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0,1] → [-1,1] 
])

def get_test_set(images_dir="./dataset/test_set/clean/processed", csv_path="./dataset/test_set/test_set.csv", 
                 label_map_path="./dataset/rcmalli_vggface_labels_v2.npy"):
    return TestSet(images_dir, csv_path, label_map_path)

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

# Funzione per creare il dataset di test clean a partire da un file CSV
def create_test_set(csv_file, dataset_directory_origin, dataset_directory_destination, number_img):
    # Crea la directory di destinazione se non esiste
    os.makedirs(dataset_directory_destination, exist_ok=True)
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

# Funzione per processare le immagini del test set, portandole ad una size di 224x224
def process_dataset(dataset_directory_destination, dataset_directory_processed):
    
    def save_image(orig_path, face_img):
        relative_path = os.path.relpath(orig_path, dataset_directory_destination)
        save_path = os.path.join(dataset_directory_processed, relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        face_img.save(save_path)

    for root, _, files in os.walk(dataset_directory_destination):
        for fname in tqdm(files, desc="Processing"):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, fname)
                try:
                    image = Image.open(img_path).convert("RGB")
                    image = transforms.Resize(256)(image)
                    image = transforms.CenterCrop(224)(image)
                    save_image(img_path, image)

                except Exception as e:
                    print(f"[X] Errore con immagine {img_path}: {e}")

if __name__ == "__main__":
    random.seed(33) # Imposta il seed per la riproducibilità
    csv_file = './dataset/test_set/test_set.csv' # Percorso al file CSV contenente gli ID delle persone da inserire nel dataset finale (es. 'n000016'). Ogni riga deve contenere almeno un ID
    dataset_directory_origin = './dataset/vggface2_train/train' # Directory di origine contenente le sottocartelle per ogni ID persona con le immagini
    dataset_directory_destination = './dataset/test_set/clean/original' # Directory in cui salvare le immagini selezionate
    dataset_directory_processed = './dataset/test_set/clean/processed' # Directory in cui salvare le immagini proccessate
    number_img_test = 10 # Numero massimo di immagini da copiare per ciascun ID (10 per 100 persone = 1000 img)
    create_test_set(csv_file, dataset_directory_origin, dataset_directory_destination, number_img_test)
    process_dataset(dataset_directory_destination, dataset_directory_processed)