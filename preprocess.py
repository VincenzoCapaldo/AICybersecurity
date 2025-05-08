from glob import glob
import os
import csv
import shutil
import random
import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch

# Funzione per processare le immagini del test set, portandole ad una size di 224x224 float [-1,1]
def process_test_set(dataset_directory_destination, dataset_directory_processed):
    # Inizializza MTCNN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=224, margin=80, select_largest=True, post_process=False, device=device)

    def salva_immagine_con_struttura(orig_path, face_img):
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
                    face = mtcnn(image)
                    if face is not None:
                        face_np = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # normalizzazione [-1, 1]
                        face_pil = Image.fromarray(face_np)
                    else:
                        print(f"[!] Volto non rilevato: {img_path}")
                        # Effettua un resize + center_crop
                        resize, crop = 256, 224
                        image = image.resize((resize, resize))
                        face_pil = image.crop(((resize-crop)//2, (resize-crop)//2, (resize+crop)//2, (resize+crop)//2))
                        face_pil = face_pil.resize((224, 224))

                    salva_immagine_con_struttura(img_path, face_pil)

                except Exception as e:
                    print(f"[X] Errore con immagine {img_path}: {e}")

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

if __name__ == "__main__":
    # Parametri
    random.seed(2025) # Imposta il seed per la riproducibilità
    Train_set = False # True se si vuole creare il training set, False se si vuole creare il test set
    if (Train_set):
        number_img_train = 1000 # Numero di immagini totale del training set
        dataset_directory_origin = './dataset/vggface2_train/train' # Directory di origine contenente le sottocartelle per ogni ID persona con le immagini
        dataset_directory_destination = './dataset/detectors_train_set/clean' # Directory in cui salvare le immagini selezionate
        os.makedirs(dataset_directory_destination, exist_ok=True)
        create_detectors_training_set(dataset_directory_origin, dataset_directory_destination, number_img_train)
    else:
        csv_file = './dataset/test_set/test_set.csv' # Percorso al file CSV contenente gli ID delle persone da inserire nel dataset finale (es. 'n000016'). Ogni riga deve contenere almeno un ID
        dataset_directory_origin = './dataset/vggface2_train/train' # Directory di origine contenente le sottocartelle per ogni ID persona con le immagini
        dataset_directory_destination = './dataset/test_set/clean/original' # Directory in cui salvare le immagini selezionate
        dataset_directory_processed = './dataset/test_set/clean/processed' # Directory in cui salvare le immagini proccessate
        number_img_test = 10 # Numero massimo di immagini da copiare per ciascun ID (10 per 100 persone = 1000 img)
        #create_test_set(csv_file, dataset_directory_origin, dataset_directory_destination, number_img_test)
        process_test_set(dataset_directory_destination, dataset_directory_processed)