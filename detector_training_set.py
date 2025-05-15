from glob import glob
import os
import random
import shutil
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from attacks import BIM, CW, DF, FGSM, PGD
from nets import setup_classifierNN1
from utils import save_images_as_npy

NUM_CLASSES = 8631  # numero di classi nel dataset VGGFace2
DIM_TRAIN = 1000 # numero di immagini del training set del detector

# Trasformazioni da applicare a un'immagine
trans = transforms.Compose([
    transforms.ToTensor(), # converte l'immagine [0, 255] in tensore [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # converte il valore dei canali R, G, B da [0, 1] a [-1, 1]
])

def get_train_set(images_dir="./dataset/detectors_train_set/clean/processed"):
    return TrainSet(images_dir)

# Classe per la gestione del train set
class TrainSet(Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.n_max_images = DIM_TRAIN
        self.samples = []

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"La directory {self.images_dir} non esiste.")
        
        # Scorre i file nella directory e aggiunge i file .jpg alla lista dei sample
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

# Funzione per creare il dataset di train clean a partire da un file CSV
def generate_train_clean(dataset_directory_origin, dataset_directory_destination, number_img):
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

# Generazione del train set avversario per i detectors
def generate_train_adv(classifier, images, attack_types, with_targeted=False):
    for attack_name in attack_types:
        
        if attack_name == "fgsm":
            attack = FGSM(classifier)
            epsilon_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
            save_dir = "./dataset/detectors_train_set/adversarial_examples/fgsm"
            n_samples = images.shape[0]
            split_size = n_samples // len(epsilon_values)
            for i, eps in enumerate(epsilon_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values)-1 else n_samples
                # Generazione campioni 50% targeted 50% untargeted
                if with_targeted:
                    x_subset_untargeted = images[start:(start+end)//2]
                    x_subset_targeted = images[(start+end)//2:end]
                    adv_examples_untargeted = attack.generate_attack(x_subset_untargeted, eps)
                    save_images_as_npy(adv_examples_untargeted, f"untargeted_eps_{eps}", save_dir)
                    targeted_labels = attack.randint(low=0, high=NUM_CLASSES-1, size=(len(x_subset_targeted),))
                    adv_examples_targeted = attack.generate_attack(x_subset_targeted, eps, targeted=True, targeted_labels=targeted_labels)
                    save_images_as_npy(adv_examples_targeted, f"targeted_eps_{eps}", save_dir)
                # Generazione campioni 100% untargeted
                else:
                    x_subset_untargeted = images[start:end]
                    adv_examples_untargeted = attack.generate_attack(x_subset_untargeted, eps)
                    save_images_as_npy(adv_examples_untargeted, f"untargeted_eps_{eps}", save_dir)
            print("Training adversarial examples generated and saved successfully for fgsm.")
        
        if attack_name == "bim":
            attack = BIM(classifier)
            epsilon_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
            save_dir = "./dataset/detectors_train_set/adversarial_examples/bim"
            n_samples = images.shape[0]
            split_size = n_samples // len(epsilon_values)
            for i, eps in enumerate(epsilon_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values)-1 else n_samples
                # Generazione campioni 50% targeted 50% untargeted
                if with_targeted:
                    x_subset_untargeted = images[start:(start+end)//2]
                    x_subset_targeted = images[(start+end)//2:end]
                    adv_examples_untargeted = attack.generate_attack(x_subset_untargeted, eps, epsilon_step=0.01, max_iter=10)
                    save_images_as_npy(adv_examples_untargeted, f"untargeted_eps_{eps}", save_dir)
                    targeted_labels = torch.randint(low=0, high=NUM_CLASSES-1, size=(len(x_subset_targeted),))
                    adv_examples_targeted = attack.generate_attack(x_subset_targeted, eps, epsilon_step=0.01, max_iter=10, targeted=True, targeted_labels=targeted_labels)
                    save_images_as_npy(adv_examples_targeted, f"targeted_eps_{eps}", save_dir)
                # Generazione campioni 100% untargeted
                else:
                    x_subset_untargeted = images[start:end]
                    adv_examples_untargeted = attack.generate_attack(x_subset_untargeted, eps, epsilon_step=0.01, max_iter=10)
                    save_images_as_npy(adv_examples_untargeted, f"untargeted_eps_{eps}", save_dir)
            print("Training adversarial examples generated and saved successfully for bim.")
        
        if attack_name == "pgd":
            attack = PGD(classifier)
            epsilon_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
            save_dir = "./dataset/detectors_train_set/adversarial_examples/pgd"
            n_samples = images.shape[0]
            split_size = n_samples // len(epsilon_values)
            for i, eps in enumerate(epsilon_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values)-1 else n_samples
                # Generazione campioni 50% targeted 50% untargeted
                if with_targeted:
                    x_subset_untargeted = images[start:(start+end)//2]
                    x_subset_targeted = images[(start+end)//2:end]
                    adv_examples_untargeted = attack.generate_attack(x_subset_untargeted, eps, epsilon_step=0.01, max_iter=10)
                    save_images_as_npy(adv_examples_untargeted, f"untargeted_eps_{eps}", save_dir)
                    targeted_labels = torch.randint(low=0, high=NUM_CLASSES-1, size=(len(x_subset_targeted),))
                    adv_examples_targeted = attack.generate_attack(x_subset_targeted, eps, epsilon_step=0.01, max_iter=10, targeted=True, targeted_labels=targeted_labels)
                    save_images_as_npy(adv_examples_targeted, f"targeted_eps_{eps}", save_dir)
                # Generazione campioni 100% untargeted
                else:
                    x_subset_untargeted = images[start:end]
                    adv_examples_untargeted = attack.generate_attack(x_subset_untargeted, eps, epsilon_step=0.01, max_iter=10)
                    save_images_as_npy(adv_examples_untargeted, f"untargeted_eps_{eps}", save_dir)
            print("Training adversarial examples generated and saved successfully for pgd.")

        if attack_name == "df":
            attack = DF(classifier)
            epsilon_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            save_dir = "./dataset/detectors_train_set/adversarial_examples/df"
            n_samples = images.shape[0]
            split_size = n_samples // len(epsilon_values)
            for i, eps in enumerate(epsilon_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values)-1 else n_samples
                x_subset = images[start:end]
                adv_examples = attack.generate_attack(x_subset, eps, nb_grads=10, max_iter=10)
                save_images_as_npy(adv_examples, f"eps_{eps}", save_dir)
            print("Training adversarial examples generated and saved successfully for df.")

        if attack_name == "cw":
            attack = CW(classifier)
            confidence_values = [0.01, 0.1, 1]
            save_dir = "./dataset/detectors_train_set/adversarial_examples/cw"
            n_samples = images.shape[0]
            split_size = n_samples // len(confidence_values)
            for i, conf in enumerate(confidence_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(confidence_values)-1 else n_samples
                # Generazione campioni 50% targeted 50% untargeted
                if with_targeted:
                    x_subset_untargeted = images[start:(start+end)//2]
                    x_subset_targeted = images[(start+end)//2:end]
                    adv_examples_untargeted = attack.generate_attack(x_subset_untargeted, conf, learning_rate=0.01, max_iter=3)
                    save_images_as_npy(adv_examples_untargeted, f"untargeted_conf_{conf}", save_dir)
                    targeted_labels = torch.randint(low=0, high=NUM_CLASSES-1, size=(len(x_subset_targeted),))
                    adv_examples_targeted = attack.generate_attack(x_subset_targeted, conf, learning_rate=0.01, max_iter=3, targeted=True, targeted_labels=targeted_labels)
                    save_images_as_npy(adv_examples_targeted, f"targeted_conf_{conf}", save_dir)
                # Generazione campioni 100% untargeted
                else:
                    x_subset_untargeted = images[start:end]
                    adv_examples_untargeted = attack.generate_attack(x_subset_untargeted, conf, learning_rate=0.01, max_iter=3)
                    save_images_as_npy(adv_examples_untargeted, f"untargeted_eps_{eps}", save_dir)
            print("Training adversarial examples generated and saved successfully for cw.")

if __name__ == "__main__":
    random.seed(33) # Imposta il seed per la riproducibilità
    dataset_directory_origin = './dataset/vggface2_train/train' # Directory di origine contenente le sottocartelle per ogni ID persona con le immagini
    dataset_directory_destination = './dataset/detectors_train_set/clean/original' # Directory in cui salvare le immagini selezionate
    dataset_directory_processed = './dataset/detectors_train_set/clean/processed' # Directory in cui salvare le immagini proccessate
    os.makedirs(dataset_directory_destination, exist_ok=True)
    generate_train_clean(dataset_directory_origin, dataset_directory_destination, DIM_TRAIN)
    process_dataset(dataset_directory_destination, dataset_directory_processed)
    
    # Generazione dati adv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack_types = ["df", "cw"]
    train_images_clean = get_train_set().get_images() 
    generate_train_adv(setup_classifierNN1(device), train_images_clean, attack_types)
    