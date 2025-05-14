import numpy as np
import os
from nets import get_detector, setup_detector_classifier
from art.defences.detector.evasion import BinaryInputDetector
import torch
from detector_training_set import get_train_set
from utils import *


def main():
    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directory per i modelli
    os.makedirs("./models", exist_ok=True)
    
    # indica i detector da addestrare o caricare
    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]
    attack_types = ["fgsm", "bim", "pgd"]

    # Training set di partenza, con immagini clean
    train_images_clean = get_train_set().get_images()
    nb_train = train_images_clean.shape[0]

    #### FASE DI TRAINING ####
    if True:
        for attack_type in attack_types:
            model_path = os.path.join("./models", f"{attack_type}_detector.pth")
            detector_classifier = setup_detector_classifier(device)

            print(f"Training detector for attack: {attack_type}")
            detector = BinaryInputDetector(detector_classifier)
            
            # Trainining set avversario
            training_set_path = os.path.join("./dataset/detectors_train_set/adversarial_examples/", attack_type)
            train_images_adv = load_images_from_npy_folder(training_set_path)
            train_images_adv = np.concatenate(train_images_adv, axis=0)
            print(f"Train clean images shape: {np.shape(train_images_clean)}")
            print(f"Train images adversarial shape: {np.shape(train_images_adv)}")

            # Concatenazione delle immagini clean e avversarie
            x_train_detector = np.concatenate((train_images_clean, train_images_adv), axis=0)

            # Creazione delle etichette per il training set
            y_train_detector = np.concatenate((np.array([[1, 0]] * nb_train), np.array([[0, 1]] * np.shape(train_images_adv)[0])), axis=0)

            # Inizio addestramento del detector
            detector.fit(x_train_detector, y_train_detector, nb_epochs=30, batch_size=16, verbose=True)
            detector_classifier.model.eval()

            # Salvataggio dello state_dict del modello
            torch.save(detector_classifier.model.state_dict(), model_path)
            print(f"Detector salvato in: {model_path}")
    else:
        for attack_type in attack_types:
            model_path = os.path.join("./models", f"{attack_type}_detector.pth")
            detector = get_detector(device)

            print(f"Training detector for attack: {attack_type}")
            
            # Trainining set avversario
            training_set_path = os.path.join("./dataset/detectors_train_set/adversarial_examples/", attack_type)
            train_images_adv=load_images_from_npy_folder(training_set_path)
            train_images_adv = np.concatenate(train_images_adv, axis=0)
            print(f"Train clean images shape: {np.shape(train_images_clean)}")
            print(f"Train images adversarial shape: {np.shape(train_images_adv)}")

            # Concatenazione delle immagini clean e avversarie
            x_train_detector = np.concatenate((train_images_clean, train_images_adv), axis=0)

            # Creazione delle etichette per il training set
            y_train_detector = np.concatenate((np.array([[1, 0]] * nb_train), np.array([[0, 1]] * np.shape(train_images_adv)[0])), axis=0)

            # Inizio addestramento del detector
            # Converti i numpy array in tensori PyTorch
            x_train_tensor = torch.tensor(x_train_detector, dtype=torch.float32)
            y_train_tensor = torch.tensor(np.argmax(y_train_detector, axis=1), dtype=torch.long)
            # Train 
            detector.fit(x_train_tensor, y_train_tensor, nb_epochs=40, batch_size=16, verbose=True, device=device, patience=5)
            # Salvataggio dello state_dict del modello
            torch.save(detector.state_dict(), model_path)
            print(f"Detector salvato in: {model_path}")

if __name__ == "__main__":
    main()