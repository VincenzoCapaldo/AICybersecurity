import argparse
import numpy as np
import os
from generate_adv_example import generate_train_adv
from nets import setup_classifierNN1, setup_detector_classifier
from art.defences.detector.evasion import BinaryInputDetector
import torch
from dataset import get_train_set
from utils import *


def main():
    parser = argparse.ArgumentParser(description="Run adversarial attacks on a classifier.")
    parser.add_argument('--generate_train_set_adv', type=bool, default=True, help='True to generate train set adv')
    parser.add_argument('--verbose', type=bool, default=True, help='True to generate train set adv')
    args = parser.parse_args()

    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directory per i modelli
    os.makedirs("./models", exist_ok=True)
    
    # indica i detector da addestrare o caricare
    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]
    attack_types = ["pgd"]

    # Training set di partenza, con immagini clean
    train_images_clean = get_train_set().get_images()
    nb_train = train_images_clean.shape[0]

    # GENERAZIONE TRAIN SET ADV
    if args.generate_train_set_adv:
        generate_train_adv(setup_classifierNN1(device), train_images_clean, attack_types, verbose=args.verbose)

    #### FASE DI TRAINING ####
    detectors = {}
    for attack_type in attack_types:
        model_path = os.path.join("./models", f"{attack_type}_detector.pth")
        detector_classifier = setup_detector_classifier(device)

        print(f"Training detector for attack: {attack_type}")
        detectors[attack_type] = BinaryInputDetector(detector_classifier)
        
        # Trainining set avversario
        training_set_path = os.path.join("./dataset/detectors_train_set/adversarial_examples/", attack_type)
        train_images_adv=load_images_from_npy_folder(training_set_path)
        #for i in enumerate(train_images_adv):
        #    print(f"{i}: {train_images_adv[i].shape}")
        train_images_adv = np.concatenate(train_images_adv, axis=0)
        #train_images_adv = np.array(train_images_adv).reshape(-1, 3, 224, 224)
        print(f"Train clean images shape: {np.shape(train_images_clean)}")
        print(f"Train images adversarial shape: {np.shape(train_images_adv)}")

        # Concatenazione delle immagini clean e avversarie
        x_train_detector = np.concatenate((train_images_clean, train_images_adv), axis=0)

        # Creazione delle etichette per il training set
        y_train_detector = np.concatenate((np.array([[1, 0]] * nb_train), np.array([[0, 1]] * np.shape(train_images_adv)[0])), axis=0)

        # Inizio addestramento del detector
        detectors[attack_type].fit(x_train_detector, y_train_detector, nb_epochs=20, batch_size=16, verbose=True)
        detector_classifier.model.eval()

        # Salvataggio dello state_dict del modello
        torch.save(detector_classifier.model.state_dict(), model_path)
        print(f"Detector salvato in: {model_path}")

if __name__ == "__main__":
    main()