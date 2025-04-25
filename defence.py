import argparse
import numpy as np
import os
from nets import setup_classifierNN1, setup_detector_classifier
from art.defences.detector.evasion import BinaryInputDetector
import torch
from dataset import get_test_set, get_train_set
from utils import *
from security_evaluation_curve import run_fgsm, run_bim, run_pgd, run_df, run_cw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_detectors', type=bool, default=False, help='Se True, addestra i detector; altrimenti carica i modelli salvati e procede con la valutazione')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold per le rilevazioni dei detector')
    parser.add_argument("--attack", type=str, default="fgsm", choices=["fgsm", "bim", "pgd", "df", "cw"], help="Type of attack to test")
    parser.add_argument("--targeted", type=bool, default=False, help="Test on targeted attack")
    args = parser.parse_args()

    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directory per i modelli
    os.makedirs("./models", exist_ok=True)

    # Train or load Detectors
    detectors = {}
    # indica i detector da addestrare o caricare
    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]

    # Fase di train dei detector
    if args.train_detectors:
        # Training set di partenza, con immagini clean
        train_images_clean = get_train_set().get_images()
        nb_train = train_images_clean.shape[0]
        for attack_type in attack_types:
            model_path = os.path.join("./models", f"{attack_type}_detector.pth")
            detector_classifier = setup_detector_classifier(device)

            print(f"Training detector for attack: {attack_type}")
            detectors[attack_type] = BinaryInputDetector(detector_classifier)
            
            # Trainining set avversario
            training_set_path = os.path.join("./dataset/detectors_train_set/adversarial_examples", attack_type)
            train_images_adv=load_images_from_npy_folder(training_set_path)

            # Concatenazione delle immagini clean e avversarie
            x_train_detector = np.concatenate((train_images_clean, train_images_adv), axis=0)

            # Creazione delle etichette per il training set
            y_train_detector = np.concatenate((np.array([[1, 0]] * nb_train), np.array([[0, 1]] * nb_train)), axis=0)

            # Inizio addestramento del detector
            detectors[attack_type].fit(x_train_detector, y_train_detector, nb_epochs=20, batch_size=16, verbose=True)
            detector_classifier.model.eval()

            # Salvataggio dello state_dict del modello
            torch.save(detector_classifier.model.state_dict(), model_path)
            print(f"Detector salvato in: {model_path}")
    
    # Carica i detector dai modelli salvati
    else:
        for attack_type in attack_types:
            model_path = os.path.join("./models", f"{attack_type}_detector.pth")
            detector_classifier = setup_detector_classifier(device)
            detector_classifier.model.load_state_dict(torch.load(model_path, map_location=device))
            detector_classifier.model.eval()
            detectors[attack_type] = BinaryInputDetector(detector_classifier)
            print(f"Detector caricato da: {model_path}")

        #### FASE DI VALUTAZIONE ####
        # Setup del classificatore
        classifier= setup_classifierNN1(device)

        # Carica le immagini e le etichette del test set
        test_set = get_test_set()
        test_images, test_labels = test_set.get_images()

        # Calcolo dell'accuracy sulle immagini clean rispetto alle label vere
        accuracy_clean = compute_accuracy(classifier, test_images, test_labels)
        print(f"Accuracy del classificatore NN1 su dati clean: {accuracy_clean}")

        # Valutare detectors + classifier sui dati clean del test set
        nb_test = test_images.shape[0]
        adv_labels = np.zeros(nb_test, dtype=bool) # Tutti i campioni sono puliti (classe 0)
        accuracy_clean, fp = compute_accuracy_with_detectors(classifier, test_images, test_labels, adv_labels, detectors, threshold=args.threshold, verbose=True)
        print(f"Accuracy del classificatore NN1 col filtraggio dei detectors: {accuracy_clean:.4f}")
        print(f"Numero di immagini scartate dai detectors (FP): {fp}")

        # Calcolo della targeted accuracy sulle immagini clean rispetto alle label della classe target
        target_class_label = "Cristiano_Ronaldo"
        target_class = test_set.get_true_label(target_class_label)
        targeted_labels = target_class * torch.ones(test_labels.size, dtype=torch.long)
        targeted_accuracy_clean, fp = compute_accuracy_with_detectors(classifier, test_images, targeted_labels, adv_labels, detectors, threshold=args.threshold, verbose=True)
        print(f"Accuracy del classificatore NN1 col filtraggio dei detectors: {targeted_accuracy_clean:.4f}")
        print(f"Numero di immagini scartate dai detectors (FP): {fp}")

        # Valutare detectors + classifier sui dati adversarial
        name = "NN1 + detectors"
        if args.attack == "fgsm":
            run_fgsm(classifier, name, args.targeted, accuracy_clean, targeted_accuracy_clean, target_class, detectors, args.threshold)
        elif args.attack == "bim":
            run_bim(classifier, name, args.targeted, accuracy_clean, targeted_accuracy_clean, target_class, detectors, args.threshold)
        elif args.attack == "pgd":
            run_pgd(classifier, name, args.targeted, accuracy_clean, targeted_accuracy_clean, target_class, detectors, args.threshold)
        elif args.attack == "df":
            run_df(classifier, name, args.targeted, accuracy_clean, targeted_accuracy_clean, target_class, detectors, args.threshold)
        elif args.attack == "cw":
            run_cw(classifier, name, args.targeted, accuracy_clean, targeted_accuracy_clean, target_class, detectors, args.threshold)

if __name__ == "__main__":
    main()