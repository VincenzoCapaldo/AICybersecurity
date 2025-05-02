import argparse
import torch
from nets import setup_classifierNN1, setup_classifierNN2
from dataset import get_test_set
from security_evaluation_curve import run_fgsm, run_bim, run_pgd, run_df, run_cw
from utils import *


NUM_CLASSES = 8631


def main():
    parser = argparse.ArgumentParser(description="Run adversarial attacks on a classifier.")
    parser.add_argument("--test_classifierNN1", type=bool, default=True, help="If true test on classifierNN1, otherwise test on classifierNN2")
    parser.add_argument("--targeted", type=bool, default=True, help="Run a targeted attack")
    args = parser.parse_args()
    
    # Attacchi selezionato
    attacks = ["fgsm", "bim", "pgd", "df", "cw"]
    attacks = ["fgsm", "bim", "pgd"]
    print(f"Selected attacks: {attacks}")
    print(f"Targeted attack: {args.targeted}")
    
    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup del classificatore
    if args.test_classifierNN1:
        classifier = setup_classifierNN1(device)
        name = "NN1"
    else:
        classifier = setup_classifierNN2(device)
        name = "NN2"

    #### FASE DI VALUTAZIONE ####
    test_set = get_test_set()
    images, labels = test_set.get_images()

    # Preprocessing delle immagini per il secondo classificatore
    if not args.test_classifierNN1:
        images = process_images(images)

    # Calcolo dell'accuracy sulle immagini clean rispetto alle label vere
    accuracy_clean = compute_accuracy(classifier, images, labels)
    print(f"Accuracy del classificatore {name} su dati clean: {accuracy_clean}")

    # Calcolo della targeted accuracy sulle immagini clean rispetto alle label della classe target
    target_class_label = "Cristiano_Ronaldo"
    target_class = test_set.get_true_label(target_class_label)
    targeted_labels = target_class * torch.ones(labels.size, dtype=torch.long)
    targeted_accuracy_clean = compute_accuracy(classifier, images, targeted_labels)
    print(f"Targeted accuracy del classificatore {name} su dati clean: {targeted_accuracy_clean}")

    # Avvio dell'attacco selezionato
    if "fgsm" in attacks:
        run_fgsm(classifier, name, args.targeted, test_set, accuracy_clean, targeted_accuracy_clean, target_class)
    if "bim" in attacks:
        run_bim(classifier, name, args.targeted, test_set, accuracy_clean, targeted_accuracy_clean, target_class)
    if "pgd" in attacks:
        run_pgd(classifier, name, args.targeted, test_set, accuracy_clean, targeted_accuracy_clean, target_class)
    if "df" in attacks:
        run_df(classifier, name, test_set, accuracy_clean)
    if "cw" in attacks:
        run_cw(classifier, name, args.targeted, test_set, accuracy_clean, targeted_accuracy_clean, target_class)


if __name__ == "__main__":
    main()