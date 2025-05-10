import argparse
import torch
from generate_adv_example import generate_test_adv
from nets import setup_classifierNN1, setup_classifierNN2
from dataset import get_test_set
from security_evaluation_curve import run_fgsm, run_bim, run_pgd, run_df, run_cw
from utils import *


NUM_CLASSES = 8631


def main():
    parser = argparse.ArgumentParser(description="Run adversarial attacks on a classifier.")
    parser.add_argument("--classifier_name", type=str, default="NN1", choices=["NN1", "NN2", "NN1 + detectors"], help="Classifier to test")
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold per le rilevazioni dei detector')
    parser.add_argument("--verbose", type=bool, default=True, help="Print detailed information during the generation of adversarial examples")
    parser.add_argument('--generate_test_set_adv', type=bool, default=True, help='True to generate test set adv')
    args = parser.parse_args()
    
    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Attacchi selezionati 
    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]
    attack_types = ["fgsm", "bim"]
    print(f"Selected attacks: {attack_types}")

    # Test set clean
    test_set = get_test_set()
    images, labels = test_set.get_images()

    # Setup del classificatore da testare
    detectors = None
    if args.classifier_name == "NN2":
        classifier = setup_classifierNN2(device)
        images = process_images(images) # Preprocessing delle immagini per il secondo classificatore
    else:
        classifier = setup_classifierNN1(device)
        
    #### FASE DI VALUTAZIONE SUI DATI CLEAN ####

    # Calcolo dell'accuracy sulle immagini clean rispetto alle label vere
    accuracy_clean = compute_accuracy(classifier, images, labels)
    print(f"Accuracy del classificatore {args.classifier_name} su dati clean: {accuracy_clean}")

    # Calcolo della targeted accuracy sulle immagini clean rispetto alle label della classe target
    target_class_label = "Cristiano_Ronaldo"
    target_class = test_set.get_true_label(target_class_label)
    targeted_labels = target_class * torch.ones(labels.size, dtype=torch.long)
    targeted_accuracy_clean = compute_accuracy(classifier, images, targeted_labels)
    print(f"Targeted accuracy del classificatore {args.classifier_name} su dati clean: {targeted_accuracy_clean}")

    # Valutazione del classificatore col sistema di difesa
    if args.classifier_name == "NN1 + detectors":
        # Caricamento dei detectors
        detectors = load_detectors(attack_types, device)

        # Valutare detectors + classifier sui dati clean del test set
        adv_labels = np.zeros(images.shape[0], dtype=bool) # Tutti i campioni sono puliti (classe 0)
        accuracy_clean, fp = compute_accuracy_with_detectors(classifier, images, labels, adv_labels, detectors, args.threshold, False, verbose=True)
        print(f"Accuracy del classificatore NN1 col filtraggio dei detectors: {accuracy_clean:.4f}")
        print(f"Numero di immagini scartate dai detectors (FP): {fp}")

        # Calcolo della targeted accuracy sulle immagini clean rispetto alle label della classe target
        targeted_accuracy_clean, fp = compute_accuracy_with_detectors(classifier, images, targeted_labels, adv_labels, detectors, args.threshold, True, verbose=True)
        print(f"Accuracy del classificatore NN1 col filtraggio dei detectors: {targeted_accuracy_clean:.4f}")
        print(f"Numero di immagini scartate dai detectors (FP): {fp}")

    #### FASE DI VALUTAZIONE SUI DATI ADV ####

    # GENERAZIONE TEST SET ADV
    if args.generate_test_set_adv:
        generate_test_adv(setup_classifierNN1(device), images, attack_types, verbose=args.verbose)
        generate_test_adv(setup_classifierNN1(device), images, attack_types, True, target_class, args.verbose) # TARGETED

    # Avvio dell'attacco selezionato UNTARGETED
    if "fgsm" in attack_types:
        run_fgsm(classifier, args.classifier_name, test_set, accuracy_clean, detectors=detectors, threshold=args.threshold)
    if "bim" in attack_types:
        run_bim(classifier, args.classifier_name, test_set, accuracy_clean, detectors=detectors, threshold=args.threshold)
    if "pgd" in attack_types:
        run_pgd(classifier, args.classifier_name, test_set, accuracy_clean, detectors=detectors, threshold=args.threshold)
    if "df" in attack_types:
        run_df(classifier, args.classifier_name, test_set, accuracy_clean, detectors=detectors, threshold=args.threshold)
    if "cw" in attack_types:
        run_cw(classifier, args.classifier_name, test_set, accuracy_clean, detectors=detectors, threshold=args.threshold)

    # Avvio dell'attacco selezionato TARGETED
    if "fgsm" in attack_types:
        run_fgsm(classifier, args.classifier_name, test_set, accuracy_clean, True, targeted_accuracy_clean, target_class, detectors, args.threshold)
    if "bim" in attack_types:
        run_bim(classifier, args.classifier_name, test_set, accuracy_clean, True, targeted_accuracy_clean, target_class, detectors, args.threshold)
    if "pgd" in attack_types:
        run_pgd(classifier, args.classifier_name, test_set, accuracy_clean, True, targeted_accuracy_clean, target_class, detectors, args.threshold)
    if "cw" in attack_types:
        run_cw(classifier, args.classifier_name, test_set, accuracy_clean, True, targeted_accuracy_clean, target_class, detectors, args.threshold)


if __name__ == "__main__":
    main()