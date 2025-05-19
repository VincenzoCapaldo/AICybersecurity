import argparse
import torch
from nets import setup_classifierNN1, setup_classifierNN2
from test_set import get_test_set
from security_evaluation_curve import run_fgsm, run_bim, run_pgd, run_df, run_cw
from utils import *


NUM_CLASSES = 8631  # numero di classi nel dataset VGGFace2

def main():
    parser = argparse.ArgumentParser(description="Run adversarial attacks on a classifier.")
    parser.add_argument('--generate_samples', type=bool, default=False, help='True to generate test set adv')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]
    attack_types = ["fgsm", "bim", "pgd"]

    # Test set clean
    test_set = get_test_set()
    clean_images, clean_labels = test_set.get_images()

    # Setup dei classificatori
    classifiers_name = ["NN1", "NN2", "NN1 + detector"] # Nome dei classificatori
    classifiers = {}
    classifiers["NN1"] = setup_classifierNN1(device)
    classifiers["NN2"] = setup_classifierNN2(device)
    classifiers["NN1 + detector"] = setup_classifierNN1(device)

    #### FASE DI VALUTAZIONE SUI DATI CLEAN DEI CLASSIFICATORI ####
    accuracies_clean = {}
    targeted_accuracies_clean = {}
    adv_labels = np.zeros(clean_images.shape[0], dtype=bool) # Tutti i campioni sono puliti (classe 0)
    for name in classifiers_name:
        # Caricamento delle immagini clean
        if name == "NN1":
            images = clean_images
            detectors = None
        elif name == "NN2":
            images = process_images(clean_images) # Preprocessing delle immagini per il secondo classificatore
            detectors = None
        else:
            images = clean_images
            detectors = load_detectors(attack_types, device)

        # Calcolo dell'accuracy sulle immagini clean rispetto alle label vere
        accuracies_clean[name] = compute_accuracy(classifiers[name], images, clean_labels, adv_labels, detectors)
        print(f"Accuracy del classificatore {name} su dati clean: {accuracies_clean[name]:.3f}")

        # Calcolo della targeted accuracy sulle immagini clean rispetto alle label della classe target
        target_class_label = "Cristiano_Ronaldo"
        target_class = test_set.get_true_label(target_class_label)
        targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
        targeted_accuracies_clean[name] = compute_accuracy(classifiers[name], images, targeted_labels, adv_labels, detectors, targeted=True)
        print(f"Targeted accuracy del classificatore {name} su dati clean: {targeted_accuracies_clean[name]:.3f}")

    #### FASE DI VALUTAZIONE SUI DATI ADV ####

    # Avvio dell'attacco selezionato UNTARGETED
    if "fgsm" in attack_types:
        run_fgsm(classifiers, classifiers_name, detectors, test_set, accuracies_clean, generate_samples=args.generate_samples)
    if "bim" in attack_types:
        run_bim(classifiers, classifiers_name, detectors, test_set, accuracies_clean, generate_samples=args.generate_samples)
    if "pgd" in attack_types:
        run_pgd(classifiers, classifiers_name, detectors, test_set, accuracies_clean, generate_samples=args.generate_samples)
    if "df" in attack_types:
        run_df(classifiers, classifiers_name, detectors, test_set, accuracies_clean, generate_samples=args.generate_samples)
    if "cw" in attack_types:
        run_cw(classifiers, classifiers_name, detectors, test_set, accuracies_clean, generate_samples=args.generate_samples)

    # Avvio dell'attacco selezionato TARGETED
    if "fgsm" in attack_types:
        run_fgsm(classifiers, classifiers_name, detectors, test_set, accuracies_clean, True, target_class, targeted_accuracies_clean, args.generate_samples)
    if "bim" in attack_types:
        run_bim(classifiers, classifiers_name, detectors, test_set, accuracies_clean, True, target_class, targeted_accuracies_clean, args.generate_samples)
    if "pgd" in attack_types:
        run_pgd(classifiers, classifiers_name, detectors, test_set, accuracies_clean, True, target_class, targeted_accuracies_clean, args.generate_samples)
    if "cw" in attack_types:
        run_cw(classifiers, classifiers_name, detectors, test_set, accuracies_clean, True, target_class, targeted_accuracies_clean, args.generate_samples)


if __name__ == "__main__":
    main()