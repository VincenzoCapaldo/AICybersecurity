import argparse
from nets import get_NN1, get_NN2
import torch
from torch.optim import Adam
from art.estimators.classification import PyTorchClassifier
from dataset import get_test_set
from security_evaluation_curve import run_fgsm, run_bim, run_pgd, run_df, run_cw
from utils import *

NUM_CLASSES = 8631

def setup_classifiers(device, classify=True):
    # Istanzio le reti
    nn1 = get_NN1(device, classify)
    nn2 = get_NN2(device)

    # Definizione dei classificatori
    classifierNN1 = PyTorchClassifier(
        model=nn1,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=Adam(nn1.parameters(), lr=0.001),
        input_shape=(3, 160, 160),
        channels_first=True,
        nb_classes=NUM_CLASSES,
        clip_values=(0.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    classifierNN2 = PyTorchClassifier(
        model=nn2,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=Adam(nn2.parameters(), lr=0.001),
        input_shape=(3, 224, 224),
        channels_first=True,
        nb_classes=NUM_CLASSES,
        clip_values=(0.0, 255.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    return classifierNN1, classifierNN2


def main():
    parser = argparse.ArgumentParser(description="Run adversarial attacks on classifiers.")
    parser.add_argument("--attack", type=str, default="fgsm", choices=["fgsm", "bim", "pgd", "df", "cw"], help="Type of attack to run")
    parser.add_argument("--targeted", type=bool, default=False, help="Run a targeted attack")
    args = parser.parse_args()
    
    # Attacco selezionato
    print(f"Selected attack: {args.attack}")
    print(f"Targeted attack: {args.targeted}")
    
    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup dei classificatori
    classifierNN1, classifierNN2 = setup_classifiers(device)

    # Caricamento del test_set
    test_set = get_test_set()
    images, labels = test_set.get_images()

    # Preprocessing delle immagini per il secondo classificatore
    images_nn2 = process_images(images, use_padding=False)

    # Calcolo dell'accuracy sulle immagini clean rispetto alle label vere
    accuracy_nn1_clean = compute_accuracy(classifierNN1, images, labels)
    print(f"Accuracy del classificatore NN1 su dati clean: {accuracy_nn1_clean}")
    accuracy_nn2_clean = compute_accuracy(classifierNN2, images_nn2, labels)
    print(f"Accuracy del classificatore NN2 su dati clean: {accuracy_nn2_clean}")

    # Calcolo della targeted accuracy sulle immagini clean rispetto alle label della classe target
    target_class_label = "Cristiano_Ronaldo"
    target_class = test_set.get_true_label(target_class_label)
    targeted_labels = target_class * torch.ones(labels.size, dtype=torch.long)
    targeted_accuracy_clean_nn1 = compute_accuracy(classifierNN1, images, targeted_labels)
    print(f"Targeted accuracy del classificatore NN1 su dati clean: {targeted_accuracy_clean_nn1}")
    targeted_accuracy_clean_nn2 = compute_accuracy(classifierNN2, images_nn2, targeted_labels)
    print(f"Targeted accuracy del classificatore NN2 su dati clean: {targeted_accuracy_clean_nn2}")

    # Avvio dell'attacco selezionato
    if args.attack == "fgsm":
        run_fgsm(classifierNN1, classifierNN2, images, labels, accuracy_nn1_clean, accuracy_nn2_clean, args.targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, target_class, detectors=None)
    elif args.attack == "bim":
        run_bim(classifierNN1, classifierNN2, images, labels, accuracy_nn1_clean, accuracy_nn2_clean, args.targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, target_class, detectors=None)
    elif args.attack == "pgd":
        run_pgd(classifierNN1, classifierNN2, images, labels, accuracy_nn1_clean, accuracy_nn2_clean, args.targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, target_class, detectors=None)
    elif args.attack == "df":
        classifierNN1, _ = setup_classifiers(device, classify=False)
        run_df(classifierNN1, classifierNN2, images, labels, accuracy_nn1_clean, accuracy_nn2_clean, detectors=None)
    elif args.attack == "cw":
        run_cw(classifierNN1, classifierNN2, images, labels, accuracy_nn1_clean, accuracy_nn2_clean, args.targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, target_class, detectors=None)


if __name__ == "__main__":
    main()