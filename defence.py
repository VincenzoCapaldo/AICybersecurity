import argparse
import numpy as np
import os
from nets import get_NN1, get_detector
from art.defences.detector.evasion import BinaryInputDetector
import torch
from torch.optim import Adam
from art.estimators.classification import PyTorchClassifier
from dataset import get_test_set, get_train_set
from utils import *
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, DeepFool, CarliniLInfMethod
from security_evaluation_curve import run_fgsm, run_bim, run_pgd, run_df, run_cw

NUM_CLASSES = 8631  # Numero di classi nel dataset VGGFace2


def setup_classifier(device, classify=True):
    # Istanzio la rete
    nn1 = get_NN1(device, classify)

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
    return classifierNN1


def setup_detector_classifier(device):
    # Istanzio la rete
    detector = get_detector(device)

    # Definizione dei classificatori
    classifier = PyTorchClassifier(
        model=detector,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, detector.parameters()), 
    lr=1e-3
),
        input_shape=(3, 160, 160),
        channels_first=True,
        nb_classes=2,
        clip_values=(-0.5, 0.5),
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    return classifier


def generate_adversarial_train_set(classifier, attack_types, epsilon_values, confidence_values, save_dir = "./dataset/detectors_train_set/adversarial_examples"):
    # Training set di partenza, con immagini clean
    train_set = get_train_set()
    train_images = train_set.get_images()
    
    targeted = False
    n_samples = train_images.shape[0]

    # Per ogni attacco, calcola il numero di campioni da generare e genera i campioni avversari, salvandoli in una directory
    for attack_name in attack_types:
        if attack_name in ["fgsm", "bim", "pgd", "df"]:
            split_size = n_samples // len(epsilon_values)
        elif attack_name == "cw":
            split_size = n_samples // len(confidence_values)

        if attack_name == "fgsm":
            for i, eps in enumerate(epsilon_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values) - 1 else n_samples
                x_subset = train_images[start:end]
                attack = FastGradientMethod(estimator=classifier, eps=eps, targeted=targeted)
                adv_examples = attack.generate(x=x_subset)
                save_images_as_npy(adv_examples, f"eps_{eps}", save_dir + "/fgsm")

        elif attack_name == "bim":
            for i, eps in enumerate(epsilon_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values) - 1 else n_samples
                x_subset = train_images[start:end]
                attack = BasicIterativeMethod(estimator=classifier, eps=eps, eps_step=0.005, max_iter=10)
                adv_examples = attack.generate(x=x_subset)
                save_images_as_npy(adv_examples, f"eps_{eps}", save_dir + "/bim")

        elif attack_name == "pgd":
            for i, eps in enumerate(epsilon_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values) - 1 else n_samples
                x_subset = train_images[start:end]
                attack = ProjectedGradientDescent(estimator=classifier, eps=eps, eps_step=0.005, max_iter=10)
                adv_examples = attack.generate(x=x_subset)
                save_images_as_npy(adv_examples, f"eps_{eps}", save_dir + "/pgd")

        elif attack_name == "df":
            for i, eps in enumerate(epsilon_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values) - 1 else n_samples
                x_subset = train_images[start:end]
                attack = DeepFool(classifier=classifier, epsilon = eps, max_iter=5, batch_size=16)
                adv_examples = attack.generate(x=x_subset)
                save_images_as_npy(adv_examples, f"eps_{eps}", save_dir + "/df")

        elif attack_name == "cw":
            for i, conf in enumerate(confidence_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(confidence_values) - 1 else n_samples
                x_subset = train_images[start:end]
                attack = CarliniLInfMethod(
                    classifier=classifier,
                    confidence=conf,
                    max_iter=5,
                    learning_rate=0.01,
                    batch_size=8
                )
                adv_examples = attack.generate(x=x_subset)
                save_images_as_npy(adv_examples, f"confidence_{conf}", save_dir + "/cw")

        print (f"Dataset generato per l'attacco: {attack_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_train', type=bool, default=True, help='Se True, genera il train adv set; altrimenti no')
    parser.add_argument('--train_detectors', type=bool, default=True, help='Se True, addestra i detector; altrimenti carica i modelli salvati e procede con la valutazione')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold per le rilevazioni dei detector')
    parser.add_argument("--attack", type=str, default="fgsm", choices=["fgsm", "bim", "pgd", "df", "cw"], help="Type of attack to test")
    parser.add_argument("--targeted", type=bool, default=False, help="Test on targeted attack")
    args = parser.parse_args()

    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directory per i modelli
    os.makedirs("./models", exist_ok=True)

    # Setup dei classificatori
    classifierNN1= setup_classifier(device)

    # Generazione del training set avversario
    if args.generate_train:
        #attack_types = ["fgsm", "bim", "pgd", "df", "cw"]
        attack_types = ["df", "cw"]
        epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        confidence_values = [0.1, 0.5, 1, 5, 10]  # Valori per cw
        generate_adversarial_train_set(classifierNN1, attack_types, epsilon_values, confidence_values)


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
        # Carica le immagini e le etichette del test set
        test_set = get_test_set()
        test_images, test_labels = test_set.get_images()

        # Calcolo dell'accuracy sulle immagini clean rispetto alle label vere
        accuracy_nn1_clean = compute_accuracy(classifierNN1, test_images, test_labels)
        print(f"Accuracy del classificatore NN1 su dati clean: {accuracy_nn1_clean}")

        # Valutare detectors + classifier sui dati clean del test set
        nb_test = test_images.shape[0]
        adv_labels = np.zeros(nb_test, dtype=bool) # Tutti i campioni sono puliti (classe 0)
        accuracy_clean, fp = compute_accuracy_with_detectors(classifierNN1, test_images, test_labels, adv_labels, detectors, threshold=args.threshold, verbose=True)
        print(f"Accuracy del classificatore col filtraggio dei detectors: {accuracy_clean:.4f}")
        print(f"Numero di immagini scartate dai detectors (FP): {fp}")

        # Calcolo della targeted accuracy sulle immagini clean rispetto alle label della classe target
        target_class_label = "Cristiano_Ronaldo"
        target_class = test_set.get_true_label(target_class_label)
        targeted_labels = target_class * torch.ones(test_labels.size, dtype=torch.long)
        targeted_accuracy_clean, fp = compute_accuracy_with_detectors(classifierNN1, test_images, targeted_labels, adv_labels, detectors, threshold=args.threshold, verbose=True)
        print(f"Accuracy del classificatore col filtraggio dei detectors: {targeted_accuracy_clean:.4f}")
        print(f"Numero di immagini scartate dai detectors (FP): {fp}")

        # Valutare detectors + classifier sui dati adversarial
        if args.attack == "fgsm":
            run_fgsm(classifierNN1, None, test_images, test_labels, accuracy_clean, None, args.targeted, targeted_accuracy_clean, None, target_class, detectors, args.threshold)
        elif args.attack == "bim":
            run_bim(classifierNN1, None, test_images, test_labels, accuracy_clean, None, args.targeted, targeted_accuracy_clean, None, target_class, detectors, args.threshold)
        elif args.attack == "pgd":
            run_pgd(classifierNN1, None, test_images, test_labels, accuracy_clean, None, args.targeted, targeted_accuracy_clean, None, target_class, detectors, args.threshold)
        elif args.attack == "df":
            classifierNN1 = setup_classifier(device, classify=False)
            run_df(classifierNN1, None, test_images, test_labels, accuracy_clean, None, detectors, args.threshold)
        elif args.attack == "cw":
            run_cw(classifierNN1, None, test_images, test_labels, accuracy_clean, None, args.targeted, targeted_accuracy_clean, None, target_class, detectors, args.threshold)


if __name__ == "__main__":
    main()