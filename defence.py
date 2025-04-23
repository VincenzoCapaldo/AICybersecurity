import argparse
import numpy as np
from nets import get_NN1, get_detector
from art.defences.detector.evasion import BinaryInputDetector
import torch
from torch.optim import Adam
from scipy.special import softmax
from art.estimators.classification import PyTorchClassifier
from dataset import get_dataset
from utils import *
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, DeepFool, CarliniLInfMethod


NUM_CLASSES = 8631  # Numero di classi nel dataset VGGFace2


def train_test_split(images, labels, test_size=0.2, shuffle=True, random_seed=2025):
    """
    Divide immagini e etichette in train e test set.

    Parametri:
        images (np.ndarray): Array delle immagini.
        labels (np.ndarray): Array delle etichette.
        test_size (float): Percentuale del dataset da usare come test (es. 0.2 per 20%).
        shuffle (bool): Se True, mescola i dati prima dello split.
        random_seed (int, opzionale): Seed per riproducibilità.

    Ritorna:
        train_images, train_labels, test_images, test_labels
    """
    assert images.shape[0] == labels.shape[0], "Numero di immagini ed etichette non corrisponde."
    
    num_samples = images.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    split_idx = int(num_samples * (1 - test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_images = images[train_idx]
    train_labels = labels[train_idx]
    test_images = images[test_idx]
    test_labels = labels[test_idx]

    return train_images, train_labels, test_images, test_labels


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
        optimizer=Adam(detector.parameters(), lr=0.001),
        input_shape=(3, 160, 160),
        channels_first=True,
        nb_classes=2,
        clip_values=(-0.5, 0.5),
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    return classifier


def generate_adversarial_examples(classifier, attack_type, x_test):
    epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    confidence_values = [0.1, 0.5, 1, 5, 10]  # Valori per cw
    targeted = False

    n_samples = x_test.shape[0]
    adv_examples = []

    if attack_type in ["fgsm", "bim", "pgd", "df"]:
        split_size = n_samples // len(epsilon_values)
    elif attack_type == "cw":
        split_size = n_samples // len(confidence_values)

    if attack_type == "fgsm":
        for i, eps in enumerate(epsilon_values):
            start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values) - 1 else n_samples
            x_subset = x_test[start:end]
            attack = FastGradientMethod(estimator=classifier, eps=eps, targeted=targeted)
            adv_examples.append(attack.generate(x=x_subset))

    elif attack_type == "bim":
        for i, eps in enumerate(epsilon_values):
            start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values) - 1 else n_samples
            x_subset = x_test[start:end]
            attack = BasicIterativeMethod(estimator=classifier, eps=eps, eps_step=0.005, max_iter=10)
            adv_examples.append(attack.generate(x=x_subset))

    elif attack_type == "pgd":
        for i, eps in enumerate(epsilon_values):
            start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values) - 1 else n_samples
            x_subset = x_test[start:end]
            attack = ProjectedGradientDescent(estimator=classifier, eps=eps, eps_step=0.005, max_iter=10)
            adv_examples.append(attack.generate(x=x_subset))

    elif attack_type == "df":
        for i, eps in enumerate(epsilon_values):
            start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values) - 1 else n_samples
            x_subset = x_test[start:end]
            attack = DeepFool(classifier=classifier, epsilon = eps, max_iter=5, batch_size=16)
            adv_examples.append(attack.generate(x=x_subset))

    elif attack_type == "cw":
        for i, conf in enumerate(confidence_values):
            start, end = i * split_size, (i + 1) * split_size if i < len(confidence_values) - 1 else n_samples
            x_subset = x_test[start:end]
            attack = CarliniLInfMethod(
                classifier=classifier,
                confidence=conf,
                max_iter=5,
                learning_rate=0.01,
                batch_size=8
            )
            adv_examples.append(attack.generate(x=x_subset))

    x_test_adv = np.concatenate(adv_examples, axis=0)
    return x_test_adv


def compute_accuracy_with_detectors(classifier, x_test, y_test, y_adv, detectors, threshold=0.5):
    """
    Calcola l'accuracy penalizzando i falsi positivi dei detector.
    - classifier: il classificatore (con metodo predict).
    - x_clean, y_clean: dati NON adversariali.
    - y_adv: etichette per i campioni avversari (True/False).
    - detectors: dict di detector ART.
    - threshold: soglia per considerare un campione come avversario.
    Ritorna: (accuracy_effettiva, n_samples_utili, n_falsi_positivi)
    """
    # Maschera di campioni rifiutati da almeno un detector
    rejected_mask = np.zeros(x_test.shape[0], dtype=bool)

    for name, detector in detectors.items():
        report, _ = detector.detect(x_test)
        logits = np.array(report["predictions"])  # shape (n_samples, 2)
        probs = softmax(logits, axis=1)
        adversarial_probs = probs[:, 1]
        is_adversarial = adversarial_probs > threshold # True se avversario; False se pulito
        rejected_mask = np.logical_or(is_adversarial, rejected_mask)  # Un campione è scartato se almeno un detector lo scarta
        detection_error = np.logical_xor(is_adversarial, y_adv)
        print(f"Detector {name} ha scartato {np.sum(is_adversarial)} campioni (soglia={threshold}).")
        print(f"Detector {name} ha rilevato erroneamente {np.sum(detection_error)} campioni (soglia={threshold}).")
        print(f"Detector {name} ha rilevato correttamente {x_test.shape[0] - np.sum(detection_error)} campioni (soglia={threshold}).")

    accepted_mask = np.logical_not(rejected_mask)  # Inverti la maschera: True se accettato, False se scartato
    
    # Campioni accettati = quelli che passeranno al classificatore
    x_pass = x_test[accepted_mask]
    y_pass = y_test[accepted_mask]

    # Predizioni del classificatore
    y_pred = classifier.predict(x_pass)
    y_pred_labels = np.argmax(y_pred, axis=1)

    n_total = y_test.shape[0]
    n_correct = np.sum(y_pred_labels == y_pass)  # campioni correttamente classificati
    
    is_adversarial = ~accepted_mask  # Campioni avversari: quelli scartati dai detector

    # Falsi positivi: campioni puliti scartati dai detector (0,1)
    n_fp = np.sum(np.logical_and(is_adversarial, ~y_adv))
    
    # Campioni correttamente scartati (1,1)
    n_correct_discarded = np.sum(np.logical_and(is_adversarial, y_adv)) # Veri positivi
    
    # Accuracy: corrette / totale originario (quindi penalizza falsi positivi)
    accuracy = (n_correct + n_correct_discarded) / n_total

    return accuracy, n_fp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_detectors', type=bool, default=False, help='Se True, addestra i detector; altrimenti carica i modelli salvati')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold per le rilevazioni dei detector')
    parser.add_argument("--attack", type=str, default="fgsm", choices=["fgsm", "bim", "pgd", "df", "cw"], help="Type of attack to run")
    parser.add_argument("--targeted", type=bool, default=True, help="Run a targeted attack")
    args = parser.parse_args()

    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup dei classificatori
    classifierNN1= setup_classifier(device)

    # Carica le immagini e le etichette
    images, labels = get_dataset().get_images()

    # Divisione in train e test set 80% - 20%
    train_images, train_labels, test_images, test_labels = train_test_split(images, labels, test_size=0.2)

    # Calcolo dell'accuracy sulle immagini clean rispetto alle label vere
    accuracy_nn1_clean = compute_accuracy(classifierNN1, train_images, train_labels)
    print(f"Accuracy del classificatore NN1 su dati clean: {accuracy_nn1_clean}")

    # Directory per i modelli
    os.makedirs("./models", exist_ok=True)
    attack_types = {"fgsm", "bim", "pgd"}

    # Train or load Detectors
    detectors = {}
    nb_train = train_images.shape[0]
    for attack_type in attack_types:
        model_path = os.path.join("./models", f"{attack_type}_detector.pth")
        detector_classifier = setup_detector_classifier(device)
        if args.train_detectors:
            print(f"Training detector for attack: {attack_type}")
            detectors[attack_type] = BinaryInputDetector(detector_classifier)
            if attack_type == "df":
                classifier = setup_classifier(device, classify=False)
            else:
                classifier = setup_classifier(device, classify=True)
            # Train the detector
            x_train_adv = generate_adversarial_examples(classifier, attack_type, train_images)
            x_train_detector = np.concatenate((train_images, x_train_adv), axis=0)
            y_train_detector = np.concatenate((np.array([[1, 0]] * nb_train), np.array([[0, 1]] * nb_train)), axis=0)
            detectors[attack_type].fit(x_train_detector, y_train_detector, nb_epochs=20, batch_size=16)
            
            # Salvataggio dello state_dict del modello
            torch.save(detector_classifier.model.state_dict(), model_path)
            print(f"Detector salvato in: {model_path}")
        else:
            print(f"Caricamento del detector per attack: {attack_type}")
            detector_classifier.model.load_state_dict(torch.load(model_path, map_location=device))
            detectors[attack_type] = BinaryInputDetector(detector_classifier)
            print(f"Detector caricato da: {model_path}")

    # Valutare detectors + classifier sui dati clean del test set
    nb_test = test_images.shape[0]
    adv_labels = np.zeros(nb_test, dtype=bool) # Tutti i campioni sono puliti (classe 0)
    accuracy, fp = compute_accuracy_with_detectors(classifierNN1, test_images, test_labels, adv_labels, detectors, threshold=args.threshold)
    print(f"Accuracy del classificatore col filtraggio dei detectors: {accuracy:.4f}")
    print(f"Numero di immagini scartate dai detectors (FP): {fp}")

    # Valutare detectors + classifier sui dati adversarial
    adv_labels = np.ones(nb_test, dtype=bool) # Tutti i campioni sono adversarial (classe 1)
    if args.attack == "fgsm":
        run_fgsm(classifierNN1, None, test_images, test_labels, test_set, accuracy_nn1_clean, accuracy_nn2_clean, args.targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, [target_class])
    elif args.attack == "bim":
        run_bim(classifierNN1, None, test_images, test_labels, test_set, accuracy_nn1_clean, accuracy_nn2_clean, args.targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, [target_class])
    elif args.attack == "pgd":
        run_pgd(classifierNN1, None, test_images, test_labels, test_set, accuracy_nn1_clean, accuracy_nn2_clean, args.targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, [target_class])
    elif args.attack == "df":
        classifierNN1, _ = setup_classifiers(device, classify=False)
        run_df(classifierNN1, None, test_images, test_labels, accuracy_nn1_clean, accuracy_nn2_clean)
    elif args.attack == "cw":
        run_cw(classifierNN1, None, test_images, test_labels, test_set, accuracy_nn1_clean, accuracy_nn2_clean, args.targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, [target_class])


if __name__ == "__main__":
    main()