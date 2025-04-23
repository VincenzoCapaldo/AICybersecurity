import argparse
from nets import get_NN1, get_detector
from art.defences.detector.evasion import BinaryInputDetector
import torch
from torch.optim import Adam
from art.estimators.classification import PyTorchClassifier
from dataset import get_test_set
from utils import *
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, DeepFool, CarliniLInfMethod


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
            attack = DeepFool(estimator=classifier, epsilon = eps, max_iter=5)
            adv_examples.append(attack.generate(x=x_subset))

    elif attack_type == "cw":
        for i, conf in enumerate(confidence_values):
            start, end = i * split_size, (i + 1) * split_size if i < len(confidence_values) - 1 else n_samples
            x_subset = x_test[start:end]
            attack = CarliniLInfMethod(
                estimator=classifier,
                confidence=conf,
                max_iter=5,
                learning_rate=0.01
            )
            adv_examples.append(attack.generate(x=x_subset))

    x_test_adv = np.concatenate(adv_examples, axis=0)
    return x_test_adv


def main():
    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup dei classificatori
    classifierNN1= setup_classifier(device)

    # Caricamento del test_set
    test_set = get_test_set()
    _, test_images, test_labels = test_set.get_images()

    # Calcolo dell'accuracy sulle immagini clean rispetto alle label vere
    accuracy_nn1_clean = compute_accuracy(classifierNN1, test_images, test_labels)
    print(f"Accuracy del classificatore NN1 su dati clean: {accuracy_nn1_clean}")

    attack_types = {"fgsm", "bim", "pgd", "df", "cw"}

    # Train Detectors
    detectors = {}
    nb_train = test_images.shape[0]
    for attack_type in attack_types:
        print(f"Training detector for attack :{attack_type}")
        detector_classifier = setup_detector_classifier(device)
        detectors[attack_type] = BinaryInputDetector(detector_classifier)
        x_train_adv = generate_adversarial_examples(classifierNN1, attack_type, test_images)
        x_train_detector = np.concatenate((test_images, x_train_adv), axis=0)
        y_train_detector = np.concatenate((np.array([[1, 0]] * nb_train), np.array([[0, 1]] * nb_train)), axis=0)
        detectors[attack_type].fit(x_train_detector, y_train_detector, nb_epochs=20, batch_size=16)

if __name__ == "__main__":
    main()