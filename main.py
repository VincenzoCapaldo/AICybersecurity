import argparse
from nets import get_NN1, get_NN2
import torch
from torch.optim import Adam
from art.estimators.classification import PyTorchClassifier
from dataset import get_test_set
from utils import compute_accuracy, plot_accuracy, process_images, show_image
from attacks import fgsm, bim, pgd, deepfool, carlini_wagner

NUM_CLASSES = 8631

def setup_classifiers(device):
    # Istanzio le reti
    nn1 = get_NN1(device)
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
        clip_values=(0.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    return classifierNN1, classifierNN2

        
def run_fgsm(classifier, test_images, test_labels, test_set, targeted=False):
    ### Non-targeted (error-generic) FGSM attack
    if not targeted:
        # Calcolo dell'accuracy al variare di epsilon e della perturbazione media
        epsilon_values = [0.00, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
        accuracies, max_perturbations, _ = fgsm(classifier, epsilon_values, test_images, test_labels, False)
        plot_accuracy("FGSM Non-targeted - Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies)

    ### Targeted (error-specific) FGSM Attack
    else:
        # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon e della perturbazione media (con la target_class fissato)
        epsilon_values = [0.00, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
        target_class = [test_set.get_true_label("Cristiano_Ronaldo")]
        accuracies, max_perturbations, targeted_accuracy = fgsm(classifier, epsilon_values, test_images, test_labels, True, target_class)
        plot_accuracy("FGSM Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies, True, targeted_accuracy)

        # Calcolo dell'accuracy e della targeted accuracy al variare della classe target (con epsilon fissato)
        epsilon = [0.05]
        target_class_values = test_set.get_used_labels()
        accuracies, max_perturbations, targeted_accuracy = fgsm(classifier, epsilon, test_images, test_labels, True, target_class_values)
        plot_accuracy("FGSM Targeted - Accuracy and Targeted Accuracy vs Target Class and Max Perturbations", "Target Class", target_class_values, max_perturbations, accuracies, True, targeted_accuracy)


def run_bim(classifier, test_images, test_labels, test_set, targeted=False):
    ### Non-targeted (error-generic) BIM attack
    if not targeted:
        # Calcolo dell'accuracy al variare di epsilon e della perturbazione media (con epsilon_step e epsilon_step fissati)
        epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        epsilon_step = [0.01]
        max_iter = [5]
        accuracies, max_perturbations, _ = bim(classifier, epsilon_values, epsilon_step, max_iter, test_images, test_labels, False)
        plot_accuracy("BIM Non-targeted - Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies)

        # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione media (con epsilon e max_iter fissati)
        epsilon = [0.025]
        epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
        max_iter = [5]
        accuracies, max_perturbations, _ = bim(classifier, epsilon, epsilon_step_values, max_iter, test_images, test_labels, False)
        plot_accuracy("BIM Non-targeted - Accuracy vs Epsilon Step and Max Perturbations", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies)

        # Calcolo dell'accuracy al variare di max_iter e della perturbazione media (con epsilon e epsilon_step fissati)
        epsilon = [0.025]
        epsilon_step = [0.01]
        max_iter_values = [1, 2, 5, 10, 20]
        accuracies, max_perturbations, _ = bim(classifier, epsilon, epsilon_step, max_iter_values, test_images, test_labels, False)
        plot_accuracy("BIM Non-targeted - Accuracy vs Max Iterations and Max Perturbations", "Max Iterations", max_iter_values, max_perturbations, accuracies)

    ### Targeted (error-specific) BIM attack
    else:
        # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon e della perturbazione media (con epsilon_step, max_iter e target_class fissati)
        epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        epsilon_step = [0.01]
        max_iter = [5]
        target_class = test_set.get_true_label("Cristiano_Ronaldo")
        accuracies, max_perturbations, targeted_accuracy = bim(classifier, epsilon_values, epsilon_step, max_iter, test_images, test_labels, True, target_class)
        plot_accuracy("BIM Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies, True, targeted_accuracy)

        # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon_step e della perturbazione media (con epsilon, max_iter e target_class fissati)
        epsilon = [0.025]
        epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
        max_iter = [5]
        target_class = test_set.get_true_label("Cristiano_Ronaldo")
        accuracies, max_perturbations, targeted_accuracy = bim(classifier, epsilon, epsilon_step_values, max_iter, test_images, test_labels, True, target_class)
        plot_accuracy("BIM Targeted - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies, True, targeted_accuracy)

        # Calcolo dell'accuracy e della targeted accuracy al variare di max_iter e della perturbazione media (con epsilon, epsilon_step e target_class fissati)
        epsilon = [0.025]
        epsilon_step = [0.01]
        max_iter_values = [1, 2, 5, 10, 20]
        target_class = test_set.get_true_label("Cristiano_Ronaldo")
        accuracies, max_perturbations, targeted_accuracy = bim(classifier, epsilon, epsilon_step, max_iter_values, test_images, test_labels, True, target_class)
        plot_accuracy("BIM Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations", "Max Iterations", max_iter_values, max_perturbations, accuracies, True, targeted_accuracy)

        # Calcolo dell'accuracy e della targeted accuracy al variare della classe target e della perturbazione media (con epsilon, epsilon_step e max_iter fissati)
        epsilon = [0.025]
        epsilon_step = [0.01]
        max_iter = [5]
        target_class_values = test_set.get_used_labels()
        accuracies, max_perturbations, targeted_accuracy = bim(classifier, epsilon, epsilon_step, max_iter, test_images, test_labels, True, target_class_values)
        plot_accuracy("BIM Targeted - Accuracy and Targeted Accuracy vs Target Class and Max Perturbations", "Target Class", target_class_values, max_perturbations, accuracies, True, targeted_accuracy)


def run_pgd(classifier, test_images, test_labels, test_set, targeted=False):
    ### Non-targeted (error-generic) PGD attack
    if not targeted:
        # Calcolo dell'accuracy al variare di epsilon e della perturbazione media (con epsilon_step e epsilon_step fissati)
        epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        epsilon_step = [0.01]
        max_iter = [5]
        accuracies, max_perturbations, _ = pgd(classifier, epsilon_values, epsilon_step, max_iter, test_images, test_labels, False)
        plot_accuracy("PGD Non-targeted - Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies)

        # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione media (con epsilon e max_iter fissati)
        epsilon = [0.025]
        epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
        max_iter = [5]
        accuracies, max_perturbations, _ = pgd(classifier, epsilon, epsilon_step_values, max_iter, test_images, test_labels, False)
        plot_accuracy("PGD Non-targeted - Accuracy vs Epsilon Step and Max Perturbations", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies)

        # Calcolo dell'accuracy al variare di max_iter e della perturbazione media (con epsilon e epsilon_step fissati)
        epsilon = [0.025]
        epsilon_step = [0.01]
        max_iter_values = [1, 2, 5, 10, 20]
        accuracies, max_perturbations, _ = pgd(classifier, epsilon, epsilon_step, max_iter_values, test_images, test_labels, False)
        plot_accuracy("PGD Non-targeted - Accuracy vs Max Iterations and Max Perturbations", "Max Iterations", max_iter_values, max_perturbations, accuracies)

    ### Targeted (error-specific) PGD attack
    else:
        # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon e della perturbazione media (con epsilon_step, max_iter e target_class fissati)
        epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        epsilon_step = [0.01]
        max_iter = [5]
        target_class = test_set.get_true_label("Cristiano_Ronaldo")
        accuracies, max_perturbations, targeted_accuracy = pgd(classifier, epsilon_values, epsilon_step, max_iter, test_images, test_labels, True, target_class)
        plot_accuracy("PGD Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies, True, targeted_accuracy)

        # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon_step e della perturbazione media (con epsilon, max_iter e target_class fissati)
        epsilon = [0.025]
        epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
        max_iter = [5]
        target_class = test_set.get_true_label("Cristiano_Ronaldo")
        accuracies, max_perturbations, targeted_accuracy = pgd(classifier, epsilon, epsilon_step_values, max_iter, test_images, test_labels, True, target_class)
        plot_accuracy("PGD Targeted - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies, True, targeted_accuracy)

        # Calcolo dell'accuracy e della targeted accuracy al variare di max_iter e della perturbazione media (con epsilon, epsilon_step e target_class fissati)
        epsilon = [0.025]
        epsilon_step = [0.01]
        max_iter_values = [1, 2, 5, 10, 20]
        target_class = test_set.get_true_label("Cristiano_Ronaldo")
        accuracies, max_perturbations, targeted_accuracy = pgd(classifier, epsilon, epsilon_step, max_iter_values, test_images, test_labels, True, target_class)
        plot_accuracy("PGD Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations", "Max Iterations", max_iter_values, max_perturbations, accuracies, True, targeted_accuracy)

        # Calcolo dell'accuracy e della targeted accuracy al variare della classe target e della perturbazione media (con epsilon, epsilon_step e max_iter fissati)
        epsilon = [0.025]
        epsilon_step = [0.01]
        max_iter = [5]
        target_class_values = test_set.get_used_labels()
        accuracies, max_perturbations, targeted_accuracy = pgd(classifier, epsilon, epsilon_step, max_iter, test_images, test_labels, True, target_class_values)
        plot_accuracy("PGD Targeted - Accuracy and Targeted Accuracy vs Target Class and Max Perturbations", "Target Class", target_class_values, max_perturbations, accuracies, True, targeted_accuracy)


def run_df(classifier, test_images, test_labels):
    ### Non-targeted (error-generic) DeepFool attack

    # Calcolo dell'accuracy al variare di epsilon e della perturbazione media (con max_iter fissato)
    epsilon_values = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    max_iter = [5]
    accuracies, max_perturbations, _ = deepfool(classifier, epsilon_values, max_iter, test_images, test_labels)
    plot_accuracy("DeepFool Non-targeted - Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies)

    # Calcolo dell'accuracy al variare del numero di iterazioni e della perturbazione media (con epsilon fissato)
    epsilon = [0.05]
    max_iter_values = [1, 2, 5, 10, 20]
    accuracies, max_perturbations, _ = deepfool(classifier, epsilon, max_iter_values, test_images, test_labels)
    plot_accuracy("DeepFool Non-targeted - Accuracy vs Max Iterations and Max Perturbations", "Max Iterations", max_iter_values, max_perturbations, accuracies)


def run_cw(classifier, test_images, test_labels, test_set, targeted=False):
    ### Non-targeted (error-generic) Carlini-Wagner attack
    if not targeted:
        # Calcolo dell'accuracy al variare della confidence e della perturbazione media (con max_iter e learning_rate fissati)
        confidence_values = [0.1, 0.5, 1, 2, 5, 10]
        max_iter = [5]
        learning_rate = [0.01]
        accuracies, max_perturbations, _ = carlini_wagner(classifier, confidence_values, max_iter, learning_rate, test_images, test_labels)
        plot_accuracy("Carlini-Wagner Non-targeted - Accuracy vs Confidence and Max Perturbations", "Confidence", confidence_values, max_perturbations, accuracies)

        # Calcolo dell'accuracy al variare di max_iter e della perturbazione media (con confidence e learning_rate fissati)
        confidence = [0.5]
        max_iter_values = [1, 2, 5, 7, 10]
        learning_rate = [0.01]
        accuracies, max_perturbations, _ = carlini_wagner(classifier, confidence, max_iter_values, learning_rate, test_images, test_labels)
        plot_accuracy("Carlini-Wagner Non-targeted - Accuracy vs Max Iterations and Max Perturbations", "Max Iterations", max_iter_values, max_perturbations, accuracies)

        # Calcolo dell'accuracy al variare del learning_rate e della perturbazione media (con confidence e max_iter fissati)
        confidence = [0.5]
        max_iter = [5]
        learning_rate_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        accuracies, max_perturbations, _ = carlini_wagner(classifier, confidence, max_iter, learning_rate_values, test_images, test_labels)
        plot_accuracy("Carlini-Wagner Non-targeted - Accuracy vs Learning Rate and Max Perturbations", "Learning Rate", learning_rate_values, max_perturbations, accuracies)

    ### Targeted (error-specific) Carlini-Wagner attack
    else:
        # Calcolo dell'accuracy e della targeted accuracy al variare di confidence e della perturbazione media (con max_iter, learning_rate, e target_class fissati)
        confidence_values =  [0.1, 0.5, 1, 2, 5, 10]
        max_iter = [5]
        learning_rate = [0.01]
        target_class = test_set.get_true_label("Cristiano_Ronaldo")
        accuracies, max_perturbations, targeted_accuracy = carlini_wagner(classifier, confidence_values, max_iter, learning_rate, test_images, test_labels, True, target_class)
        plot_accuracy("Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Confidence and Max Perturbations", "Confidence", confidence_values, max_perturbations, accuracies, True, targeted_accuracy)

        # Calcolo dell'accuracy e della targeted accuracy al variare di max_iter e della perturbazione media (con confidence, learning_rate e target_class fissati)
        confidence = [0.5]
        max_iter_values = [1, 2, 5, 7, 10]
        learning_rate = [0.01]
        target_class = test_set.get_true_label("Cristiano_Ronaldo")
        accuracies, max_perturbations, targeted_accuracy = carlini_wagner(classifier, confidence, max_iter_values, learning_rate, test_images, test_labels, True, target_class)
        plot_accuracy("Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations", "Max Iterations", max_iter_values, max_perturbations, accuracies, True, targeted_accuracy)

        # Calcolo dell'accuracy e della targeted accuracy al variare di learning_rate e della perturbazione media (con confidence, max_iter e target_class fissati)
        confidence = [0.5]
        max_iter = [5]
        learning_rate_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        target_class = test_set.get_true_label("Cristiano_Ronaldo")
        accuracies, max_perturbations, targeted_accuracy = carlini_wagner(classifier, confidence, max_iter, learning_rate_values, test_images, test_labels, True, target_class)
        plot_accuracy("Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Learning Rate and Max Perturbations", "Learning Rate", learning_rate_values, max_perturbations, accuracies, True, targeted_accuracy)

        # Calcolo dell'accuracy e della targeted accuracy al variare della classe target (con confidence, max_iter e learning_rate fissati)
        confidence = [0.5]
        max_iter = [5]
        learning_rate = [0.01]
        target_class_values = test_set.get_used_labels()
        accuracies, max_perturbations, targeted_accuracy = carlini_wagner(classifier, confidence, max_iter, learning_rate, test_images, test_labels, True, target_class_values)
        plot_accuracy("Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Target Class and Max Perturbations", "Target Class", target_class_values, max_perturbations, accuracies, True, targeted_accuracy)


def main():
    parser = argparse.ArgumentParser(description="Run adversarial attacks on classifiers.")
    parser.add_argument("--attack", type=str, default="cw", choices=["fgsm", "bim", "pgd", "df", "cw"], help="Type of attack to run")
    parser.add_argument("--targeted", type=bool, default=False, help="Run a targeted attack")
    args = parser.parse_args()
    
    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected attack: {args.attack}")
    print(f"Targeted attack: {args.targeted}")

    # Setup dei classificatori
    classifierNN1, classifierNN2 = setup_classifiers(device)

    # Carico il test_set
    test_set = get_test_set()
    _, test_images, test_labels = test_set.get_images()

    #test_images_nn2 = process_images(test_images, use_padding=True)  # Preprocesso le immagini per il secondo classificatore
    #show_image(test_images_nn2[4])
    #print(f"Test images shape for NN1: {test_images.shape}")
    #print(f"Test images shape for NN2: {test_images_nn2.shape}")

    # Calcolo delle performance dei classificatori sui dati clean
    accuracy_nn1_clean = compute_accuracy(classifierNN1, test_images, test_labels)
    print(f"Accuracy del classificatore NN1 su dati clean: {accuracy_nn1_clean}")
    #accuracy_nn2_clean = compute_accuracy(classifierNN2, test_images_nn2, test_labels)
    #print(f"Accuracy del classificatore NN2 su dati clean: {accuracy_nn2_clean}")

    # Avvia l'attacco selezionato
    if args.attack == "fgsm":
        run_fgsm(classifierNN1, test_images, test_labels, test_set, args.targeted)
    elif args.attack == "bim":
        run_bim(classifierNN1, test_images, test_labels, test_set, args.targeted)
    elif args.attack == "pgd":
        run_pgd(classifierNN1, test_images, test_labels, test_set, args.targeted)
    elif args.attack == "df":
        run_df(classifierNN1, test_images, test_labels)
    elif args.attack == "cw":
        run_cw(classifierNN1, test_images, test_labels, test_set, args.targeted)


if __name__ == "__main__":
    main()