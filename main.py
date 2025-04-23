import argparse
from gg import FGSM
from nets import get_NN1, get_NN2
import torch
from torch.optim import Adam
from art.estimators.classification import PyTorchClassifier
from dataset import get_test_set
from utils import *
from attacks import fgsm, bim, pgd, deepfool, carlini_wagner

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

        
def run_fgsm(classifierNN1, classifierNN2, test_images, test_labels, accuracy_clean_nn1, accuracy_clean_nn2, targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, target_class):
    attack = FGSM(test_images, test_labels, classifierNN1, classifierNN2)
    epsilon_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    accuracies, max_perturbations, targeted_accuracy = attack.compute_security_curve(epsilon_values, targeted=targeted, target_class=target_class)
    
    ### Non-targeted (error-generic) FGSM attack
    if not targeted:
        # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima
        epsilon_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
        accuracies, max_perturbations, _ = fgsm(classifierNN1, classifierNN2, epsilon_values, test_images, test_labels)
        epsilon_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        plot_accuracy("(NN1) FGSM Non-targeted - Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"])
        plot_accuracy("(NN2) FGSM Non-targeted - Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"])

    ### Targeted (error-specific) FGSM Attack
    else:
        # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon e della perturbazione massima (con target_class fissato)
        epsilon_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
        accuracies, max_perturbations, targeted_accuracy = fgsm(classifierNN1, classifierNN2, epsilon_values, test_images, test_labels, targeted, target_class)
        epsilon_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)
        plot_accuracy("(NN1) FGSM Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        plot_accuracy("(NN2) FGSM Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])

def run_bim(classifierNN1, classifierNN2, test_images, test_labels, test_set, accuracy_clean_nn1, accuracy_clean_nn2, targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, target_class):
    ### Non-targeted (error-generic) BIM attack
    if not targeted:
        # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con epsilon_step e epsilon_step fissati)
        epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        epsilon_step = [0.005]
        max_iter = [10]
        accuracies, max_perturbations, _ = bim(classifierNN1, classifierNN2, epsilon_values, epsilon_step, max_iter, test_images, test_labels)
        epsilon_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        plot_accuracy(f"(NN1) BIM Non-targeted - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"])
        plot_accuracy(f"(NN1) BIM Non-targeted - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"])

        # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione massima (con epsilon e max_iter fissati)
        epsilon = [0.05]
        epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
        max_iter = [10]
        accuracies, max_perturbations, _ = bim(classifierNN1, classifierNN2, epsilon, epsilon_step_values, max_iter, test_images, test_labels)
        epsilon_step_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        plot_accuracy(f"(NN1) BIM Non-targeted - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn1"])
        plot_accuracy(f"(NN2) BIM Non-targeted - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn2"])

        # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con epsilon e epsilon_step fissati)
        epsilon = [0.05]
        epsilon_step = [0.005]
        max_iter_values = [1, 3, 5, 7, 10]
        accuracies, max_perturbations, _ = bim(classifierNN1, classifierNN2, epsilon, epsilon_step, max_iter_values, test_images, test_labels)
        max_iter_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        plot_accuracy(f"(NN1) BIM Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"])
        plot_accuracy(f"(NN2) BIM Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"])

    ### Targeted (error-specific) BIM attack
    else:
        # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon e della perturbazione massima (con epsilon_step, max_iter e target_class fissati)
        epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        epsilon_step = [0.005]
        max_iter = [10]
        accuracies, max_perturbations, targeted_accuracy = bim(classifierNN1, classifierNN2, epsilon_values, epsilon_step, max_iter, test_images, test_labels, targeted, target_class)
        epsilon_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)
        plot_accuracy(f"(NN1) BIM Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        plot_accuracy(f"(NN2) BIM Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])

        # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon_step e della perturbazione massima (con epsilon, max_iter e target_class fissati)
        epsilon = [0.05]
        epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
        max_iter = [10]
        accuracies, max_perturbations, targeted_accuracy = bim(classifierNN1, classifierNN2, epsilon, epsilon_step_values, max_iter, test_images, test_labels, targeted, target_class)
        epsilon_step_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)
        plot_accuracy(f"(NN1) BIM Targeted - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        plot_accuracy(f"(NN2) BIM Targeted - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])

        # Calcolo dell'accuracy e della targeted accuracy al variare di max_iter e della perturbazione massima (con epsilon, epsilon_step e target_class fissati)
        epsilon = [0.05]
        epsilon_step = [0.005]
        max_iter_values = [1, 3, 5, 7, 10]
        accuracies, max_perturbations, targeted_accuracy = bim(classifierNN1, classifierNN2, epsilon, epsilon_step, max_iter_values, test_images, test_labels, targeted, target_class)
        max_iter_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)    
        plot_accuracy(f"(NN1) BIM Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        plot_accuracy(f"(NN2) BIM Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])

        # Calcolo dell'accuracy e della targeted accuracy al variare della classe target e della perturbazione massima (con epsilon, epsilon_step e max_iter fissati)
        epsilon = [0.05]
        epsilon_step = [0.005]
        max_iter = [10]
        target_class_values = test_set.get_used_labels()
        accuracies, max_perturbations, targeted_accuracy = bim(classifierNN1, classifierNN2, epsilon, epsilon_step, max_iter, test_images, test_labels, targeted, target_class_values)
        plot_accuracy("(NN1) BIM Targeted - Accuracy and Targeted Accuracy vs Target Class and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Target Class", target_class_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        plot_accuracy("(NN2) BIM Targeted - Accuracy and Targeted Accuracy vs Target Class and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Target Class", target_class_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])


def run_pgd(classifierNN1, classifierNN2, test_images, test_labels, test_set, accuracy_clean_nn1, accuracy_clean_nn2, targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, target_class):
    ### Non-targeted (error-generic) PGD attack
    if not targeted:
        # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con epsilon_step e epsilon_step fissati)
        epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        epsilon_step = [0.05]
        max_iter = [10]
        accuracies, max_perturbations, _ = pgd(classifierNN1, classifierNN2, epsilon_values, epsilon_step, max_iter, test_images, test_labels)
        epsilon_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        plot_accuracy(f"(NN1) PGD Non-targeted - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"])
        plot_accuracy(f"(NN2) PGD Non-targeted - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"])

        # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione massima (con epsilon e max_iter fissati)
        epsilon = [0.05]
        epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
        max_iter = [10]
        accuracies, max_perturbations, _ = pgd(classifierNN1, classifierNN2, epsilon, epsilon_step_values, max_iter, test_images, test_labels)
        epsilon_step_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)        
        plot_accuracy(f"(NN1) PGD Non-targeted - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn1"])
        plot_accuracy(f"(NN2) PGD Non-targeted - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn2"])

        # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con epsilon e epsilon_step fissati)
        epsilon = [0.05]
        epsilon_step = [0.005]
        max_iter_values = [1, 3, 5, 7, 10]
        accuracies, max_perturbations, _ = pgd(classifierNN1, classifierNN2, epsilon, epsilon_step, max_iter_values, test_images, test_labels)
        max_iter_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)        
        plot_accuracy(f"(NN1) PGD Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"])
        plot_accuracy(f"(NN2) PGD Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"])

    ### Targeted (error-specific) PGD attack
    else:
        # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon e della perturbazione massima (con epsilon_step, max_iter e target_class fissati)
        epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        epsilon_step = [0.05]
        max_iter = [10]
        accuracies, max_perturbations, targeted_accuracy = pgd(classifierNN1, classifierNN2, epsilon_values, epsilon_step, max_iter, test_images, test_labels, targeted, target_class)
        epsilon_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)          
        plot_accuracy(f"(NN1) PGD Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        plot_accuracy(f"(NN2) PGD Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])

        # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon_step e della perturbazione massima (con epsilon, max_iter e target_class fissati)
        epsilon = [0.05]
        epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
        max_iter = [10]
        accuracies, max_perturbations, targeted_accuracy = pgd(classifierNN1, classifierNN2, epsilon, epsilon_step_values, max_iter, test_images, test_labels, targeted, target_class)
        epsilon_step_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)         
        plot_accuracy(f"(NN1) PGD Targeted - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        plot_accuracy(f"(NN2) PGD Targeted - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])

        # Calcolo dell'accuracy e della targeted accuracy al variare di max_iter e della perturbazione massima (con epsilon, epsilon_step e target_class fissati)
        epsilon = [0.05]
        epsilon_step = [0.005]
        max_iter_values = [1, 3, 5, 7, 10]
        accuracies, max_perturbations, targeted_accuracy = pgd(classifierNN1, classifierNN2, epsilon, epsilon_step, max_iter_values, test_images, test_labels, targeted, target_class)
        max_iter_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)         
        plot_accuracy(f"(NN1) PGD Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        plot_accuracy(f"(NN2) PGD Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])

        # Calcolo dell'accuracy e della targeted accuracy al variare della classe target e della perturbazione massima (con epsilon, epsilon_step e max_iter fissati)
        epsilon = [0.05]
        epsilon_step = [0.005]
        max_iter = [10]
        target_class_values = test_set.get_used_labels()
        accuracies, max_perturbations, targeted_accuracy = pgd(classifierNN1, classifierNN2, epsilon, epsilon_step, max_iter, test_images, test_labels, targeted, target_class_values) 
        plot_accuracy("(NN1) PGD Targeted - Accuracy and Targeted Accuracy vs Target Class and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Target Class", target_class_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        plot_accuracy("(NN2) PGD Targeted - Accuracy and Targeted Accuracy vs Target Class and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Target Class", target_class_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])


def run_df(classifierNN1, classifierNN2, test_images, test_labels, accuracy_clean_nn1, accuracy_clean_nn2):
    # Nota: nella libreria ART non è implementata la versione targeted di DeepFool.
    
    ### Non-targeted (error-generic) DeepFool attack

    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con max_iter fissato)
    epsilon_values = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    max_iter = [5]
    accuracies, max_perturbations, _ = deepfool(classifierNN1, classifierNN2, epsilon_values, max_iter, test_images, test_labels)
    epsilon_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    accuracies["nn2"].insert(0, accuracy_clean_nn2)
    plot_accuracy(f"(NN1) DeepFool Non-targeted - Accuracy vs Epsilon and Max Perturbations (Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"])
    plot_accuracy(f"(NN2) DeepFool Non-targeted - Accuracy vs Epsilon and Max Perturbations (Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"])

    # Calcolo dell'accuracy al variare del numero di iterazioni e della perturbazione massima (con epsilon fissato)
    epsilon = [0.05]
    max_iter_values = [1, 3, 5, 7, 10]
    accuracies, max_perturbations, _ = deepfool(classifierNN1, classifierNN2, epsilon, max_iter_values, test_images, test_labels)
    max_iter_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    accuracies["nn2"].insert(0, accuracy_clean_nn2)
    plot_accuracy(f"(NN1) DeepFool Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"])
    plot_accuracy(f"(NN2) DeepFool Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"])


def run_cw(classifierNN1, classifierNN2, test_images, test_labels, test_set, accuracy_clean_nn1, accuracy_clean_nn2, targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, target_class):
    ### Non-targeted (error-generic) Carlini-Wagner attack
    if not targeted:
        # Calcolo dell'accuracy al variare della confidence e della perturbazione massima (con max_iter e learning_rate fissati)
        confidence_values = [0.1, 0.5, 1, 2, 5, 10]
        max_iter = [5]
        learning_rate = [0.01]
        accuracies, max_perturbations, _ = carlini_wagner(classifierNN1, classifierNN2, confidence_values, max_iter, learning_rate, test_images, test_labels)
        confidence_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        plot_accuracy(f"(NN1) Carlini-Wagner Non-targeted - Accuracy vs Confidence and Max Perturbations (Max_iter={max_iter}; Learning_rate={learning_rate})", "Confidence", confidence_values, max_perturbations, accuracies["nn1"])
        plot_accuracy(f"(NN2) Carlini-Wagner Non-targeted - Accuracy vs Confidence and Max Perturbations (Max_iter={max_iter}; Learning_rate={learning_rate})", "Confidence", confidence_values, max_perturbations, accuracies["nn2"])

        # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con confidence e learning_rate fissati)
        confidence = [0.5]
        max_iter_values = [1, 3, 5, 7, 10]
        learning_rate = [0.01]
        accuracies, max_perturbations, _ = carlini_wagner(classifierNN1, classifierNN2, confidence, max_iter_values, learning_rate, test_images, test_labels)
        max_iter_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        plot_accuracy(f"(NN1) Carlini-Wagner Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"])
        plot_accuracy(f"(NN2) Carlini-Wagner Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"])

        # Calcolo dell'accuracy al variare del learning_rate e della perturbazione massima (con confidence e max_iter fissati)
        confidence = [0.5]
        max_iter = [5]
        learning_rate_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        accuracies, max_perturbations, _ = carlini_wagner(classifierNN1, classifierNN2, confidence, max_iter, learning_rate_values, test_images, test_labels)
        learning_rate_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)      
        plot_accuracy(f"(NN1) Carlini-Wagner Non-targeted - Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})", "Learning Rate", learning_rate_values, max_perturbations, accuracies["nn1"])
        plot_accuracy(f"(NN2) Carlini-Wagner Non-targeted - Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})", "Learning Rate", learning_rate_values, max_perturbations, accuracies["nn2"])

    ### Targeted (error-specific) Carlini-Wagner attack
    else:
        # Calcolo dell'accuracy e della targeted accuracy al variare di confidence e della perturbazione massima (con max_iter, learning_rate, e target_class fissati)
        confidence_values =  [0.1, 0.5, 1, 2, 5, 10]
        max_iter = [5]
        learning_rate = [0.01]
        accuracies, max_perturbations, targeted_accuracy = carlini_wagner(classifierNN1, classifierNN2, confidence_values, max_iter, learning_rate, test_images, test_labels, targeted, target_class)
        confidence_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)         
        plot_accuracy(f"(NN1) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Confidence and Max Perturbations (Max_iter={max_iter}; Learning_rate={learning_rate})", "Confidence", confidence_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        plot_accuracy(f"(NN2) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Confidence and Max Perturbations (Max_iter={max_iter}; Learning_rate={learning_rate})", "Confidence", confidence_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])

        # Calcolo dell'accuracy e della targeted accuracy al variare di max_iter e della perturbazione massima (con confidence, learning_rate e target_class fissati)
        confidence = [0.5]
        max_iter_values = [1, 3, 5, 7, 10]
        learning_rate = [0.01]
        accuracies, max_perturbations, targeted_accuracy = carlini_wagner(classifierNN1, classifierNN2, confidence, max_iter_values, learning_rate, test_images, test_labels, targeted, target_class)
        max_iter_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)         
        plot_accuracy(f"(NN1) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        plot_accuracy(f"(NN1) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])

        # Calcolo dell'accuracy e della targeted accuracy al variare di learning_rate e della perturbazione massima (con confidence, max_iter e target_class fissati)
        confidence = [0.5]
        max_iter = [5]
        learning_rate_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        accuracies, max_perturbations, targeted_accuracy = carlini_wagner(classifierNN1, classifierNN2, confidence, max_iter, learning_rate_values, test_images, test_labels, targeted, target_class)
        learning_rate_values.insert(0, 0.0)
        max_perturbations.insert(0, 0.0)
        accuracies["nn1"].insert(0, accuracy_clean_nn1)
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)         
        plot_accuracy(f"(NN1) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})", "Learning Rate", learning_rate_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        plot_accuracy(f"(NN2) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})", "Learning Rate", learning_rate_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])

        # Calcolo dell'accuracy e della targeted accuracy al variare della classe target (con confidence, max_iter e learning_rate fissati)
        confidence = [0.5]
        max_iter = [5]
        learning_rate = [0.01]
        target_class_values = test_set.get_used_labels()
        accuracies, max_perturbations, targeted_accuracy = carlini_wagner(classifierNN1, classifierNN2, confidence, max_iter, learning_rate, test_images, test_labels, targeted, target_class_values)
        plot_accuracy("(NN1) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Target Class and Max Perturbations (Confidence={confidence}; Max_iter={max_iter}; Learning_rate={learning_rate})", "Target Class", target_class_values, max_perturbations, accuracies["nn1"], accuracy_clean_nn1, targeted_accuracy_clean_nn1, targeted, targeted_accuracy["nn1"])
        plot_accuracy("(NN2) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Target Class and Max Perturbations (Confidence={confidence}; Max_iter={max_iter}; Learning_rate={learning_rate})", "Target Class", target_class_values, max_perturbations, accuracies["nn2"], accuracy_clean_nn2, targeted_accuracy_clean_nn2, targeted, targeted_accuracy["nn2"])


def main():
    parser = argparse.ArgumentParser(description="Run adversarial attacks on classifiers.")
    parser.add_argument("--attack", type=str, default="df", choices=["fgsm", "bim", "pgd", "df", "cw"], help="Type of attack to run")
    parser.add_argument("--targeted", type=bool, default=True, help="Run a targeted attack")
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
    test_images, test_labels = test_set.get_images()

    # Preprocessing delle immagini per il secondo classificatore
    test_images_nn2 = process_images(test_images, use_padding=False)

    # Calcolo dell'accuracy sulle immagini clean rispetto alle label vere
    accuracy_nn1_clean = compute_accuracy(classifierNN1, test_images, test_labels)
    print(f"Accuracy del classificatore NN1 su dati clean: {accuracy_nn1_clean}")
    accuracy_nn2_clean = compute_accuracy(classifierNN2, test_images_nn2, test_labels)
    print(f"Accuracy del classificatore NN2 su dati clean: {accuracy_nn2_clean}")

    # Calcolo della targeted accuracy sulle immagini clean rispetto alle label della classe target
    target_class_label = "Cristiano_Ronaldo"
    target_class = test_set.get_true_label(target_class_label)
    targeted_labels = target_class * torch.ones(test_labels.size, dtype=torch.long)
    targeted_accuracy_clean_nn1 = compute_accuracy(classifierNN1, test_images, targeted_labels)
    print(f"Targeted accuracy del classificatore NN1 su dati clean: {targeted_accuracy_clean_nn1}")
    targeted_accuracy_clean_nn2 = compute_accuracy(classifierNN2, test_images_nn2, targeted_labels)
    print(f"Targeted accuracy del classificatore NN2 su dati clean: {targeted_accuracy_clean_nn2}")

    # Avvio dell'attacco selezionato
    if args.attack == "fgsm":
        run_fgsm(classifierNN1, classifierNN2, test_images, test_labels, test_set, accuracy_nn1_clean, accuracy_nn2_clean, args.targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, [target_class])
    elif args.attack == "bim":
        run_bim(classifierNN1, classifierNN2, test_images, test_labels, test_set, accuracy_nn1_clean, accuracy_nn2_clean, args.targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, [target_class])
    elif args.attack == "pgd":
        run_pgd(classifierNN1, classifierNN2, test_images, test_labels, test_set, accuracy_nn1_clean, accuracy_nn2_clean, args.targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, [target_class])
    elif args.attack == "df":
        classifierNN1, _ = setup_classifiers(device, classify=False)
        run_df(classifierNN1, classifierNN2, test_images, test_labels, accuracy_nn1_clean, accuracy_nn2_clean)
    elif args.attack == "cw":
        run_cw(classifierNN1, classifierNN2, test_images, test_labels, test_set, accuracy_nn1_clean, accuracy_nn2_clean, args.targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, [target_class])


if __name__ == "__main__":
    main()