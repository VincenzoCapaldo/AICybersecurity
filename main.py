from nets import get_NN1, get_NN2
import torch
from torch.optim import Adam
from art.estimators.classification import PyTorchClassifier
from dataset import get_test_set
from utils import compute_accuracy, plot_accuracy, process_images, show_image
from attacks import fgsm, bim, pgd, deepfool, carlini_wagner

NUM_CLASSES = 8631

def main(device):
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

    # Carico il test_set
    test_set = get_test_set()
    dataloader, test_images_nn1, test_labels = test_set.get_images()
    #test_images_nn2 = process_images(test_images_nn1, use_padding=True)  # Preprocesso le immagini per il secondo classificatore
    #show_image(test_images_nn2[4])
    #print(f"Test images shape for NN1: {test_images_nn1.shape}")
    #print(f"Test images shape for NN2: {test_images_nn2.shape}")

    # calcolo delle performance dei classificatori sui dati clean
    accuracy_nn1_clean = compute_accuracy(classifierNN1, test_images_nn1, test_labels)
    print(f"Accuracy del classificatore NN1 su dati clean: {accuracy_nn1_clean}")
    #accuracy_nn2_clean = compute_accuracy(classifierNN2, test_images_nn2, test_labels)
    #print(f"Accuracy del classificatore NN2 su dati clean: {accuracy_nn2_clean}")


    ### Non-targeted (error-generic) FGSM attack

    # Calcolo dell'accuracy al variare di epsilon e della perturbazione media
    epsilon_values = [0.00, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    accuracies, average_perturbations, _ = fgsm(classifierNN1, epsilon_values, test_images_nn1, test_labels, False)
    plot_accuracy("FGSM Non-targeted - Accuracy vs Epsilon and Average Perturbation", "Epsilon", epsilon_values, average_perturbations, accuracies)

    ### Targeted (error-specific) FGSM Attack

    # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon e della perturbazione media (con la target_class fissato)
    epsilon_values = [0.00, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    target_class = [test_set.get_true_label("Cristiano_Ronaldo")]
    accuracies, average_perturbations, targeted_accuracy = fgsm(classifierNN1, epsilon_values, test_images_nn1, test_labels, True, target_class)
    plot_accuracy("FGSM Targeted - Accuracy vs Epsilon and Average Perturbation", "Epsilon", epsilon_values, average_perturbations, accuracies, True, targeted_accuracy)

    # Calcolo dell'accuracy e della targeted accuracy al variare della classe target (con epsilon fissato)
    epsilon = [0.05]
    target_class_values = test_set.get_used_labels()
    accuracies, average_perturbations, targeted_accuracy = fgsm(classifierNN1, epsilon, test_images_nn1, test_labels, True, target_class_values)
    plot_accuracy("FGSM Targeted - Accuracy vs Target Class and Average Perturbation", "Target Class", target_class_values, average_perturbations, accuracies, True, targeted_accuracy)

    ### Non-targeted (error-generic) BIM attack

    # Calcolo dell'accuracy al variare di epsilon e della perturbazione media (con epsilon_step e epsilon_step fissati)
    epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    epsilon_step = [0.01]
    max_iter = [5]
    accuracies, average_perturbations, _ = bim(classifierNN1, epsilon_values, epsilon_step, max_iter, test_images_nn1, test_labels, False)
    plot_accuracy("BIM Non-targeted - Accuracy vs Epsilon and Average Perturbation", "Epsilon", epsilon_values, average_perturbations, accuracies)

    # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione media (con epsilon e max_iter fissati)
    epsilon = [0.025]
    epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
    max_iter = [5]
    accuracies, average_perturbations, _ = bim(classifierNN1, epsilon, epsilon_step_values, max_iter, test_images_nn1, test_labels, False)
    plot_accuracy("BIM Non-targeted - Accuracy vs Epsilon Step and Average Perturbation", "Epsilon Step", epsilon_step_values, average_perturbations, accuracies)

    # Calcolo dell'accuracy al variare di max_iter e della perturbazione media (con epsilon e epsilon_step fissati)
    epsilon = [0.025]
    epsilon_step = [0.01]
    max_iter_values = [1, 2, 5, 10, 20]
    accuracies, average_perturbations, _ = bim(classifierNN1, epsilon, epsilon_step, max_iter_values, test_images_nn1, test_labels, False)
    plot_accuracy("BIM Non-targeted - Accuracy vs Max Iterations and Average Perturbation", "Max Iterations", max_iter_values, average_perturbations, accuracies)

    ### Targeted (error-specific) BIM attack

    # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon e della perturbazione media (con epsilon_step, max_iter e target_class fissati)
    epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    epsilon_step = [0.01]
    max_iter = [5]
    target_class = test_set.get_true_label("Cristiano_Ronaldo")
    accuracies, average_perturbations, targeted_accuracy = bim(classifierNN1, epsilon_values, epsilon_step, max_iter, test_images_nn1, test_labels, True, target_class)
    plot_accuracy("BIM Targeted - Accuracy vs Epsilon and Average Perturbation", "Epsilon", epsilon_values, average_perturbations, accuracies, True, targeted_accuracy)

    # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon_step e della perturbazione media (con epsilon, max_iter e target_class fissati)
    epsilon = [0.025]
    epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
    max_iter = [5]
    target_class = test_set.get_true_label("Cristiano_Ronaldo")
    accuracies, average_perturbations, targeted_accuracy = bim(classifierNN1, epsilon, epsilon_step_values, max_iter, test_images_nn1, test_labels, True, target_class)
    plot_accuracy("BIM Targeted - Accuracy vs Epsilon Step and Average Perturbation", "Epsilon Step", epsilon_step_values, average_perturbations, accuracies, True, targeted_accuracy)

    # Calcolo dell'accuracy e della targeted accuracy al variare di max_iter e della perturbazione media (con epsilon, epsilon_step e target_class fissati)
    epsilon = [0.025]
    epsilon_step = [0.01]
    max_iter_values = [1, 2, 5, 10, 20]
    target_class = test_set.get_true_label("Cristiano_Ronaldo")
    accuracies, average_perturbations, targeted_accuracy = bim(classifierNN1, epsilon, epsilon_step, max_iter_values, test_images_nn1, test_labels, True, target_class)
    plot_accuracy("BIM Targeted - Accuracy vs Max Iterations and Average Perturbation", "Max Iterations", max_iter_values, average_perturbations, accuracies, True, targeted_accuracy)

    # Calcolo dell'accuracy e della targeted accuracy al variare della classe target e della perturbazione media (con epsilon, epsilon_step e max_iter fissati)
    epsilon = [0.025]
    epsilon_step = [0.01]
    max_iter = [5]
    target_class_values = test_set.get_used_labels()
    accuracies, average_perturbations, targeted_accuracy = bim(classifierNN1, epsilon, epsilon_step, max_iter, test_images_nn1, test_labels, True, target_class_values)
    plot_accuracy("BIM Targeted - Accuracy vs Target Class and Average Perturbation", "Target Class", target_class_values, average_perturbations, accuracies, True, targeted_accuracy)

    ### Non-targeted (error-generic) PGD attack

    # Calcolo dell'accuracy al variare di epsilon e della perturbazione media (con epsilon_step e epsilon_step fissati)
    epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    epsilon_step = [0.01]
    max_iter = [5]
    accuracies, average_perturbations, _ = pgd(classifierNN1, epsilon_values, epsilon_step, max_iter, test_images_nn1, test_labels, False)
    plot_accuracy("PGD Non-targeted - Accuracy vs Epsilon and Average Perturbation", "Epsilon", epsilon_values, average_perturbations, accuracies)

    # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione media (con epsilon e max_iter fissati)
    epsilon = [0.025]
    epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
    max_iter = [5]
    accuracies, average_perturbations, _ = pgd(classifierNN1, epsilon, epsilon_step_values, max_iter, test_images_nn1, test_labels, False)
    plot_accuracy("PGD Non-targeted - Accuracy vs Epsilon Step and Average Perturbation", "Epsilon Step", epsilon_step_values, average_perturbations, accuracies)

    # Calcolo dell'accuracy al variare di max_iter e della perturbazione media (con epsilon e epsilon_step fissati)
    epsilon = [0.025]
    epsilon_step = [0.01]
    max_iter_values = [1, 2, 5, 10, 20]
    accuracies, average_perturbations, _ = pgd(classifierNN1, epsilon, epsilon_step, max_iter_values, test_images_nn1, test_labels, False)
    plot_accuracy("PGD Non-targeted - Accuracy vs Max Iterations and Average Perturbation", "Max Iterations", max_iter_values, average_perturbations, accuracies)

    ### Targeted (error-specific) PGD attack

    # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon e della perturbazione media (con epsilon_step, max_iter e target_class fissati)
    epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    epsilon_step = [0.01]
    max_iter = [5]
    target_class = test_set.get_true_label("Cristiano_Ronaldo")
    accuracies, average_perturbations, targeted_accuracy = pgd(classifierNN1, epsilon_values, epsilon_step, max_iter, test_images_nn1, test_labels, True, target_class)
    plot_accuracy("PGD Targeted - Accuracy vs Epsilon and Average Perturbation", "Epsilon", epsilon_values, average_perturbations, accuracies, True, targeted_accuracy)

    # Calcolo dell'accuracy e della targeted accuracy al variare di epsilon_step e della perturbazione media (con epsilon, max_iter e target_class fissati)
    epsilon = [0.025]
    epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
    max_iter = [5]
    target_class = test_set.get_true_label("Cristiano_Ronaldo")
    accuracies, average_perturbations, targeted_accuracy = pgd(classifierNN1, epsilon, epsilon_step_values, max_iter, test_images_nn1, test_labels, True, target_class)
    plot_accuracy("PGD Targeted - Accuracy vs Epsilon Step and Average Perturbation", "Epsilon Step", epsilon_step_values, average_perturbations, accuracies, True, targeted_accuracy)

    # Calcolo dell'accuracy e della targeted accuracy al variare di max_iter e della perturbazione media (con epsilon, epsilon_step e target_class fissati)
    epsilon = [0.025]
    epsilon_step = [0.01]
    max_iter_values = [1, 2, 5, 10, 20]
    target_class = test_set.get_true_label("Cristiano_Ronaldo")
    accuracies, average_perturbations, targeted_accuracy = pgd(classifierNN1, epsilon, epsilon_step, max_iter_values, test_images_nn1, test_labels, True, target_class)
    plot_accuracy("PGD Targeted - Accuracy vs Max Iterations and Average Perturbation", "Max Iterations", max_iter_values, average_perturbations, accuracies, True, targeted_accuracy)

    # Calcolo dell'accuracy e della targeted accuracy al variare della classe target e della perturbazione media (con epsilon, epsilon_step e max_iter fissati)
    epsilon = [0.025]
    epsilon_step = [0.01]
    max_iter = [5]
    target_class_values = test_set.get_used_labels()
    accuracies, average_perturbations, targeted_accuracy = pgd(classifierNN1, epsilon, epsilon_step, max_iter, test_images_nn1, test_labels, True, target_class_values)
    plot_accuracy("PGD Targeted - Accuracy vs Target Class and Average Perturbation", "Target Class", target_class_values, average_perturbations, accuracies, True, targeted_accuracy)

    ### Non-targeted (error-generic) DeepFool attack

    # Calcolo dell'accuracy al variare di epsilon e della perturbazione media (con max_iter fissato)
    epsilon_values = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    max_iter = [5]
    accuracies, average_perturbations, _ = deepfool(classifierNN1, epsilon_values, max_iter, test_images_nn1, test_labels)
    plot_accuracy("DeepFool Non-targeted - Accuracy vs Epsilon and Average Perturbation", "Epsilon", epsilon_values, average_perturbations, accuracies)

    # Calcolo dell'accuracy al variare del numero di iterazioni e della perturbazione media (con epsilon fissato)
    epsilon = [0.05]
    max_iter_values = [1, 2, 5, 10, 20]
    accuracies, average_perturbations, _ = deepfool(classifierNN1, epsilon, max_iter_values, test_images_nn1, test_labels)
    plot_accuracy("DeepFool Non-targeted - Accuracy vs Max Iterations and Average Perturbation", "Max Iterations", max_iter_values, average_perturbations, accuracies)

    ### Non-targeted (error-generic) Carlini-Wagner attack

    # Calcolo dell'accuracy al variare della confidence e della perturbazione media (con max_iter e learning_rate fissati)
    confidence_values = [0.1, 0.5, 1, 2, 5, 10]
    max_iter = [5]
    learning_rate = [0.01]
    accuracies, average_perturbations, _ = carlini_wagner(classifierNN1, confidence_values, max_iter, learning_rate, test_images_nn1, test_labels)
    plot_accuracy("Carlini-Wagner Non-targeted - Accuracy vs Confidence and Average Perturbation", "Confidence", confidence_values, average_perturbations, accuracies)

    # Calcolo dell'accuracy al variare di max_iter e della perturbazione media (con confidence e learning_rate fissati)
    confidence = [0.5]
    max_iter_values = [1, 2, 5, 7, 10]
    learning_rate = [0.01]
    accuracies, average_perturbations, _ = carlini_wagner(classifierNN1, confidence, max_iter_values, learning_rate, test_images_nn1, test_labels)
    plot_accuracy("Carlini-Wagner Non-targeted - Accuracy vs Max Iterations and Average Perturbation", "Max Iterations", max_iter_values, average_perturbations, accuracies)

    # Calcolo dell'accuracy al variare del learning_rate e della perturbazione media (con confidence e max_iter fissati)
    confidence = [0.5]
    max_iter = [5]
    learning_rate_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    accuracies, average_perturbations, _ = carlini_wagner(classifierNN1, confidence, max_iter, learning_rate_values, test_images_nn1, test_labels)
    plot_accuracy("Carlini-Wagner Non-targeted - Accuracy vs Learning Rate and Average Perturbation", "Learning Rate", learning_rate_values, average_perturbations, accuracies)

    ### Targeted (error-specific) Carlini-Wagner attack
    
    # Calcolo dell'accuracy e della targeted accuracy al variare di confidence e della perturbazione media (con max_iter, learning_rate, e target_class fissati)
    confidence_values =  [0.1, 0.5, 1, 2, 5, 10]
    max_iter = [5]
    learning_rate = [0.01]
    target_class = test_set.get_true_label("Cristiano_Ronaldo")
    accuracies, average_perturbations, targeted_accuracy = carlini_wagner(classifierNN1, confidence_values, max_iter, learning_rate, test_images_nn1, test_labels, True, target_class)
    plot_accuracy("Carlini-Wagner Targeted - Accuracy vs Confidence and Average Perturbation", "Confidence", confidence_values, average_perturbations, accuracies, True, targeted_accuracy)

    # Calcolo dell'accuracy e della targeted accuracy al variare di max_iter e della perturbazione media (con confidence, learning_rate e target_class fissati)
    confidence = [0.5]
    max_iter_values = [1, 2, 5, 7, 10]
    learning_rate = [0.01]
    target_class = test_set.get_true_label("Cristiano_Ronaldo")
    accuracies, average_perturbations, targeted_accuracy = carlini_wagner(classifierNN1, confidence, max_iter_values, learning_rate, test_images_nn1, test_labels, True, target_class)
    plot_accuracy("Carlini-Wagner Targeted - Accuracy vs Max Iterations and Average Perturbation", "Max Iterations", max_iter_values, average_perturbations, accuracies, True, targeted_accuracy)

    # Calcolo dell'accuracy e della targeted accuracy al variare di learning_rate e della perturbazione media (con confidence, max_iter e target_class fissati)
    confidence = [0.5]
    max_iter = [5]
    learning_rate_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    target_class = test_set.get_true_label("Cristiano_Ronaldo")
    accuracies, average_perturbations, targeted_accuracy = carlini_wagner(classifierNN1, confidence, max_iter, learning_rate_values, test_images_nn1, test_labels, True, target_class)
    plot_accuracy("Carlini-Wagner Targeted - Accuracy vs Learning Rate and Average Perturbation", "Learning Rate", learning_rate_values, average_perturbations, accuracies, True, targeted_accuracy)

    # Calcolo dell'accuracy e della targeted accuracy al variare della classe target (con confidence, max_iter e learning_rate fissati)
    confidence = [0.5]
    max_iter = [5]
    learning_rate = [0.01]
    target_class_values = test_set.get_used_labels()
    accuracies, average_perturbations, targeted_accuracy = carlini_wagner(classifierNN1, confidence, max_iter, learning_rate, test_images_nn1, test_labels, True, target_class_values)
    plot_accuracy("Carlini-Wagner Targeted - Accuracy vs Target Class and Average Perturbation", "Target Class", target_class_values, average_perturbations, accuracies, True, targeted_accuracy)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    main(device)