from utils import *
from attacks import FGSM, BIM, PGD, DF, CW


def run_fgsm(classifierNN1, classifierNN2, test_images, test_labels, accuracy_clean_nn1, accuracy_clean_nn2, targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, target_class, detectors):
    attack = FGSM(test_images, test_labels, classifierNN1, classifierNN2, detectors)
    
    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima
    epsilon_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    accuracies, max_perturbations, targeted_accuracy = attack.compute_security_curve(epsilon_values, targeted, target_class)
    epsilon_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    if classifierNN2 is not None:
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
    if detectors is not None:
        label = "NN1 + Detector"
    else:
        label = "NN1"
    if targeted:
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        plot_accuracy(f"{label} FGSM Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        if classifierNN2 is not None:
            targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)
            plot_accuracy("(NN2) FGSM Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])
    else:
        plot_accuracy(f"{label} FGSM Non-targeted - Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"])
        if classifierNN2 is not None:
            plot_accuracy("(NN2) FGSM Non-targeted - Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"])
    

def run_bim(classifierNN1, classifierNN2, test_images, test_labels, accuracy_clean_nn1, accuracy_clean_nn2, targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, target_class, detectors):
    attack = BIM(test_images, test_labels, classifierNN1, classifierNN2, detectors)
    
    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con epsilon_step e epsilon_step fissati)
    epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    epsilon_step = [0.005]
    max_iter = [10]
    accuracies, max_perturbations, targeted_accuracy = attack.compute_security_curve(epsilon_values, epsilon_step, max_iter, targeted, target_class)
    epsilon_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    if classifierNN2 is not None:
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
    if targeted:
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        plot_accuracy(f"(NN1) BIM Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        if classifierNN2 is not None:
            targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)
            plot_accuracy(f"(NN2) BIM Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])
    else:
        plot_accuracy(f"(NN1) BIM Non-targeted - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"])
        if classifierNN2 is not None:
            plot_accuracy(f"(NN2) BIM Non-targeted - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"])
      
    # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione massima (con epsilon e max_iter fissati)
    epsilon = [0.05]
    epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
    max_iter = [10]
    accuracies, max_perturbations, targeted_accuracy = attack.compute_security_curve(epsilon, epsilon_step_values, max_iter, targeted, target_class)
    epsilon_step_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    if classifierNN2 is not None:
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
    if targeted:
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        plot_accuracy(f"(NN1) BIM Targeted - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        if classifierNN2 is not None:
            targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)
            plot_accuracy(f"(NN2) BIM Targeted - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])
    else:
        plot_accuracy(f"(NN1) BIM Non-targeted - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn1"])
        if classifierNN2 is not None:
            plot_accuracy(f"(NN2) BIM Non-targeted - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn2"])

    # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con epsilon e epsilon_step fissati)
    epsilon = [0.05]
    epsilon_step = [0.005]
    max_iter_values = [1, 3, 5, 7, 10]
    accuracies, max_perturbations, targeted_accuracy = attack.compute_security_curve(epsilon, epsilon_step_values, max_iter, targeted, target_class)
    max_iter_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    if classifierNN2 is not None:
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
    if targeted:
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        plot_accuracy(f"(NN2) BIM Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])
        if classifierNN2 is not None:
            targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)
            plot_accuracy(f"(NN1) BIM Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
    else:
        plot_accuracy(f"(NN1) BIM Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"])
        if classifierNN2 is not None:
            plot_accuracy(f"(NN2) BIM Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"])


def run_pgd(classifierNN1, classifierNN2, test_images, test_labels, accuracy_clean_nn1, accuracy_clean_nn2, targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, target_class, detectors):
    attack = PGD(test_images, test_labels, classifierNN1, classifierNN2, detectors)
    
    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con epsilon_step e epsilon_step fissati)
    epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    epsilon_step = [0.005]
    max_iter = [10]
    accuracies, max_perturbations, targeted_accuracy = attack.compute_security_curve(epsilon_values, epsilon_step, max_iter, targeted, target_class)
    epsilon_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    if classifierNN2 is not None:
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
    if targeted:
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        plot_accuracy(f"(NN1) PGD Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        if classifierNN2 is not None:        
            targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)
            plot_accuracy(f"(NN2) PGD Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])
    else:
        plot_accuracy(f"(NN1) PGD Non-targeted - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"])
        if classifierNN2 is not None:
            plot_accuracy(f"(NN2) PGD Non-targeted - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"])
      
    # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione massima (con epsilon e max_iter fissati)
    epsilon = [0.05]
    epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
    max_iter = [10]
    accuracies, max_perturbations, targeted_accuracy = attack.compute_security_curve(epsilon, epsilon_step_values, max_iter, targeted, target_class)
    epsilon_step_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    if classifierNN2 is not None:
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
    if targeted:
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        plot_accuracy(f"(NN1) PGD Targeted - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        if classifierNN2 is not None:
            targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)
            plot_accuracy(f"(NN2) PGD Targeted - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])
    else:
        plot_accuracy(f"(NN1) PGD Non-targeted - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn1"])
        if classifierNN2 is not None:
            plot_accuracy(f"(NN2) PGD Non-targeted - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies["nn2"])

    # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con epsilon e epsilon_step fissati)
    epsilon = [0.05]
    epsilon_step = [0.005]
    max_iter_values = [1, 3, 5, 7, 10]
    accuracies, max_perturbations, targeted_accuracy = attack.compute_security_curve(epsilon, epsilon_step_values, max_iter, targeted, target_class)
    max_iter_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    if classifierNN2 is not None:
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
    if targeted:
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        plot_accuracy(f"(NN1) PGD Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        if classifierNN2 is not None:
            targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)
            plot_accuracy(f"(NN2) PGD Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])
    else:
        plot_accuracy(f"(NN1) PGD Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"])
        if classifierNN2 is not None:
            plot_accuracy(f"(NN2) PGD Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"])


def run_df(classifierNN1, classifierNN2, test_images, test_labels, accuracy_clean_nn1, accuracy_clean_nn2, detectors):
    attack = DF(test_images, test_labels, classifierNN1, classifierNN2, detectors)
    # Nota: nella libreria ART non è implementata la versione targeted di DeepFool.
    
    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con max_iter fissato)
    epsilon_values = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    max_iter = [5]
    accuracies, max_perturbations = attack.compute_security_curve(epsilon_values, max_iter)
    epsilon_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    plot_accuracy(f"(NN1) DeepFool Non-targeted - Accuracy vs Epsilon and Max Perturbations (Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn1"])
    if classifierNN2 is not None:
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        plot_accuracy(f"(NN2) DeepFool Non-targeted - Accuracy vs Epsilon and Max Perturbations (Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies["nn2"])

    # Calcolo dell'accuracy al variare del numero di iterazioni e della perturbazione massima (con epsilon fissato)
    epsilon = [0.05]
    max_iter_values = [1, 3, 5, 7, 10]
    accuracies, max_perturbations = attack.compute_security_curve(epsilon, max_iter_values)
    max_iter_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    plot_accuracy(f"(NN1) DeepFool Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"])
    if classifierNN2 is not None:
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
        plot_accuracy(f"(NN2) DeepFool Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"])


def run_cw(classifierNN1, classifierNN2, test_images, test_labels, accuracy_clean_nn1, accuracy_clean_nn2, targeted, targeted_accuracy_clean_nn1, targeted_accuracy_clean_nn2, target_class, detectors):
    attack = CW(test_images, test_labels, classifierNN1, classifierNN2, detectors)

    # Calcolo dell'accuracy al variare della confidence e della perturbazione massima (con max_iter e learning_rate fissati)
    confidence_values = [0.1, 0.5, 1, 2, 5, 10]
    max_iter = [5]
    learning_rate = [0.01]
    accuracies, max_perturbations, targeted_accuracy = attack.compute_security_curve(confidence_values, max_iter, learning_rate, targeted, target_class)
    confidence_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    if classifierNN2 is not None:
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
    if targeted:
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        plot_accuracy(f"(NN1) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Confidence and Max Perturbations (Max_iter={max_iter}; Learning_rate={learning_rate})", "Confidence", confidence_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        if classifierNN2 is not None:
            targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)
            plot_accuracy(f"(NN2) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Confidence and Max Perturbations (Max_iter={max_iter}; Learning_rate={learning_rate})", "Confidence", confidence_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])
    else:
        plot_accuracy(f"(NN1) Carlini-Wagner Non-targeted - Accuracy vs Confidence and Max Perturbations (Max_iter={max_iter}; Learning_rate={learning_rate})", "Confidence", confidence_values, max_perturbations, accuracies["nn1"])
        if classifierNN2 is not None:
            plot_accuracy(f"(NN2) Carlini-Wagner Non-targeted - Accuracy vs Confidence and Max Perturbations (Max_iter={max_iter}; Learning_rate={learning_rate})", "Confidence", confidence_values, max_perturbations, accuracies["nn2"])

    # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con confidence e learning_rate fissati)
    confidence = [0.5]
    max_iter_values = [1, 3, 5, 7, 10]
    learning_rate = [0.01]
    accuracies, max_perturbations, targeted_accuracy = attack.compute_security_curve(confidence_values, max_iter, learning_rate, targeted, target_class)
    max_iter_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    if classifierNN2 is not None:
        accuracies["nn2"].insert(0, accuracy_clean_nn2)
    if targeted:
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        plot_accuracy(f"(NN1) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        if classifierNN2 is not None:
            targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)
            plot_accuracy(f"(NN2) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])
    else:
        plot_accuracy(f"(NN1) Carlini-Wagner Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn1"])
        if classifierNN2 is not None:
            plot_accuracy(f"(NN2) Carlini-Wagner Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})", "Max Iterations", max_iter_values, max_perturbations, accuracies["nn2"])
    
    # Calcolo dell'accuracy al variare del learning_rate e della perturbazione massima (con confidence e max_iter fissati)
    confidence = [0.5]
    max_iter = [5]
    learning_rate_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    accuracies, max_perturbations, targeted_accuracy = attack.compute_security_curve(confidence_values, max_iter, learning_rate, targeted, target_class)
    learning_rate_values.insert(0, 0.0)
    max_perturbations.insert(0, 0.0)
    accuracies["nn1"].insert(0, accuracy_clean_nn1)
    if classifierNN2 is not None:
        accuracies["nn2"].insert(0, accuracy_clean_nn2)  
    if targeted:
        targeted_accuracy["nn1"].insert(0, targeted_accuracy_clean_nn1)
        plot_accuracy(f"(NN1) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})", "Learning Rate", learning_rate_values, max_perturbations, accuracies["nn1"], targeted, targeted_accuracy["nn1"])
        if classifierNN2 is not None:
            targeted_accuracy["nn2"].insert(0, targeted_accuracy_clean_nn2)
            plot_accuracy(f"(NN2) Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})", "Learning Rate", learning_rate_values, max_perturbations, accuracies["nn2"], targeted, targeted_accuracy["nn2"])
    else:
        plot_accuracy(f"(NN1) Carlini-Wagner Non-targeted - Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})", "Learning Rate", learning_rate_values, max_perturbations, accuracies["nn1"])
        if classifierNN2 is not None:
            plot_accuracy(f"(NN2) Carlini-Wagner Non-targeted - Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})", "Learning Rate", learning_rate_values, max_perturbations, accuracies["nn2"])
        

def plot_accuracy(title, x_title, x, max_perturbations, accuracies, targeted=False, targeted_accuracies=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    # Accuracy and Targeted Accuracy vs x
    axes[0].plot(x, accuracies, marker='o', linestyle='-', color='b')
    if targeted:
        axes[0].plot(x, targeted_accuracies, marker='o', linestyle='-', color='r')
        axes[0].legend(["Accuracy", "Targeted Accuracy"], loc="upper right")
    else:
        axes[0].legend(["Accuracy"], loc="upper right")
    axes[0].set_xlabel(x_title)
    axes[0].grid()

    # Accuracy and Targeted Accuracy vs Max Perturbations
    axes[1].plot(max_perturbations, accuracies, marker='o', linestyle='-', color='b')
    if targeted:
        axes[1].plot(max_perturbations, targeted_accuracies, marker='o', linestyle='-', color='r')
        axes[1].legend(["Accuracy", "Targeted Accuracy"], loc="upper right")
    else:
        axes[1].legend(["Accuracy"], loc="upper right")
    axes[1].set_xlabel("Max Perturbations")
    axes[1].axvline(x=0.05, color='red', linestyle='--', linewidth=1.5) # vincolo da rispettare
    axes[1].grid()
    
    plt.tight_layout()
    filename = title.replace(".",",")+ ".png"
    save_path = os.path.join("./plot", filename)
    plt.savefig(save_path)
    print(f"Plot {title}.png salvato.")