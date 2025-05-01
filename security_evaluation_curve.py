from utils import *

def run_fgsm(classifier, name, targeted, test_set, accuracy_clean, targeted_accuracy_clean, target_class, detectors=None, threshold=0.05):
    images_dir = "./dataset/test_set/adversarial_examples/fgsm/"
    target_dir = "targeted" if targeted else "untargeted"
    load_dir = images_dir + target_dir
    imgs_adv = load_images_from_npy_folder(load_dir)

    clean_images, clean_labels = test_set.get_images()
    
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]

    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima
    epsilon_values = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore   
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
    if targeted:
        plot_accuracy(f"{name} FGSM Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} FGSM Non-targeted - Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies)


def run_bim(classifier, name, targeted, test_set, accuracy_clean, targeted_accuracy_clean, target_class, detectors=None, threshold=0.05):
    images_dir = "./dataset/test_set/adversarial_examples/bim/"
    target_dir = "targeted" if targeted else "untargeted"
    clean_images, clean_labels = test_set.get_images()

    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con epsilon_step e epsilon_step fissati)
    load_dir = images_dir + target_dir + "/plot1"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    epsilon_values = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    epsilon_step = [0.005]
    max_iter = [10]
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
    if targeted:
        plot_accuracy(f"{name} BIM Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} BIM Non-targeted - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies)
    
    # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione massima (con epsilon e max_iter fissati)
    load_dir = images_dir + target_dir + "/plot2"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    epsilon = [0.05]
    epsilon_step_values = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025]
    max_iter = [10]
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
    if targeted:
        plot_accuracy(f"{name} BIM Targeted - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} BIM Non-targeted - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies)
    
    # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con epsilon e epsilon_step fissati)
    load_dir = images_dir + target_dir + "/plot3"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    epsilon = [0.05]
    epsilon_step = [0.005]
    max_iter_values = [0, 1, 3, 5, 7, 10]
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
    if targeted:
        plot_accuracy(f"{name} BIM Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} BIM Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies)
    

def run_pgd(classifier, name, targeted, test_set, accuracy_clean, targeted_accuracy_clean, target_class, detectors=None, threshold=0.05):
    images_dir = "./dataset/test_set/adversarial_examples/pgd/"
    target_dir = "targeted" if targeted else "untargeted"
    clean_images, clean_labels = test_set.get_images()

    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con epsilon_step e epsilon_step fissati)
    load_dir = images_dir + target_dir + "/plot1"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    epsilon_values = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    epsilon_step = [0.01]
    max_iter = [10]
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
    if targeted:
        plot_accuracy(f"{name} PGD Targeted - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} PGD Non-targeted - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies)
    
    # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione massima (con epsilon e max_iter fissati)
    load_dir = images_dir + target_dir + "/plot2"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    epsilon = [0.05]
    epsilon_step_values = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025]
    max_iter = [10]
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
    if targeted:
        plot_accuracy(f"{name} PGD Targeted - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} PGD Non-targeted - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies)
    
    # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con epsilon e epsilon_step fissati)
    load_dir = images_dir + target_dir + "/plot3"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    epsilon = [0.05]
    epsilon_step = [0.01]
    max_iter_values = [0, 1, 3, 5, 7, 10]
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
    if targeted:
        plot_accuracy(f"{name} PGD Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} PGD Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies)
    

def run_df(classifier, name, test_set, accuracy_clean, detectors=None, threshold=0.05):
    images_dir = "./dataset/test_set/adversarial_examples/df/"
    clean_images, clean_labels = test_set.get_images()
    
    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con max_iter fissato)
    load_dir = images_dir + "/plot1"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    epsilon_values = [0, 1, 10, 50, 100]
    max_iter = [10]
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
    plot_accuracy(f"{name} DeepFool Non-targeted - Accuracy vs Epsilon and Max Perturbations (Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies)

    # Calcolo dell'accuracy al variare del numero di iterazioni e della perturbazione massima (con epsilon fissato)
    load_dir = images_dir + "/plot2"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    epsilon = [1]
    max_iter_values = [0, 1, 3, 5, 7, 10, 100]
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
    plot_accuracy(f"{name} DeepFool Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon})", "Max Iterations", max_iter_values, max_perturbations, accuracies)


def run_cw(classifier, name, targeted, test_set, accuracy_clean, targeted_accuracy_clean, target_class, detectors=None, threshold=0.05):
    images_dir = "./dataset/test_set/adversarial_examples/cw/"
    target_dir = "targeted" if targeted else "untargeted"
    clean_images, clean_labels = test_set.get_images()
    
    # Calcolo dell'accuracy al variare della confidence e della perturbazione massima (con max_iter e learning_rate fissati)
    load_dir = images_dir + target_dir + "/plot1"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    confidence_values = [0.0, 0.01, 0.1, 1]
    max_iter = [3]
    learning_rate = [0.01]
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
    if targeted:
        plot_accuracy(f"{name} Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Confidence and Max Perturbations (Max_iter={max_iter}; Learning_rate={learning_rate})", "Confidence", confidence_values, max_perturbations, accuracies, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} Carlini-Wagner Non-targeted - Accuracy vs Confidence and Max Perturbations (Max_iter={max_iter}; Learning_rate={learning_rate})", "Confidence", confidence_values, max_perturbations, accuracies)
    
    # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con confidence e learning_rate fissati)
    load_dir = images_dir + target_dir + "/plot2"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    confidence = [0.1]
    max_iter_values = [0, 1, 3, 5]
    learning_rate = [0.01]
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
    if targeted:
        plot_accuracy(f"{name} Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})", "Max Iterations", max_iter_values, max_perturbations, accuracies, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} Carlini-Wagner Non-targeted - Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})", "Max Iterations", max_iter_values, max_perturbations, accuracies)
        
    # Calcolo dell'accuracy al variare del learning_rate e della perturbazione massima (con confidence e max_iter fissati)
    load_dir = images_dir + target_dir + "/plot3"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    confidence = [0.1]
    max_iter = [3]
    learning_rate_values = [0.0, 0.01, 0.05, 0.1]
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
    if targeted:
        plot_accuracy(f"{name} Carlini-Wagner Targeted - Accuracy and Targeted Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})", "Learning Rate", learning_rate_values, max_perturbations, accuracies, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} Carlini-Wagner Non-targeted - Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})", "Learning Rate", learning_rate_values, max_perturbations, accuracies)


def plot_accuracy(title, x_title, x, max_perturbations, accuracies, targeted=False, targeted_accuracies=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    #print(x, max_perturbations, accuracies)
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