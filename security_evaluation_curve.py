from utils import *

def run_fgsm(classifier, name, test_set, accuracy_clean, targeted=False, targeted_accuracy_clean=0.0, target_class=0, detectors=None, threshold=0.05):
    # Directory del test set di adv examples
    images_dir = "./dataset/test_set/adversarial_examples/"
    attack_dir = "fgsm/targeted" if targeted else "fgsm/untargeted"
    load_dir = images_dir + attack_dir
    imgs_adv = load_images_from_npy_folder(load_dir)

    clean_images, clean_labels = test_set.get_images()
    
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]

    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima
    epsilon_values = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
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
        plot_accuracy(f"{name} - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies, attack_dir, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} - Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies, attack_dir)


def run_bim(classifier, name, test_set, accuracy_clean, targeted=False, targeted_accuracy_clean=0.0, target_class=0, detectors=None, threshold=0.05):
    images_dir = "./dataset/test_set/adversarial_examples/"
    attack_dir = "bim/targeted" if targeted else "bim/untargeted"
    clean_images, clean_labels = test_set.get_images()

    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con epsilon_step e epsilon_step fissati)
    load_dir = images_dir + attack_dir + "/plot1"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    epsilon_values = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
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
        plot_accuracy(f"{name} - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies, attack_dir, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies, attack_dir)
    
    # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione massima (con epsilon e max_iter fissati)
    load_dir = images_dir + attack_dir + "/plot2"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    epsilon = [0.1]
    epsilon_step_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
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
            """ PLOT 2 non supportato per detectors
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
        """
        if targeted:
            plot_accuracy(f"{name} - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies, attack_dir, targeted, targeted_accuracies)
        else:
            plot_accuracy(f"{name} - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies, attack_dir)
        
    # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con epsilon e epsilon_step fissati)
    load_dir = images_dir + attack_dir + "/plot3"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    epsilon = [0.1]
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
            """
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
        """
        if targeted:
            plot_accuracy(f"{name} - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies, attack_dir, targeted, targeted_accuracies)
        else:
            plot_accuracy(f"{name} - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies, attack_dir)
        

def run_pgd(classifier, name, test_set, accuracy_clean, targeted=False, targeted_accuracy_clean=0.0, target_class=0, detectors=None, threshold=0.05):
    images_dir = "./dataset/test_set/adversarial_examples/"
    attack_dir = "pgd/targeted" if targeted else "pgd/untargeted"
    clean_images, clean_labels = test_set.get_images()

    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con epsilon_step e epsilon_step fissati)
    load_dir = images_dir + attack_dir + "/plot1"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    epsilon_values = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
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
        plot_accuracy(f"{name} - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies, attack_dir, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies, attack_dir)
    
    # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione massima (con epsilon e max_iter fissati)
    load_dir = images_dir + attack_dir + "/plot2"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    epsilon = [0.1]
    epsilon_step_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
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
            """
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
        """
        if targeted:
            plot_accuracy(f"{name} - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies, attack_dir, targeted, targeted_accuracies)
        else:
            plot_accuracy(f"{name} - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})", "Epsilon Step", epsilon_step_values, max_perturbations, accuracies, attack_dir)
        
    # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con epsilon e epsilon_step fissati)
    load_dir = images_dir + attack_dir + "/plot3"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]
    epsilon = [0.1]
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
            """
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
            if targeted:
                targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, threshold, targeted=True)[0])
        """
        if targeted:
            plot_accuracy(f"{name} - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies, attack_dir, targeted, targeted_accuracies)
        else:
            plot_accuracy(f"{name} - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; (Epsilon_step={epsilon_step})", "Max Iterations", max_iter_values, max_perturbations, accuracies, attack_dir)
        

def run_df(classifier, name, test_set, accuracy_clean, detectors=None, threshold=0.05):
    images_dir = "./dataset/test_set/adversarial_examples/"
    attack_dir = "df"
    clean_images, clean_labels = test_set.get_images()
    
    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con max_iter fissato)
    load_dir = images_dir + attack_dir + "/plot1"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    epsilon_values = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
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
    plot_accuracy(f"{name} - Accuracy vs Epsilon and Max Perturbations (Max_iter={max_iter})", "Epsilon", epsilon_values, max_perturbations, accuracies, attack_dir)

    # Calcolo dell'accuracy al variare del numero di iterazioni e della perturbazione massima (con epsilon fissato)
    load_dir = images_dir + attack_dir + "/plot2"
    imgs_adv = load_images_from_npy_folder(load_dir)
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    epsilon = [1e-2]
    max_iter_values = [0, 1, 5, 10, 20, 50]
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, threshold, targeted=False)[0])
    plot_accuracy(f"{name} - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon})", "Max Iterations", max_iter_values, max_perturbations, accuracies, attack_dir)


def run_cw(classifier, name, test_set, accuracy_clean, targeted=False, targeted_accuracy_clean=0.0, target_class=0, detectors=None, threshold=0.05):
    images_dir = "./dataset/test_set/adversarial_examples/"
    attack_dir = "cw/targeted" if targeted else "cw/untargeted"
    clean_images, clean_labels = test_set.get_images()
    
    # Calcolo dell'accuracy al variare della confidence e della perturbazione massima (con max_iter e learning_rate fissati)
    load_dir = images_dir + attack_dir + "/plot1"
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
        plot_accuracy(f"{name} - Accuracy and Targeted Accuracy vs Confidence and Max Perturbations (Max_iter={max_iter}; Learning_rate={learning_rate})", "Confidence", confidence_values, max_perturbations, accuracies, attack_dir, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} - Accuracy vs Confidence and Max Perturbations (Max_iter={max_iter}; Learning_rate={learning_rate})", "Confidence", confidence_values, max_perturbations, accuracies, attack_dir)
    
    # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con confidence e learning_rate fissati)
    load_dir = images_dir + attack_dir + "/plot2"
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
        plot_accuracy(f"{name} - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})", "Max Iterations", max_iter_values, max_perturbations, accuracies, attack_dir, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} - Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})", "Max Iterations", max_iter_values, max_perturbations, accuracies, attack_dir)

    # Calcolo dell'accuracy al variare del learning_rate e della perturbazione massima (con confidence e max_iter fissati)
    load_dir = images_dir + attack_dir + "/plot3"
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
        plot_accuracy(f"{name} - Accuracy and Targeted Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})", "Learning Rate", learning_rate_values, max_perturbations, accuracies, attack_dir, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} - Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})", "Learning Rate", learning_rate_values, max_perturbations, accuracies, attack_dir)