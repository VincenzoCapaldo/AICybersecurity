import os
import torch
import argparse
from matplotlib import pyplot as plt
from nets import setup_classifierNN1, setup_classifierNN2
from utils import *
from attacks import  FGSM, BIM, PGD, DF, CW
from test_set import get_test_set
from utils import *

NUM_CLASSES = 8631 # numero di classi nel dataset VGGFace2

# Funzione che calcola la perturbazione massima tra le immagini originali e le immagini adversarial
def compute_max_perturbation(test_images, test_images_adv, show_distribution=False):
    # Mostra l'istogramma della perturbazione massima:
    if show_distribution:
        max_pert_sample = np.max(np.abs(test_images_adv - test_images), axis=(1, 2, 3))
        plt.figure(figsize=(6, 4))
        plt.hist(max_pert_sample, bins=50, color='blue', alpha=0.7)
        plt.title('Distribuzione delle max perturbations')
        plt.xlabel('Max perturbation')
        plt.ylabel('Frequenza')
        plt.grid(True)
        plt.show()

    # Ritorna la perturbazione massima:
    return np.max(np.abs(test_images_adv - test_images))


# Funzione che calcola l'accuratezza del classificatore sui campioni x_test (y_test contiene le classi vere)
def compute_accuracy(classifier, x_test, y_test):
    # Predizione del classificatore
    probs = classifier.predict(x_test) # contiene la probabilità per ogni classe
    y_pred = np.argmax(probs, axis=1) # contiene la classe con maggiore probabilità

    # Calcolo dell'accuracy
    accuracy = accuracy_score(y_pred, y_test)
    return accuracy 


# Funzione che calcola l'accuratezza del classificatore + detectors sui campioni x_test (y_test contiene le classi vere e y_adv se i campioni sono clean o adversarial) 
def compute_accuracy_with_detectors(classifier, x_test, y_test, y_adv, detectors, threshold=0.5, targeted=False):
    # Calcolo dei campioni rifiutati (0 se accettati da tutti i classificatori, 1 se rifiutati da almeno un classificatore)
    rejected_samples = np.zeros(x_test.shape[0], dtype=bool) # inizialmente i campioni sono tutti accettati
    for name, detector in detectors.items(): # per ogni detector
        report, _ = detector.detect(x_test) # chiamata al detector
        logits = np.array(report["predictions"]) # contiene i logits delle classi "clean" e "adversarial"
        probs = softmax(logits, axis=1) # contiene le probabilità delle classi "clean" e "adversarial"
        probs_adv = probs[:, 1] # contiene solo le probabilità delle classi "adversarial"
        is_adversarial = probs_adv > threshold # controlla se i campioni sono adversarial
        detection_error = np.sum(is_adversarial != y_adv) # conta il numero di errori del detectors
        rejected_samples = np.logical_or(is_adversarial, rejected_samples) # i detectors vengono messi in or
        print(f"Detector {name} ha scartato {np.sum(is_adversarial)} campioni.")
        print(f"Detector {name} ha rilevato correttamente {x_test.shape[0] - np.sum(detection_error)} campioni.")
        print(f"Detector {name} ha rilevato erroneamente {np.sum(detection_error)} campioni.")
        
    # Calcolo dei campioni accettati (1 se accettati da tutti i classificatori, 0 se rifiutati da almeno un classificatore)
    accepted_samples = np.logical_not(rejected_samples)
    x_accepted = x_test[accepted_samples]
    y_accepted = y_test[accepted_samples]
    if isinstance(y_accepted, torch.Tensor):
        y_accepted = y_accepted.cpu().numpy()

    # I campioni accettati vengono dati al classificatore per essere classificati
    if x_accepted.shape[0] > 0:
        probs = classifier.predict(x_accepted) # contiene la probabilità per ogni classe
        y_pred = np.argmax(probs, axis=1) # contiene la classe con maggiore probabilità
        correctly_classified = np.sum(y_pred == y_accepted) # campioni correttamente classificati
    else:
        correctly_classified = 0
        print("Nessun campione è stato accettato dai detector.")

    # Calcolo del numero di campioni correttamente rilevati (true positive)
    correctly_detected = np.sum(np.logical_and(rejected_samples, y_adv))
    
    # Calcolo dell'accuracy o della targeted_accuracy (il parametro targeted indica quale delle due calcolare)
    num_samples = y_test.shape[0]
    if targeted:
        targeted_accuracy = correctly_classified / num_samples # i campioni correctly_detected non vengono considerati perchè in tal caso l'attacco non è andato a buon fine
        return targeted_accuracy
    else:    
        accuracy = (correctly_classified + correctly_detected) / num_samples # i campioni correctly_detected vengono considerati correttamente classificati
        return accuracy


# Funzione per disegnare la security evaluation curve: accuracy e targeted accuracy al variare di x (un parametro specifico dell'attacco) e della perturbazione massima
def plot_curve(title, x_title, x, max_perturbation, accuracies, security_evaluation_curve_dir, targeted=False, targeted_accuracies=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    # Accuracy e Targeted Accuracy vs x (un parametro specifico dell'attacco)
    axes[0].plot(x, accuracies, marker='o', linestyle='-', color='b')
    if targeted:
        axes[0].plot(x, targeted_accuracies, marker='o', linestyle='-', color='r')
        axes[0].legend(["Accuracy", "Targeted Accuracy"], loc="upper right")
    else:
        axes[0].legend(["Accuracy"], loc="upper right")
    axes[0].set_xlabel(x_title)
    axes[0].grid()

    # Accuracy e Targeted Accuracy vs Max Perturbation
    axes[1].plot(max_perturbation, accuracies, marker='o', linestyle='-', color='b')
    if targeted:
        axes[1].plot(max_perturbation, targeted_accuracies, marker='o', linestyle='-', color='r')
        axes[1].legend(["Accuracy", "Targeted Accuracy"], loc="upper right")
    else:
        axes[1].legend(["Accuracy"], loc="upper right")
    axes[1].set_xlabel("Max Perturbation")
    axes[1].axvline(x=0.1, color='red', linestyle='--', linewidth=1.5) # linea rossa verticale sul vincolo da rispettare
    axes[1].grid()
    
    plt.tight_layout()
    save_path = os.path.join(security_evaluation_curve_dir, title)
    os.makedirs(security_evaluation_curve_dir, exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot {title}.png salvato.")
    plt.close()


# Funziona che genera i campioni adversarial FGSM (se generate_samples=True) e la relativa security evaluation curve.
def run_fgsm(generate_samples, device, name, classifier, detectors, test_set, accuracy_clean, targeted=False, target_class=None, targeted_accuracy_clean=None):
    attack_dir = "fgsm/targeted/" if targeted else "fgsm/untargeted/"
    test_set_adversarial_dir = "./dataset/test_set/adversarial_examples/" + attack_dir + "samples_plot1"
    security_evaluation_curve_dir = "./plots/security_evaluation_curve/" + attack_dir + "plot1"

    if targeted:
        targeted_labels = target_class * torch.ones(len(test_set), dtype=torch.long)
    else:
        targeted_labels = None

    # Caricamento dei campioni clean
    clean_images, clean_labels = test_set.get_images()

    # Calcolo dell'accuracy al variare di epsilon
    plot = {
        "epsilon_values": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
        "title_untargeted": f"{name} - Accuracy vs Epsilon and Max Perturbations",
        "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations",
        "x_axis_name": "Epsilon"
    }

    # Generazione e salvataggio dei campioni adversarial (se generate_samples=True)
    if generate_samples:
        attack = FGSM(setup_classifierNN1(device))
        i=0
        for epsilon in plot["epsilon_values"]:
            imgs_adv = attack.generate_images(clean_images, epsilon, targeted, targeted_labels)
            save_images_as_npy(imgs_adv, f"{i}_eps_{epsilon}", test_set_adversarial_dir)
            i+=1
        print("Test adversarial examples generated and saved successfully for fgsm.")
    
    # Caricamento dei campioni adversarial
    list_imgs_adv = load_images_from_npy(test_set_adversarial_dir)
    
    # Aggiunta delle performance sui dati clean
    x_axis_value = [0.0] + plot["epsilon_values"]
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]

    # Calcolo e plotting dell'accuracy e della perturbazione massima
    for imgs_adv in list_imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, imgs_adv))
        if name == "NN2":
            imgs_adv = process_images(imgs_adv) # preprocessing per il secondo classificatore   
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, imgs_adv, clean_labels))
            if targeted:
                targeted_accuracies.append(compute_accuracy(classifier, imgs_adv, targeted_labels))
        else:
            adv_flag = np.ones(len(imgs_adv), dtype=bool) # i campioni da valutare sono adversarial
            accuracies.append(compute_accuracy_with_detectors(classifier, imgs_adv, clean_labels, adv_flag, detectors, targeted=False))
            if targeted:
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, imgs_adv, targeted_labels, adv_flag, detectors, targeted=True))
    if targeted:
        plot_curve(plot["title_targeted"], plot["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir, targeted, targeted_accuracies)
    else:
        plot_curve(plot["title_untargeted"], plot["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)


# Funziona che genera i campioni adversarial BIM (se generate_samples=True) e la relativa security evaluation curve.
def run_bim(generate_samples, device, name, classifier, detectors, test_set, accuracy_clean, targeted=False, target_class=None, targeted_accuracy_clean=None):
    attack_dir = "bim/targeted/" if targeted else "bim/untargeted/"
    test_set_adversarial_dir = "./dataset/test_set/adversarial_examples/" + attack_dir
    evaluation_curve_dir = "./plots/security_evaluation_curve/" + attack_dir

    if targeted:
        targeted_labels = target_class * torch.ones(len(test_set), dtype=torch.long)
    else:
        targeted_labels = None

    # Caricamento dei campioni clean
    clean_images, clean_labels = test_set.get_images()

    plots = {
        # Calcolo dell'accuracy al variare di epsilon (con epsilon_step e max_iter fissati)
        "plot1": {
            "epsilon_values": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
            "epsilon_step_values": [0.01],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon and Max Perturbations (Epsilon_step=0,01; Max_iter=10)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step=0,01; Max_iter=10)",
            "x_axis_name": "Epsilon"
        },
        # Calcolo dell'accuracy al variare di epsilon_step (con epsilon e max_iter fissati)
        "plot2": {
            "epsilon_values": [0.1],
            "epsilon_step_values": [0.01, 0.02, 0.03, 0.04, 0.05],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon Step and Max Perturbations (Epsilon=0,1; Max_iter=10)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon=0,1; Max_iter=10)",
            "x_axis_name": "Epsilon Step"
        },
        # Calcolo dell'accuracy al variare di max_iter (con epsilon e epsilon_step fissati)
        "plot3": {
            "epsilon_values": [0.1],
            "epsilon_step_values": [0.01],
            "max_iter_values": [1, 3, 5, 7, 10],
            "title_untargeted": f"{name} - Accuracy vs Max Iterations and Max Perturbations (Epsilon=0,1; Epsilon_step=0,01)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon=0,1; Epsilon_step=0,01)",
            "x_axis_name": "Max Iterations"
        }
    }

    for plot_name, plot_data in plots.items():
        test_set_adv_dir = test_set_adversarial_dir + "samples_" + plot_name
        security_evaluation_curve_dir = evaluation_curve_dir + plot_name
        
        # Generazione e salvataggio dei campioni adversarial (se generate_samples=True)
        if generate_samples:
            attack = BIM(setup_classifierNN1(device))
            i=0
            for epsilon in plot_data["epsilon_values"]:
                for epsilon_step in plot_data["epsilon_step_values"]:
                    for max_iter in plot_data["max_iter_values"]:
                        imgs_adv = attack.generate_images(clean_images, epsilon, epsilon_step, max_iter, targeted, targeted_labels)
                        save_images_as_npy(imgs_adv, f"{i}_eps_{epsilon};eps_step_{epsilon_step};max_iter_{max_iter}", test_set_adv_dir)
                        i+=1
            print("Test adversarial examples generated and saved successfully for bim.")
        
        # Caricamento dei campioni adversarial
        list_imgs_adv = load_images_from_npy(test_set_adv_dir)
        
        # Aggiunta delle performance sui dati clean
        if plot_name=="plot1":
            x_axis_value = [0.0] + plot_data["epsilon_values"]
        elif plot_name=="plot2":
            x_axis_value = [0.0] + plot_data["epsilon_step_values"]
        elif plot_name=="plot3":
            x_axis_value = [0] + plot_data["max_iter_values"]
        max_perturbations = [0.0]
        accuracies = [accuracy_clean]
        if targeted:
            targeted_accuracies = [targeted_accuracy_clean]
        
        # Calcolo e plotting dell'accuracy e della perturbazione massima
        for imgs_adv in list_imgs_adv:
            max_perturbations.append(compute_max_perturbation(clean_images, imgs_adv))
            if name == "NN2":
                imgs_adv = process_images(imgs_adv) # preprocessing per il secondo classificatore
            if detectors is None:
                accuracies.append(compute_accuracy(classifier, imgs_adv, clean_labels))
                if targeted:
                    targeted_accuracies.append(compute_accuracy(classifier, imgs_adv, targeted_labels))
            else:
                adv_flag = np.ones(len(imgs_adv), dtype=bool) # i campioni da valutare sono adversarial
                accuracies.append(compute_accuracy_with_detectors(classifier, imgs_adv, clean_labels, adv_flag, detectors, targeted=False))
                if targeted:
                    targeted_accuracies.append(compute_accuracy_with_detectors(classifier, imgs_adv, targeted_labels, adv_flag, detectors, targeted=True))
        if targeted:
            plot_curve(plot_data["title_targeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir, targeted, targeted_accuracies)
        else:
            plot_curve(plot_data["title_untargeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)


# Funziona che genera i campioni adversarial PGD (se generate_samples=True) e la relativa security evaluation curve.
def run_pgd(generate_samples, device, name, classifier, detectors, test_set, accuracy_clean, targeted=False, target_class=None, targeted_accuracy_clean=None):
    attack_dir = "pgd/targeted/" if targeted else "pgd/untargeted/"
    test_set_adversarial_dir = "./dataset/test_set/adversarial_examples/" + attack_dir
    evaluation_curve_dir = "./plots/security_evaluation_curve/" + attack_dir

    if targeted:
        targeted_labels = target_class * torch.ones(len(test_set), dtype=torch.long)
    else:
        targeted_labels = None

    # Caricamento dei campioni clean
    clean_images, clean_labels = test_set.get_images()

    plots = {
        # Calcolo dell'accuracy al variare di epsilon (con epsilon_step e max_iter fissati)
        "plot1": {
            "epsilon_values": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
            "epsilon_step_values": [0.01],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon and Max Perturbations (Epsilon_step=0,01; Max_iter=10)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step=0,01; Max_iter=10)",
            "x_axis_name": "Epsilon"
        },
        # Calcolo dell'accuracy al variare di epsilon_step (con epsilon e max_iter fissati)
        "plot2": {
            "epsilon_values": [0.1],
            "epsilon_step_values": [0.01, 0.02, 0.03, 0.04, 0.05],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon Step and Max Perturbations (Epsilon=0,1; Max_iter=10)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon=0,1; Max_iter=10)",
            "x_axis_name": "Epsilon Step"
        },
        # Calcolo dell'accuracy al variare di max_iter (con epsilon e epsilon_step fissati)
        "plot3": {
            "epsilon_values": [0.1],
            "epsilon_step_values": [0.01],
            "max_iter_values": [1, 3, 5, 7, 10],
            "title_untargeted": f"{name} - Accuracy vs Max Iterations and Max Perturbations (Epsilon=0,1; Epsilon_step=0,01)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon=0,1; Epsilon_step=0,01)",
            "x_axis_name": "Max Iterations"
        }
    }

    for plot_name, plot_data in plots.items():
        test_set_adv_dir = test_set_adversarial_dir + "samples_" + plot_name
        security_evaluation_curve_dir = evaluation_curve_dir + plot_name
        
        # Generazione e salvataggio dei campioni adversarial (se generate_samples=True)
        if generate_samples:
            attack = PGD(setup_classifierNN1(device))
            i=0
            for epsilon in plot_data["epsilon_values"]:
                for epsilon_step in plot_data["epsilon_step_values"]:
                    for max_iter in plot_data["max_iter_values"]:
                        imgs_adv = attack.generate_images(clean_images, epsilon, epsilon_step, max_iter, targeted, targeted_labels)
                        save_images_as_npy(imgs_adv, f"{i}_eps_{epsilon};eps_step_{epsilon_step};max_iter_{max_iter}", test_set_adv_dir)
                        i+=1
            print("Test adversarial examples generated and saved successfully for pgd.")
        
        # Caricamento dei campioni adversarial
        list_imgs_adv = load_images_from_npy(test_set_adv_dir)
    
        # Aggiunta delle performance sui dati clean
        if plot_name=="plot1":
            x_axis_value = [0.0] + plot_data["epsilon_values"]
        elif plot_name=="plot2":
            x_axis_value = [0.0] + plot_data["epsilon_step_values"]
        elif plot_name=="plot3":
            x_axis_value = [0] + plot_data["max_iter_values"]
        max_perturbations = [0.0]
        accuracies = [accuracy_clean]
        if targeted:
            targeted_accuracies = [targeted_accuracy_clean]

        # Calcolo e plotting dell'accuracy e della perturbazione massima
        for imgs_adv in list_imgs_adv:
            max_perturbations.append(compute_max_perturbation(clean_images, imgs_adv))
            if name == "NN2":
                imgs_adv = process_images(imgs_adv) # preprocessing per il secondo classificatore
            if detectors is None:
                accuracies.append(compute_accuracy(classifier, imgs_adv, clean_labels))
                if targeted:
                    targeted_accuracies.append(compute_accuracy(classifier, imgs_adv, targeted_labels))
            else:
                adv_flag = np.ones(len(imgs_adv), dtype=bool) # i campioni da valutare sono adversarial
                accuracies.append(compute_accuracy_with_detectors(classifier, imgs_adv, clean_labels, adv_flag, detectors, targeted=False))
                if targeted:
                    targeted_accuracies.append(compute_accuracy_with_detectors(classifier, imgs_adv, targeted_labels, adv_flag, detectors, targeted=True))
        if targeted:
            plot_curve(plot_data["title_targeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir, targeted, targeted_accuracies)
        else:
            plot_curve(plot_data["title_untargeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)
        

# Funziona che genera i campioni adversarial DF (se generate_samples=True) e la relativa security evaluation curve.
def run_df(generate_samples, device, name, classifier, detectors, test_set, accuracy_clean):
    attack_dir = "df/untargeted/"
    test_set_adversarial_dir = "./dataset/test_set/adversarial_examples/" + attack_dir
    evaluation_curve_dir = "./plots/security_evaluation_curve/" + attack_dir

    # Caricamento dei campioni clean
    clean_images, clean_labels = test_set.get_images()

    plots = {
        # Calcolo dell'accuracy al variare di epsilon (con nb_grads e max_iter fissati)
        "plot1": {
            "epsilon_values": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
            "nb_grads_values": [10],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon and Max Perturbations (Nb_grads=10; Max_iter=10)",
            "x_axis_name": "Epsilon"
        },
        # Calcolo dell'accuracy al variare di epsilon_step (con epsilon e max_iter fissati)
        "plot2": {
            "epsilon_values": [1e-2],
            "nb_grads_values": [5, 10, 20, 50],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Nb Grads and Max Perturbations (Epsilon=1e-2; Max_iter=10)",
            "x_axis_name": "Nb Grads"
        },
        # Calcolo dell'accuracy al variare di max_iter (con epsilon e nb_grads fissati)
        "plot3": {
            "epsilon_values": [1e-2],
            "nb_grads_values": [10],
            "max_iter_values": [1, 3, 5, 7, 10],
            "title_untargeted": f"{name} - Accuracy vs Max Iterations and Max Perturbations (Epsilon=1e-2; Nb_grads=10)",
            "x_axis_name": "Max Iterations"
        }
    }

    for plot_name, plot_data in plots.items():
        test_set_adv_dir = test_set_adversarial_dir + "samples_" + plot_name
        security_evaluation_curve_dir = evaluation_curve_dir + plot_name
        
        # Generazione e salvataggio dei campioni adversarial (se generate_samples=True)
        if generate_samples:
            attack = DF(setup_classifierNN1(device))
            i=0
            for epsilon in plot_data["epsilon_values"]:
                for nb_grads in plot_data["nb_grads_values"]:
                    for max_iter in plot_data["max_iter_values"]:
                        imgs_adv = attack.generate_images(clean_images, epsilon, nb_grads, max_iter)
                        save_images_as_npy(imgs_adv, f"{i}_eps_{epsilon};nb_grads_{nb_grads};max_iter_{max_iter}", test_set_adv_dir)
                        i+=1
            print("Test adversarial examples generated and saved successfully for df.")
        
        # Caricamento dei campioni adversarial
        list_imgs_adv = load_images_from_npy(test_set_adv_dir)

        # Aggiunta delle performance sui dati clean
        if plot_name=="plot1":
            x_axis_value = [0.0] + plot_data["epsilon_values"]
        elif plot_name=="plot2":
            x_axis_value = [0] + plot_data["nb_grads_values"]
        elif plot_name=="plot3":
            x_axis_value = [0] + plot_data["max_iter_values"]
        max_perturbations = [0.0]
        accuracies = [accuracy_clean]

        # Calcolo e plotting dell'accuracy e della perturbazione massima
        for imgs_adv in list_imgs_adv:
            max_perturbations.append(compute_max_perturbation(clean_images, imgs_adv))
            if name == "NN2":
                imgs_adv = process_images(imgs_adv) # preprocessing per il secondo classificatore
            if detectors is None:
                accuracies.append(compute_accuracy(classifier, imgs_adv, clean_labels))
            else:
                adv_flag = np.ones(len(imgs_adv), dtype=bool) # i campioni da valutare sono adversarial
                accuracies.append(compute_accuracy_with_detectors(classifier, imgs_adv, clean_labels, adv_flag, detectors, targeted=False))
        plot_curve(plot_data["title_untargeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)
       

# Funziona che genera i campioni adversarial CW (se generate_samples=True) e la relativa security evaluation curve.
def run_cw(generate_samples, device, name, classifier, detectors, test_set, accuracy_clean, targeted=False, target_class=None, targeted_accuracy_clean=None):
    attack_dir = "cw/targeted/" if targeted else "cw/untargeted/"
    test_set_adversarial_dir = "./dataset/test_set/adversarial_examples/" + attack_dir
    evaluation_curve_dir = "./plots/security_evaluation_curve/" + attack_dir

    if targeted:
        targeted_labels = target_class * torch.ones(len(test_set), dtype=torch.long)
    else:
        targeted_labels = None

    # Caricamento dei campioni clean
    clean_images, clean_labels = test_set.get_images()

    plots = {
        # Calcolo dell'accuracy al variare di confidence (con learning_rate e max_iter fissati)
        "plot1": {
            "confidence_values": [0.01, 0.1, 1],
            "learning_rate_values": [0.01],
            "max_iter_values": [3],
            "title_untargeted": f"{name} - Accuracy vs Confidence and Max Perturbations (Learning_rate=0,01; Max_iter=3)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Confidence and Max Perturbations (Learning_rate=0,01; Max_iter=3)",
            "x_axis_name": "Confidence"
        },
        # Calcolo dell'accuracy al variare di learning_rate (con confidence e max_iter fissati)
        "plot2": {
            "confidence_values": [0.1],
            "learning_rate_values": [0.01, 0.05, 0.1],
            "max_iter_values": [3],
            "title_untargeted": f"{name} - Accuracy vs Learning Rate and Max Perturbations (Confidence=0,1; Max_iter=3)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Learning Rate and Max Perturbations (Confidence=0,1; Max_iter=3)",
            "x_axis_name": "Learning Rate"
        },
        # Calcolo dell'accuracy al variare di max_iter (con confidence e learning_rate fissati)
        "plot3": {
            "confidence_values": [0.1],
            "learning_rate_values": [0.01],
            "max_iter_values": [1, 3, 5],
            "title_untargeted": f"{name} - Accuracy vs Max Iterations and Max Perturbations (Confidence=0,1; Learning_rate=0,01)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Confidence=0,1; Learning_rate=0,01)",
            "x_axis_name": "Max Iterations"
        }
    }

    for plot_name, plot_data in plots.items():
        test_set_adv_dir = test_set_adversarial_dir + "samples_" + plot_name
        security_evaluation_curve_dir = evaluation_curve_dir + plot_name
        
        # Generazione e salvataggio dei campioni adversarial (se generate_samples=True)
        if generate_samples:
            attack = CW(setup_classifierNN1(device))
            i=0
            for confidence in plot_data["confidence_values"]:
                for learning_rate in plot_data["learning_rate_values"]:
                    for max_iter in plot_data["max_iter_values"]:
                        imgs_adv = attack.generate_images(clean_images, confidence, learning_rate, max_iter, targeted, targeted_labels)
                        save_images_as_npy(imgs_adv, f"{i}_confidence_{confidence};learning_rate_{learning_rate};max_iter_{max_iter}", test_set_adv_dir)
                        i+=1
            print("Test adversarial examples generated and saved successfully for cw.")
        
        # Caricamento dei campioni adversarial
        list_imgs_adv = load_images_from_npy(test_set_adv_dir)
    
        # Aggiunta delle performance sui dati clean
        if plot_name=="plot1":
            x_axis_value = [0.0] + plot_data["confidence_values"]
        elif plot_name=="plot2":
            x_axis_value = [0.0] + plot_data["learning_rate_values"]
        elif plot_name=="plot3":
            x_axis_value = [0] + plot_data["max_iter_values"]
        max_perturbations = [0.0]
        accuracies = [accuracy_clean]
        if targeted:
            targeted_accuracies = [targeted_accuracy_clean]
        
        # Calcolo e plotting dell'accuracy e della perturbazione massima
        for imgs_adv in list_imgs_adv:
            max_perturbations.append(compute_max_perturbation(clean_images, imgs_adv))
            if name == "NN2":
                imgs_adv = process_images(imgs_adv) # preprocessing per il secondo classificatore
            if detectors is None:
                accuracies.append(compute_accuracy(classifier, imgs_adv, clean_labels))
                if targeted:
                    targeted_accuracies.append(compute_accuracy(classifier, imgs_adv, targeted_labels))
            else:
                adv_flag = np.ones(len(imgs_adv), dtype=bool) # i campioni da valutare sono adversarial
                accuracies.append(compute_accuracy_with_detectors(classifier, imgs_adv, clean_labels, adv_flag, detectors, targeted=False))
                if targeted:
                    targeted_accuracies.append(compute_accuracy_with_detectors(classifier, imgs_adv, targeted_labels, adv_flag, detectors, targeted=True))
        if targeted:
            plot_curve(plot_data["title_targeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir, targeted, targeted_accuracies)
        else:
            plot_curve(plot_data["title_untargeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_name", type=str, default="NN1", choices=["NN1", "NN2", "NN1 + detectors"], help="Classifier to test")
    parser.add_argument('--generate_samples', type=bool, default=False, help='true to generate the adversarial images of the test set and generate the security evaluation curves, false to only generate the security evaluation curves')
    args = parser.parse_args()
    
    classifier_name = args.classifier_name
    generate_samples = args.generate_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]

    # Caricamento delle immagini clean (e delle rispettive etichette) del test set
    test_set = get_test_set()
    clean_images, clean_labels = test_set.get_images()

    # Setup del classificatore da testare
    if classifier_name == "NN2":
        classifier = setup_classifierNN2(device)
        clean_images = process_images(clean_images) # preprocessing delle immagini per il secondo classificatore
    else: # "NN1" o "NN1 + detectors"
        classifier = setup_classifierNN1(device)
        
    # Valutazione delle performance sui campioni clean
    if classifier_name == "NN1 + detectors":
        # Caricamento dei detectors:
        detectors = load_detectors(attack_types, device)
        # Calcolo dell'accuracy sulle immagini clean rispetto alle label vere:
        adv_labels = np.zeros(clean_images.shape[0], dtype=bool) # i campioni da valutare sono clean
        accuracy_clean, _ = compute_accuracy_with_detectors(classifier, clean_images, clean_labels, adv_labels, detectors, targeted=False)
        print(f"Accuracy del classificatore {classifier_name} su dati clean: {accuracy_clean:.3f}")
        # Calcolo della targeted accuracy sulle immagini clean rispetto alle label della classe target:
        target_class_label = "Cristiano_Ronaldo"
        target_class = test_set.get_true_label(target_class_label)
        targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
        targeted_accuracy_clean, _ = compute_accuracy_with_detectors(classifier, clean_images, targeted_labels, adv_labels, detectors, targeted=True)
        print(f"Targeted accuracy del classificatore {classifier_name} su dati clean: {targeted_accuracy_clean:.3f}")
    else: # "NN1" o "NN2"
        detectors = None # non ci sono detectors
        # Calcolo dell'accuracy sulle immagini clean rispetto alle label vere:
        accuracy_clean = compute_accuracy(classifier, clean_images, clean_labels)
        print(f"Accuracy del classificatore {classifier_name} su dati clean: {accuracy_clean:.3f}")
        # Calcolo della targeted accuracy sulle immagini clean rispetto alle label della classe target:
        target_class_label = "Cristiano_Ronaldo"
        target_class = test_set.get_true_label(target_class_label)
        targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
        targeted_accuracy_clean = compute_accuracy(classifier, clean_images, targeted_labels)
        print(f"Targeted accuracy del classificatore {classifier_name} su dati clean: {targeted_accuracy_clean:.3f}")

    # Valutazione delle performance sui campioni adversarial (security evaluation curve)
    # Attacchi untargeted:
    if "fgsm" in attack_types:
        run_fgsm(generate_samples, device, classifier_name, classifier, detectors, test_set, accuracy_clean)
    if "bim" in attack_types:
        run_bim(generate_samples, device, classifier_name, classifier, detectors, test_set, accuracy_clean)
    if "pgd" in attack_types:
        run_pgd(generate_samples, device, classifier_name, classifier, detectors, test_set, accuracy_clean)
    if "df" in attack_types:
        run_df(generate_samples, device, classifier_name, classifier, detectors, test_set, accuracy_clean)
    if "cw" in attack_types:
        run_cw(generate_samples, device, classifier_name, classifier, detectors, test_set, accuracy_clean)
    # Attacchi targeted:
    if "fgsm" in attack_types:
        run_fgsm(generate_samples, device, classifier_name, classifier, detectors, test_set, accuracy_clean, True, target_class, targeted_accuracy_clean)
    if "bim" in attack_types:
        run_bim(generate_samples, device, classifier_name, classifier, detectors, test_set, accuracy_clean, True, target_class, targeted_accuracy_clean)
    if "pgd" in attack_types:
        run_pgd(generate_samples, device, classifier_name, classifier, detectors, test_set, accuracy_clean, True, target_class, targeted_accuracy_clean)
    if "cw" in attack_types:
        run_cw(generate_samples, device, classifier_name, classifier, detectors, test_set, accuracy_clean, True, target_class, targeted_accuracy_clean)

if __name__ == "__main__":
    main()