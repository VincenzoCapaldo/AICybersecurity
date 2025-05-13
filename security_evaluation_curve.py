import os
from matplotlib import pyplot as plt
import torch
from attacks import  FGSM, BIM, PGD, DF, CW
from utils import *

def plot_accuracy(title, x_title, x, max_perturbations, accuracies, security_evaluation_curve_dir, targeted=False, targeted_accuracies=None):
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
    axes[1].axvline(x=0.1, color='red', linestyle='--', linewidth=1.5) # vincolo da rispettare
    axes[1].grid()
    
    plt.tight_layout()
    filename = title.replace(".",",")+ ".png"
    plot_dir = os.path.join("./plot", security_evaluation_curve_dir)
    save_path = os.path.join(plot_dir, filename)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot {title}.png salvato.")
    plt.close()


def run_fgsm(classifier, name, test_set, accuracy_clean, detectors=None, targeted=False, target_class=None, targeted_accuracy_clean=None, generate_samples=False):
    
    attack_dir = "fgsm/targeted/plot" if targeted else "fgsm/untargeted/plot"
    test_set_adv_dir = "./dataset/test_set/adversarial_examples/" + attack_dir
    security_evaluation_curve_dir = "./security_evaluation_curve/" + attack_dir

    epsilon_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]

    if targeted:
        targeted_labels = target_class * torch.ones(len(test_set), dtype=torch.long)

    if generate_samples:
        attack = FGSM(classifier)
        i=0
        for epsilon in epsilon_values:
            # Generazione delle immagini avversarie
            imgs_adv = attack.generate_attack(test_set, epsilon, targeted, targeted_labels)
            save_images_as_npy(imgs_adv, f"{i}_eps_{epsilon}", test_set_adv_dir)
            i+=1
        print("Test adversarial examples generated and saved successfully for fgsm.")
    else:
        imgs_adv = load_images_from_npy_folder(test_set_adv_dir)
        clean_images, clean_labels = test_set.get_images()
    
    epsilon_values.insert(0, 0.0) # aggiunge 0.00 in posizione 0
    max_perturbations = [0.0]
    accuracies = [accuracy_clean]
    if targeted:
        targeted_accuracies = [targeted_accuracy_clean]

    # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima
    for img_adv in imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
        if name == "NN2":
            img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore   
        if detectors is None:
            accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
            if targeted:
                targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
        else:
            adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
            accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, targeted=False)[0])
            if targeted:
                targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, targeted=True)[0])
    if targeted:
        plot_accuracy(f"{name} - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies, security_evaluation_curve_dir, targeted, targeted_accuracies)
    else:
        plot_accuracy(f"{name} - Accuracy vs Epsilon and Max Perturbations", "Epsilon", epsilon_values, max_perturbations, accuracies, security_evaluation_curve_dir)


def run_bim(classifier, name, test_set, accuracy_clean, detectors=None, targeted=False, target_class=None, targeted_accuracy_clean=None, generate_samples=False):
    
    attack_dir = "bim/targeted" if targeted else "bim/untargeted"
    test_set_adv_dir = "./dataset/test_set/adversarial_examples/" + attack_dir
    security_evaluation_curve_dir = "./security_evaluation_curve/" + attack_dir

    if targeted:
        targeted_labels = target_class * torch.ones(len(test_set), dtype=torch.long)

    plots = {
        # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con epsilon_step e max_iter fissati)
        "plot1": {
            "epsilon_values": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
            "epsilon_step_values": [0.01],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})",
            "x_axis_name": "Epsilon"
        },
        # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione massima (con epsilon e max_iter fissati)
        "plot2": {
            "epsilon_values": [0.1],
            "epsilon_step_values": [0.01, 0.02, 0.03, 0.04, 0.05],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})",
            "x_axis_name": "Epsilon Step"
        },
        # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con epsilon e epsilon_step fissati)
        "plot3": {
            "epsilon_values": [0.1],
            "epsilon_step_values": [0.01],
            "max_iter_values": [1, 3, 5, 7, 10],
            "title_untargeted": f"{name} - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; Epsilon_step={epsilon_step})",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; Epsilon_step={epsilon_step})",
            "x_axis_name": "Max Iterations"
        }
    }

    for plot_name, plot_data in plots.items():
        test_set_adv_dir = test_set_adv_dir + plot_name
        security_evaluation_curve_dir = security_evaluation_curve_dir + plot_name
        if generate_samples:
            attack = BIM(classifier)
            i=0
            for epsilon in plot_data["epsilon_values"]:
                for epsilon_step in plot_data["epsilon_step_values"]:
                    for max_iter in plot_data["max_iter_values"]:
                        # Generazione delle immagini avversarie
                        imgs_adv = attack.generate_attack(test_set, epsilon, epsilon_step, max_iter, targeted, targeted_labels)
                        save_images_as_npy(imgs_adv, f"{i}_eps_{epsilon};eps_step_{epsilon_step};max_iter_{max_iter}", test_set_adv_dir)
                        i+=1
            print("Test adversarial examples generated and saved successfully for bim.")
        else:
            imgs_adv = load_images_from_npy_folder(test_set_adv_dir)
            clean_images, clean_labels = test_set.get_images()
    
        if plot_name=="plot1":
            x_axis_value = plot_data["epsilon_values"].insert(0, 0.0) # aggiunge 0.0 in posizione 0
        elif plot_name=="plot2":
            x_axis_value = plot_data["epsilon_step_values"].insert(0, 0.0) # aggiunge 0.0 in posizione 0
        elif plot_name=="plot3":
            x_axis_value = plot_data["max_iter_values"].insert(0, 0) # aggiunge 0 in posizione 0

        max_perturbations = [0.0]
        accuracies = [accuracy_clean]
        if targeted:
            targeted_accuracies = [targeted_accuracy_clean]
        for img_adv in imgs_adv:
            max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
            if name == "NN2":
                img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
            if detectors is None:
                accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
                if targeted:
                    targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
            else:
                adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
                accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, targeted=False)[0])
                if targeted:
                    targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, targeted=True)[0])
        if targeted:
            plot_accuracy(plot_data["title_targeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir, targeted, targeted_accuracies)
        else:
            plot_accuracy(plot_data["title_untargeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)


def run_pgd(classifier, name, test_set, accuracy_clean, detectors=None, targeted=False, target_class=None, targeted_accuracy_clean=None, generate_samples=False):
    
    attack_dir = "pgd/targeted" if targeted else "pgd/untargeted"
    test_set_adv_dir = "./dataset/test_set/adversarial_examples/" + attack_dir
    security_evaluation_curve_dir = "./security_evaluation_curve/" + attack_dir

    if targeted:
        targeted_labels = target_class * torch.ones(len(test_set), dtype=torch.long)
    
    plots = {
        # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con epsilon_step e max_iter fissati)
        "plot1": {
            "epsilon_values": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
            "epsilon_step_values": [0.01],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step={epsilon_step}; Max_iter={max_iter})",
            "x_axis_name": "Epsilon"
        },
        # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione massima (con epsilon e max_iter fissati)
        "plot2": {
            "epsilon_values": [0.1],
            "epsilon_step_values": [0.01, 0.02, 0.03, 0.04, 0.05],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})",
            "x_axis_name": "Epsilon Step"
        },
        # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con epsilon e epsilon_step fissati)
        "plot3": {
            "epsilon_values": [0.1],
            "epsilon_step_values": [0.01],
            "max_iter_values": [1, 3, 5, 7, 10],
            "title_untargeted": f"{name} - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; Epsilon_step={epsilon_step})",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; Epsilon_step={epsilon_step})",
            "x_axis_name": "Max Iterations"
        }
    }

    for plot_name, plot_data in plots.items():
        test_set_adv_dir = test_set_adv_dir + plot_name
        security_evaluation_curve_dir = security_evaluation_curve_dir + plot_name
        if generate_samples:
            attack = PGD(classifier)
            i=0
            for epsilon in plot_data["epsilon_values"]:
                for epsilon_step in plot_data["epsilon_step_values"]:
                    for max_iter in plot_data["max_iter_values"]:
                        # Generazione delle immagini avversarie
                        imgs_adv = attack.generate_attack(test_set, epsilon, epsilon_step, max_iter, targeted, targeted_labels)
                        save_images_as_npy(imgs_adv, f"{i}_eps_{epsilon};eps_step_{epsilon_step};max_iter_{max_iter}", test_set_adv_dir)
                        i+=1
            print("Test adversarial examples generated and saved successfully for pgd.")
        else:
            imgs_adv = load_images_from_npy_folder(test_set_adv_dir)
            clean_images, clean_labels = test_set.get_images()
    
        if plot_name=="plot1":
            x_axis_value = plot_data["epsilon_values"].insert(0, 0.0) # aggiunge 0.0 in posizione 0
        elif plot_name=="plot2":
            x_axis_value = plot_data["epsilon_step_values"].insert(0, 0.0) # aggiunge 0.0 in posizione 0
        elif plot_name=="plot3":
            x_axis_value = plot_data["max_iter_values"].insert(0, 0) # aggiunge 0 in posizione 0

        max_perturbations = [0.0]
        accuracies = [accuracy_clean]
        if targeted:
            targeted_accuracies = [targeted_accuracy_clean]
        for img_adv in imgs_adv:
            max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
            if name == "NN2":
                img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
            if detectors is None:
                accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
                if targeted:
                    targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
            else:
                adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
                accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, targeted=False)[0])
                if targeted:
                    targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, targeted=True)[0])
        if targeted:
            plot_accuracy(plot_data["title_targeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir, targeted, targeted_accuracies)
        else:
            plot_accuracy(plot_data["title_untargeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)
        

def run_df(classifier, name, test_set, accuracy_clean, detectors=None, generate_samples=False):
   
    attack_dir = "df/targeted"
    test_set_adv_dir = "./dataset/test_set/adversarial_examples/" + attack_dir
    security_evaluation_curve_dir = "./security_evaluation_curve/" + attack_dir

    plots = {
        # Calcolo dell'accuracy al variare di epsilon e della perturbazione massima (con nb_grads e max_iter fissati)
        "plot1": {
            "epsilon_values": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
            "nb_grads_values": [10],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon and Max Perturbations (Nb_grads={nb_grads}; Max_iter={max_iter})",
            "x_axis_name": "Epsilon"
        },
        # Calcolo dell'accuracy al variare di epsilon_step e della perturbazione massima (con epsilon e max_iter fissati)
        "plot2": {
            "epsilon_values": [1e-2],
            "nb_grads_values": [5, 10, 20, 50],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Nb Grads and Max Perturbations (Epsilon={epsilon}; Max_iter={max_iter})",
            "x_axis_name": "Nb Grads"
        },
        # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con epsilon e nb_grads fissati)
        "plot3": {
            "epsilon_values": [1e-2],
            "nb_grads_values": [10],
            "max_iter_values": [1, 3, 5, 7, 10],
            "title_untargeted": f"{name} - Accuracy vs Max Iterations and Max Perturbations (Epsilon={epsilon}; Nb_grads={nb_grads})",
            "x_axis_name": "Max Iterations"
        }
    }

    for plot_name, plot_data in plots.items():
        test_set_adv_dir = test_set_adv_dir + plot_name
        security_evaluation_curve_dir = security_evaluation_curve_dir + plot_name
        if generate_samples:
            attack = DF(classifier)
            i=0
            for epsilon in plot_data["epsilon_values"]:
                for nb_grads in plot_data["nb_grads_values"]:
                    for max_iter in plot_data["max_iter_values"]:
                        # Generazione delle immagini avversarie
                        imgs_adv = attack.generate_attack(test_set, epsilon, nb_grads, max_iter)
                        save_images_as_npy(imgs_adv, f"{i}_eps_{epsilon};nb_grads_{nb_grads};max_iter_{max_iter}", test_set_adv_dir)
                        i+=1
            print("Test adversarial examples generated and saved successfully for df.")
        else:
            imgs_adv = load_images_from_npy_folder(test_set_adv_dir)
            clean_images, clean_labels = test_set.get_images()
    
        if plot_name=="plot1":
            x_axis_value = plot_data["epsilon_values"].insert(0, 0.0) # aggiunge 0.0 in posizione 0
        elif plot_name=="plot2":
            x_axis_value = plot_data["nb_grads_values"].insert(0, 0.0) # aggiunge 0.0 in posizione 0
        elif plot_name=="plot3":
            x_axis_value = plot_data["max_iter_values"].insert(0, 0) # aggiunge 0 in posizione 0
            
        max_perturbations = [0.0]
        accuracies = [accuracy_clean]
        for img_adv in imgs_adv:
            max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
            if name == "NN2":
                img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
            if detectors is None:
                accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
            else:
                adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
                accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, targeted=False)[0])
        
        plot_accuracy(plot_data["title_untargeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)
       

def run_cw(classifier, name, test_set, accuracy_clean, detectors=None, targeted=False, target_class=None, targeted_accuracy_clean=None, generate_samples=False):
    
    attack_dir = "cw/targeted" if targeted else "cw/untargeted"
    test_set_adv_dir = "./dataset/test_set/adversarial_examples/" + attack_dir
    security_evaluation_curve_dir = "./security_evaluation_curve/" + attack_dir

    if targeted:
        targeted_labels = target_class * torch.ones(len(test_set), dtype=torch.long)
    
    plots = {
        # Calcolo dell'accuracy al variare di confidence e della perturbazione massima (con learning_rate e max_iter fissati)
        "plot1": {
            "confidence_values": [0.01, 0.1, 1],
            "learning_rate_values": [0.01],
            "max_iter_values": [3],
            "title_untargeted": f"{name} - Accuracy vs Confidence and Max Perturbations (Learning_rate={learning_rate}; Max_iter={max_iter})",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Confidence and Max Perturbations (Learning_rate={learning_rate}; Max_iter={max_iter})",
            "x_axis_name": "Confidence"
        },
        # Calcolo dell'accuracy al variare di learning_rate e della perturbazione massima (con confidence e max_iter fissati)
        "plot2": {
            "confidence_values": [0.1],
            "learning_rate_values": [0.01, 0.05, 0.1],
            "max_iter_values": [3],
            "title_untargeted": f"{name} - Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Learning Rate and Max Perturbations (Confidence={confidence}; Max_iter={max_iter})",
            "x_axis_name": "Learning Rate"
        },
        # Calcolo dell'accuracy al variare di max_iter e della perturbazione massima (con confidence e learning_rate fissati)
        "plot3": {
            "confidence_values": [0.1],
            "learning_rate_values": [0.01],
            "max_iter_values": [1, 3, 5],
            "title_untargeted": f"{name} - Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Confidence={confidence}; Learning_rate={learning_rate})",
            "x_axis_name": "Max Iterations"
        }
    }

    for plot_name, plot_data in plots.items():
        test_set_adv_dir = test_set_adv_dir + plot_name
        security_evaluation_curve_dir = security_evaluation_curve_dir + plot_name
        if generate_samples:
            attack = CW(classifier)
            i=0
            for confidence in plot_data["confidence_values"]:
                for learning_rate in plot_data["learning_rate_values"]:
                    for max_iter in plot_data["max_iter_values"]:
                        # Generazione delle immagini avversarie
                        imgs_adv = attack.generate_attack(test_set, confidence, learning_rate, max_iter, targeted, targeted_labels)
                        save_images_as_npy(imgs_adv, f"{i}_confidence_{confidence};learning_rate_{learning_rate};max_iter_{max_iter}", test_set_adv_dir)
                        i+=1
            print("Test adversarial examples generated and saved successfully for cw.")
        else:
            imgs_adv = load_images_from_npy_folder(test_set_adv_dir)
            clean_images, clean_labels = test_set.get_images()
    
        if plot_name=="plot1":
            x_axis_value = plot_data["confidence_values"].insert(0, 0.0) # aggiunge 0.0 in posizione 0
        elif plot_name=="plot2":
            x_axis_value = plot_data["learning_rate_values"].insert(0, 0.0) # aggiunge 0.0 in posizione 0
        elif plot_name=="plot3":
            x_axis_value = plot_data["max_iter_values"].insert(0, 0) # aggiunge 0 in posizione 0

        max_perturbations = [0.0]
        accuracies = [accuracy_clean]
        if targeted:
            targeted_accuracies = [targeted_accuracy_clean]
        for img_adv in imgs_adv:
            max_perturbations.append(compute_max_perturbation(clean_images, img_adv))
            if name == "NN2":
                img_adv = process_images(img_adv)  # Preprocessing per il secondo classificatore
            if detectors is None:
                accuracies.append(compute_accuracy(classifier, img_adv, clean_labels))
                if targeted:
                    targeted_accuracies.append(compute_accuracy(classifier, img_adv, targeted_labels))
            else:
                adv_labels = np.ones(len(img_adv), dtype=bool) # label associate a immagini avversarie (classe 1)
                accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, clean_labels, adv_labels, detectors, targeted=False)[0])
                if targeted:
                    targeted_accuracies.append(compute_accuracy_with_detectors(classifier, img_adv, targeted_labels, adv_labels, detectors, targeted=True)[0])
        if targeted:
            plot_accuracy(plot_data["title_targeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir, targeted, targeted_accuracies)
        else:
            plot_accuracy(plot_data["title_untargeted"], plot_data["x_axis_name"], x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)