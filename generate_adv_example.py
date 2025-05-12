import argparse
from nets import setup_classifierNN1
import torch
from dataset import get_test_set, get_train_set
from utils import *
from attacks import FGSM, BIM, PGD, DF, CW


# Generazione del train set avversario per i detectors
def generate_train_adv(classifier, train_images, attack_types, verbose=True):
    for attack_name in attack_types:
        if attack_name == "fgsm":
            attack = FGSM(classifier)
            values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
            save_dir = "./dataset/detectors_train_set/adversarial_examples/fgsm"
        
        elif attack_name == "bim":
            attack = BIM(classifier)
            values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
            save_dir = "./dataset/detectors_train_set/adversarial_examples/bim"
        
        elif attack_name == "pgd":
            attack = PGD(classifier)
            values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
            save_dir = "./dataset/detectors_train_set/adversarial_examples/pgd"
        
        elif attack_name == "df":
            attack = DF(classifier)
            values = [0.01, 0.02, 0.03, 0.04, 0.05]  # DA CAMBIARE
            save_dir = "./dataset/detectors_train_set/adversarial_examples/df"
        
        elif attack_name == "cw":
            attack = CW(classifier)
            values = [0.01, 0.1, 1]  # DA CAMBIARE
            save_dir = "./dataset/detectors_train_set/adversarial_examples/cw"
        else:
            raise ValueError(f"Unknown attack type: {attack_name}")
        attack.generate_train_adv(train_images, values, save_dir, verbose=verbose)  # Generazione train set


# Generazione del test set avversario
def generate_test_adv(classifier, test_images, attack_types, targeted=False, target_class=0, verbose=True):
    images_dir = "./dataset/test_set/adversarial_examples/"
    target_dir = "targeted" if targeted else "untargeted"
    for attack_name in attack_types:
        save_dir = images_dir + f"{attack_name}/" + target_dir # Cartella di salvataggio immagini adv

        if attack_name == "fgsm":
            attack = FGSM(classifier)
            
            ## PLOT 1 - epsilon variabile ##
            epsilon_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
            attack.generate_test_adv(test_images, epsilon_values, save_dir, targeted, target_class, verbose=verbose)

        elif attack_name == "bim":
            attack = BIM(classifier)
            
            ## PLOT 1 - epsilon variabile ##
            epsilon_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
            epsilon_step_values = [0.01]
            max_iter_values = [10]
            attack.generate_test_adv(test_images, epsilon_values, epsilon_step_values, max_iter_values, save_dir + "/plot1", targeted, target_class, verbose=verbose)
            
            ## PLOT 2 - epsilon_step variabile ##
            epsilon_values = [0.1]
            epsilon_step_values = [0.01, 0.02, 0.03, 0.04, 0.05]
            max_iter_values = [10]
            attack.generate_test_adv(test_images, epsilon_values, epsilon_step_values, max_iter_values, save_dir + "/plot2", targeted, target_class, verbose=verbose)

            ## PLOT 3 - max_iter variabile ##
            epsilon_values = [0.1]
            epsilon_step_values = [0.01]
            max_iter_values = [1, 3, 5, 7, 10]
            attack.generate_test_adv(test_images, epsilon_values, epsilon_step_values, max_iter_values, save_dir + "/plot3", targeted, target_class, verbose=verbose)

        elif attack_name == "pgd":
            attack = PGD(classifier)

            ## PLOT 1 - epsilon variabile ##
            epsilon_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
            epsilon_step_values = [0.01]
            max_iter_values = [10]
            attack.generate_test_adv(test_images, epsilon_values, epsilon_step_values, max_iter_values, save_dir + "/plot1", targeted, target_class, verbose=verbose)
            
            ## PLOT 2 - epsilon_step variabile ##
            epsilon_values = [0.1]
            epsilon_step_values = [0.01, 0.02, 0.03, 0.04, 0.05]
            max_iter_values = [10]
            attack.generate_test_adv(test_images, epsilon_values, epsilon_step_values, max_iter_values, save_dir + "/plot2", targeted, target_class, verbose=verbose)

            ## PLOT 3 - max_iter variabile ##
            epsilon_values = [0.1]
            epsilon_step_values = [0.01]
            max_iter_values = [1, 3, 5, 7, 10]
            attack.generate_test_adv(test_images, epsilon_values, epsilon_step_values, max_iter_values, save_dir + "/plot3", targeted, target_class, verbose=verbose)

        elif attack_name == "df":
            save_dir = images_dir + f"{attack_name}"
            attack = DF(classifier)
            
            ## PLOT 1 - epsilon variabile ##
            epsilon_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            max_iter_values = [10]
            attack.generate_test_adv(test_images, epsilon_values, max_iter_values, save_dir + "/plot1", verbose=verbose)

            ## PLOT 2 - max_iter variabile ## 
            epsilon_values = [1e-2]
            max_iter_values = [1, 5, 10, 20, 50]
            attack.generate_test_adv(test_images, epsilon_values, max_iter_values, save_dir + "/plot2", verbose=verbose)

        elif attack_name == "cw":
            attack = CW(classifier)

            ## PLOT 1 - confidence variabile ##
            confidence_values = [0.01, 0.1, 1]
            max_iter_values = [3]
            learning_rate_values = [0.01]
            attack.generate_test_adv(test_images, confidence_values, max_iter_values, learning_rate_values, save_dir + "/plot1", targeted, target_class, verbose=verbose)
            
            ## PLOT 2 - max_iter variabile ##
            confidence_values = [0.1]
            max_iter_values = [1, 3, 5]
            learning_rate_values = [0.01]
            attack.generate_test_adv(test_images, confidence_values, max_iter_values, learning_rate_values, save_dir + "/plot2", targeted, target_class, verbose=verbose)

            ## PLOT 3 - learning_rate variabile ##
            confidence_values = [0.1]
            max_iter_values = [3]
            learning_rate_values = [0.01, 0.05, 0.1]
            attack.generate_test_adv(test_images, confidence_values, max_iter_values, learning_rate_values, save_dir + "/plot3", targeted, target_class, verbose=verbose)
        else:
            raise ValueError(f"Unknown attack type: {attack_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_train_adv', type=bool, default=False, help='Se True, genera il train adv set; altrimenti genera il test adv set')
    parser.add_argument("--targeted", type=bool, default=False, help="Generate targeted attacks")
    parser.add_argument("--verbose", type=bool, default=True, help="Print detailed information during the generation of adversarial examples")
    args = parser.parse_args()

    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Attacchi selezionati
    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]
    attack_types = ["df"]

    classifier = setup_classifierNN1(device)  # Inizializza il classificatore
    if args.generate_train_adv:
        train_images = get_train_set().get_images()  # Training set di partenza, con immagini clean
        generate_train_adv(classifier, train_images, attack_types, args.verbose)
    else:
        test_set = get_test_set()  # Test set di partenza, con immagini clean
        test_images, _ = test_set.get_images()
        target_class_label = "Cristiano_Ronaldo"
        target_class = test_set.get_true_label(target_class_label)
        generate_test_adv(classifier, test_images, attack_types, args.targeted, target_class, args.verbose)
            
if __name__ == "__main__":
    main()