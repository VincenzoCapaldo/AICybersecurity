import argparse
from nets import setup_classifierNN1
import torch
from dataset import get_test_set, get_train_set
from utils import *
from attacks import FGSM, BIM, PGD, DF, CW


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_train_adv', type=bool, default=True, help='Se True, genera il train adv set; altrimenti genera il test adv set')
    parser.add_argument("--targeted", type=bool, default=False, help="Generate targeted attacks")
    parser.add_argument("--verbose", type=bool, default=True, help="Print detailed information during the generation of adversarial examples")
    args = parser.parse_args()

    # Controlla se CUDA è disponibile e imposta il dispositivo di conseguenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Attacchi selezionati
    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]

    # Generazione del training set avversario
    if args.generate_train_adv:
        for attack_name in attack_types:
            if attack_name == "fgsm":
                # Inizializza il classificatore
                classifier = setup_classifierNN1(device=device)
                attack = FGSM(classifier)
                values = [0.01, 0.02, 0.03, 0.04, 0.05]
                save_dir = "./dataset/detectors_train_set/adversarial_examples/fgsm"
            elif attack_name == "bim":
                # Inizializza il classificatore
                classifier = setup_classifierNN1(device=device)
                attack = BIM(classifier)
                values = [0.01, 0.02, 0.03, 0.04, 0.05]
                save_dir = "./dataset/detectors_train_set/adversarial_examples/bim"
            elif attack_name == "pgd":
                # Inizializza il classificatore
                classifier = setup_classifierNN1(device=device)
                attack = PGD(classifier)
                values = [0.01, 0.02, 0.03, 0.04, 0.05]
                save_dir = "./dataset/detectors_train_set/adversarial_examples/pgd"
            elif attack_name == "df":
                # Inizializza il classificatore
                classifier = setup_classifierNN1(device=device)
                attack = DF(classifier)
                values = [0.01, 0.02, 0.03, 0.04, 0.05]
                save_dir = "./dataset/detectors_train_set/adversarial_examples/df"
            elif attack_name == "cw":
                # Inizializza il classificatore
                classifier = setup_classifierNN1(device=device)
                attack = CW(classifier)
                values = [0.1, 0.5, 1, 5, 10]
                save_dir = "./dataset/detectors_train_set/adversarial_examples/cw"
            else:
                raise ValueError(f"Unknown attack type: {attack_name}")

            # Training set di partenza, con immagini clean
            train_images = get_train_set().get_images()
            attack.generate_train_adv(train_images, values, save_dir, verbose=args.verbose)
            
    # Generazione del test set avversario
    else:
        # Test set di partenza, con immagini clean
        test_set = get_test_set()
        test_images, _ = test_set.get_images()
        target_class_label = "Cristiano_Ronaldo"
        target_class = test_set.get_true_label(target_class_label)
        images_dir = "./dataset/test_set/adversarial_examples/"
        target_dir = "targeted" if args.targeted else "untargeted"

        for attack_name in attack_types:
            if attack_name == "fgsm":
                save_dir = images_dir + f"{attack_name}/" + target_dir
                # Inizializza il classificatore
                classifier = setup_classifierNN1(device=device)
                attack = FGSM(classifier)
                
                ## PLOT 1 - epsilon variabile ##
                epsilon_values = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
                attack.generate_test_adv(test_images, epsilon_values, save_dir, args.targeted, target_class, verbose=args.verbose)

            elif attack_name == "bim":
                save_dir = images_dir + f"{attack_name}/" + target_dir
                # Inizializza il classificatore
                classifier = setup_classifierNN1(device=device)
                attack = BIM(classifier)
                
                ## PLOT 1 - epsilon variabile ##
                epsilon_values = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
                epsilon_step_values = [0.005]
                max_iter_values = [10]
                attack.generate_test_adv(test_images, epsilon_values, epsilon_step_values, max_iter_values, save_dir + "/plot1", args.targeted, target_class, verbose=args.verbose)
                
                ## PLOT 2 - epsilon_step variabile ##
                epsilon_values = [0.05]
                epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
                max_iter_values = [10]
                attack.generate_test_adv(test_images, epsilon_values, epsilon_step_values, max_iter_values, save_dir + "/plot2", args.targeted, target_class, verbose=args.verbose)

                ## PLOT 3 - max_iter variabile ##
                epsilon_values = [0.05]
                epsilon_step_values = [0.005]
                max_iter_values = [1, 3, 5, 7, 10]
                attack.generate_test_adv(test_images, epsilon_values, epsilon_step_values, max_iter_values, save_dir + "/plot3", args.targeted, target_class, verbose=args.verbose)

            elif attack_name == "pgd":
                save_dir = images_dir + f"{attack_name}/" + target_dir
                # Inizializza il classificatore
                classifier = setup_classifierNN1(device=device)
                attack = PGD(classifier)

                ## PLOT 1 - epsilon variabile ##
                epsilon_values = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
                epsilon_step_values = [0.005]
                max_iter_values = [10]
                attack.generate_test_adv(test_images, epsilon_values, epsilon_step_values, max_iter_values, save_dir + "/plot1", args.targeted, target_class, verbose=args.verbose)
                
                ## PLOT 2 - epsilon_step variabile ##
                epsilon_values = [0.05]
                epsilon_step_values = [0.005, 0.01, 0.015, 0.02, 0.025]
                max_iter_values = [10]
                attack.generate_test_adv(test_images, epsilon_values, epsilon_step_values, max_iter_values, save_dir + "/plot2", args.targeted, target_class, verbose=args.verbose)

                ## PLOT 3 - max_iter variabile ##
                epsilon_values = [0.05]
                epsilon_step_values = [0.005]
                max_iter_values = [1, 3, 5, 7, 10]
                attack.generate_test_adv(test_images, epsilon_values, epsilon_step_values, max_iter_values, save_dir + "/plot3", args.targeted, target_class, verbose=args.verbose)

            elif attack_name == "df":
                save_dir = images_dir + f"{attack_name}"
                # Inizializza il classificatore
                classifier = setup_classifierNN1(device=device)
                attack = DF(classifier)
                
                ## PLOT 1 - epsilon variabile ##
                epsilon_values = [50, 100, 200]
                max_iter_values = [10]
                attack.generate_test_adv(test_images, epsilon_values, max_iter_values, save_dir + "/plot1", verbose=args.verbose)

                ## PLOT 2 - max_iter variabile ## 
                epsilon_values = [0.05]
                max_iter_values = [1, 3, 5, 7, 10]
                attack.generate_test_adv(test_images, epsilon_values, max_iter_values, save_dir + "/plot2", verbose=args.verbose)

            elif attack_name == "cw":
                save_dir = images_dir + f"{attack_name}/" + target_dir
                # Inizializza il classificatore
                classifier = setup_classifierNN1(device=device)
                attack = CW(classifier)

                ## PLOT 1 - confidence variabile ##
                confidence_values = [0.1, 0.5, 1, 2, 5, 10]
                max_iter_values = [5]
                learning_rate_values = [0.01]
                attack.generate_test_adv(test_images, confidence_values, max_iter_values, learning_rate_values, save_dir + "/plot1", args.targeted, target_class, verbose=args.verbose)
                
                ## PLOT 2 - max_iter variabile ##
                confidence_values = [0.5]
                max_iter_values = [1, 3, 5, 7, 10]
                learning_rate_values = [0.01]
                attack.generate_test_adv(test_images, confidence_values, max_iter_values, learning_rate_values, save_dir + "/plot2", args.targeted, target_class, verbose=args.verbose)

                ## PLOT 3 - learning_rate variabile ##
                confidence_values = [0.5]
                max_iter_values = [5]
                learning_rate_values = [0.001, 0.005, 0.01, 0.05, 0.1]
                attack.generate_test_adv(test_images, confidence_values, max_iter_values, learning_rate_values, save_dir + "/plot3", args.targeted, target_class, verbose=args.verbose)

            else:
                raise ValueError(f"Unknown attack type: {attack_name}")
            
if __name__ == "__main__":
    main()