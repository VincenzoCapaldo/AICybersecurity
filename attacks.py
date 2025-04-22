from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, DeepFool, CarliniLInfMethod
from utils import compute_accuracy

NUM_CLASSES = 8631

def fgsm(classifierNN1, classifierNN2, epsilon_values, test_images, test_labels, targeted=False, target_class_values=None):
    accuracies = defaultdict(list)
    targeted_accuracies = defaultdict(list)
    max_perturbations = []

    for epsilon in epsilon_values:

        # Definizione dell'attacco
        attack = FastGradientMethod(estimator=classifierNN1, eps=epsilon, targeted=targeted)

        if targeted:
            for target_class in target_class_values:
                # Generazione delle immagini avversarie
                targeted_labels = target_class * torch.ones(test_labels.size, dtype=torch.long)
                one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
                test_images_adv = attack.generate(test_images, one_hot_targeted_labels)

                # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label vere
                accuracy = compute_accuracy(classifierNN1, test_images_adv, test_labels)
                accuracies["nn1"].append(accuracy)

                # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label della classe target
                targeted_attack_accuracy = compute_accuracy(classifierNN1, test_images_adv, targeted_labels)
                targeted_accuracies["nn1"].append(targeted_attack_accuracy)

                # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label vere
                accuracy = compute_accuracy(classifierNN2, test_images_adv, test_labels)
                accuracies["nn2"].append(accuracy)

                # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label della classe target
                targeted_attack_accuracy = compute_accuracy(classifierNN2, test_images_adv, targeted_labels)
                targeted_accuracies["nn2"].append(targeted_attack_accuracy)

                # Calcolo della perturbazione massima
                max_perturbation = np.max(np.abs(test_images_adv - test_images))
                max_perturbations.append(max_perturbation)

        else:
            # Generazione delle immagini avversarie
            test_images_adv = attack.generate(test_images)

            # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label vere
            accuracy = compute_accuracy(classifierNN1, test_images_adv, test_labels)
            accuracies["nn1"].append(accuracy)

            # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label vere
            accuracy = compute_accuracy(classifierNN2, test_images_adv, test_labels)
            accuracies["nn2"].append(accuracy)

            # Calcolo della perturbazione massima
            max_perturbation = np.max(np.abs(test_images_adv - test_images))
            max_perturbations.append(max_perturbation)

    return accuracies, max_perturbations, targeted_accuracies


def bim(classifierNN1, classifierNN2, epsilon_values, epsilon_step_values, max_iter_values, test_images, test_labels, targeted=False, target_class_values=None):
    accuracies = defaultdict(list)
    targeted_accuracies = defaultdict(list)
    max_perturbations = []

    for epsilon in epsilon_values:
        for epsilon_step in epsilon_step_values:
            for max_iter in max_iter_values:

                # Definizione dell'attacco
                attack = BasicIterativeMethod(estimator=classifierNN1, eps=epsilon, eps_step=epsilon_step, max_iter=max_iter, targeted=targeted)

                if targeted:
                    for target_class in target_class_values:
                        # Generazione delle immagini avversarie
                        targeted_labels = target_class * torch.ones(test_labels.size, dtype=torch.long)
                        one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
                        test_images_adv = attack.generate(test_images, one_hot_targeted_labels)

                        # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label vere
                        accuracy = compute_accuracy(classifierNN1, test_images_adv, test_labels)
                        accuracies["nn1"].append(accuracy)

                        # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label della classe target
                        targeted_attack_accuracy = compute_accuracy(classifierNN1, test_images_adv, targeted_labels)
                        targeted_accuracies["nn1"].append(targeted_attack_accuracy)

                        # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label vere
                        accuracy = compute_accuracy(classifierNN2, test_images_adv, test_labels)
                        accuracies["nn2"].append(accuracy)

                        # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label della classe target
                        targeted_attack_accuracy = compute_accuracy(classifierNN2, test_images_adv, targeted_labels)
                        targeted_accuracies["nn2"].append(targeted_attack_accuracy)

                        # Calcolo della perturbazione massima
                        max_perturbation = np.max(np.abs(test_images_adv - test_images))
                        max_perturbations.append(max_perturbation)
                else:
                    # Generazione delle immagini avversarie
                    test_images_adv = attack.generate(test_images)

                    # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label vere
                    accuracy = compute_accuracy(classifierNN1, test_images_adv, test_labels)
                    accuracies["nn1"].append(accuracy)

                    # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label vere
                    accuracy = compute_accuracy(classifierNN2, test_images_adv, test_labels)
                    accuracies["nn2"].append(accuracy)

                    # Calcolo della perturbazione massima
                    max_perturbation = np.max(np.abs(test_images_adv - test_images))
                    max_perturbations.append(max_perturbation)

    return accuracies, max_perturbations, targeted_accuracies


def pgd(classifierNN1, classifierNN2, epsilon_values, epsilon_step_values, max_iter_values, test_images, test_labels, targeted=False, target_class_values=None):
    accuracies = defaultdict(list)
    targeted_accuracies = defaultdict(list)
    max_perturbations = []

    for epsilon in epsilon_values:
        for epsilon_step in epsilon_step_values:
            for max_iter in max_iter_values:

                # Definizione dell'attacco
                attack = ProjectedGradientDescent(estimator=classifierNN1, eps=epsilon, eps_step=epsilon_step, max_iter=max_iter, random_eps=True, targeted=targeted)

                if targeted:
                    for target_class in target_class_values:
                        # Generazione delle immagini avversarie
                        targeted_labels = target_class * torch.ones(test_labels.size, dtype=torch.long)
                        one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
                        test_images_adv = attack.generate(test_images, one_hot_targeted_labels)

                        # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label vere
                        accuracy = compute_accuracy(classifierNN1, test_images_adv, test_labels)
                        accuracies["nn1"].append(accuracy)

                        # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label della classe target
                        targeted_attack_accuracy = compute_accuracy(classifierNN1, test_images_adv, targeted_labels)
                        targeted_accuracies["nn1"].append(targeted_attack_accuracy)

                        # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label vere
                        accuracy = compute_accuracy(classifierNN2, test_images_adv, test_labels)
                        accuracies["nn2"].append(accuracy)

                        # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label della classe target
                        targeted_attack_accuracy = compute_accuracy(classifierNN2, test_images_adv, targeted_labels)
                        targeted_accuracies["nn2"].append(targeted_attack_accuracy)

                        # Calcolo della perturbazione massima
                        max_perturbation = np.max(np.abs(test_images_adv - test_images))
                        max_perturbations.append(max_perturbation)
                else:
                    # Generazione delle immagini avversarie
                    test_images_adv = attack.generate(test_images)

                    # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label vere
                    accuracy = compute_accuracy(classifierNN1, test_images_adv, test_labels)
                    accuracies["nn1"].append(accuracy)

                    # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label vere
                    accuracy = compute_accuracy(classifierNN2, test_images_adv, test_labels)
                    accuracies["nn2"].append(accuracy)

                    # Calcolo della perturbazione massima
                    max_perturbation = np.max(np.abs(test_images_adv - test_images))
                    max_perturbations.append(max_perturbation)

    return accuracies, max_perturbations, targeted_accuracies


def deepfool(classifierNN1, classifierNN2, epsilon_values, max_iter_values, test_images, test_labels):
    accuracies = defaultdict(list)
    max_perturbations = []

    for epsilon in epsilon_values:
        for max_iter in max_iter_values:

            # Definizione dell'attacco
            attack = DeepFool(classifierNN1=classifierNN1, epsilon=epsilon, max_iter=max_iter, batch_size=64)

            # Generazione delle immagini avversarie
            test_images_adv = attack.generate(test_images)

            # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label vere
            accuracy = compute_accuracy(classifierNN1, test_images_adv, test_labels)
            accuracies["nn1"].append(accuracy)

            # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label vere
            accuracy = compute_accuracy(classifierNN2, test_images_adv, test_labels)
            accuracies["nn2"].append(accuracy)

            # Calcolo della perturbazione massima
            max_perturbation = np.max(np.abs(test_images_adv - test_images))
            max_perturbations.append(max_perturbation)

    return accuracies, max_perturbations


def carlini_wagner(classifierNN1, classifierNN2, confidence_values, max_iter_values, learning_rate_values, test_images, test_labels, targeted=False, target_class_values=None):
    accuracies = defaultdict(list)
    targeted_accuracies = defaultdict(list)
    max_perturbations = []

    for confidence in confidence_values:
        for max_iter in max_iter_values:
            for learning_rate in learning_rate_values:

                # Definizione dell'attacco
                attack = CarliniLInfMethod(classifierNN1=classifierNN1, confidence=confidence, max_iter=max_iter, learning_rate=learning_rate, targeted=targeted)

                if targeted:
                    for target_class in target_class_values:
                        # Generazione delle immagini avversarie
                        targeted_labels = target_class * torch.ones(test_labels.size, dtype=torch.long)
                        one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
                        test_images_adv = attack.generate(test_images, one_hot_targeted_labels)

                        # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label vere
                        accuracy = compute_accuracy(classifierNN1, test_images_adv, test_labels)
                        accuracies["nn1"].append(accuracy)

                        # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label della classe target
                        targeted_attack_accuracy = compute_accuracy(classifierNN1, test_images_adv, targeted_labels)
                        targeted_accuracies["nn1"].append(targeted_attack_accuracy)

                        # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label vere
                        accuracy = compute_accuracy(classifierNN2, test_images_adv, test_labels)
                        accuracies["nn2"].append(accuracy)

                        # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label della classe target
                        targeted_attack_accuracy = compute_accuracy(classifierNN2, test_images_adv, targeted_labels)
                        targeted_accuracies["nn2"].append(targeted_attack_accuracy)

                        # Calcolo della perturbazione massima
                        max_perturbation = np.max(np.abs(test_images_adv - test_images))
                        max_perturbations.append(max_perturbation)
                else:
                    # Generazione delle immagini avversarie
                    test_images_adv = attack.generate(test_images)

                    # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label vere
                    accuracy = compute_accuracy(classifierNN1, test_images_adv, test_labels)
                    accuracies["nn1"].append(accuracy)

                    # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label vere
                    accuracy = compute_accuracy(classifierNN2, test_images_adv, test_labels)
                    accuracies["nn2"].append(accuracy)

                    # Calcolo della perturbazione massima
                    max_perturbation = np.max(np.abs(test_images_adv - test_images))
                    max_perturbations.append(max_perturbation)

    return accuracies, max_perturbations, targeted_accuracies
