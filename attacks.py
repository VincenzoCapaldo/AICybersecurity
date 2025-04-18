import numpy as np
import torch
from sklearn.metrics import accuracy_score
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, DeepFool, CarliniLInfMethod
from utils import compute_accuracy

NUM_CLASSES = 8631

def fgsm(classifier, epsilon_values, test_images, test_labels, targeted=False, target_class_values=None):
    accuracies = []
    max_perturbations = []
    targeted_accuracies = []

    for epsilon in epsilon_values:

        # Definizione dell'attacco
        attack = FastGradientMethod(estimator=classifier, eps=epsilon, targeted=targeted)

        if targeted:
            for target_class in target_class_values:
                # Generazione delle immagini avversarie
                targeted_labels = target_class * torch.ones(test_labels.size, dtype=torch.long)
                one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
                test_images_adv = attack.generate(test_images, one_hot_targeted_labels)

                # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                accuracy = compute_accuracy(classifier, test_images_adv, test_labels)
                accuracies.append(accuracy)

                # Calcolo della perturbazione massima
                max_perturbation = np.max(np.abs(test_images_adv - test_images))
                max_perturbations.append(max_perturbation)

                # Calcolo dell'accuracy sulle immagini modificate rispetto alle label della classe target
                targeted_attack_accuracy = compute_accuracy(classifier, test_images_adv, targeted_labels)
                targeted_accuracies.append(targeted_attack_accuracy)
        else:
            # Generazione delle immagini avversarie
            test_images_adv = attack.generate(test_images)

            # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
            accuracy = compute_accuracy(classifier, test_images_adv, test_labels)
            accuracies.append(accuracy)

            # Calcolo della perturbazione massima
            max_perturbation = np.max(np.abs(test_images_adv - test_images))
            max_perturbations.append(max_perturbation)

    return accuracies, max_perturbations, targeted_accuracies


def bim(classifier, epsilon_values, epsilon_step_values, max_iter_values, test_images, test_labels, targeted=False, target_class_values=None):
    accuracies = []
    max_perturbations = []
    targeted_accuracies = []

    for epsilon in epsilon_values:
        for epsilon_step in epsilon_step_values:
            for max_iter in max_iter_values:

                # Definizione dell'attacco
                attack = BasicIterativeMethod(estimator=classifier, eps=epsilon, eps_step=epsilon_step, max_iter=max_iter, targeted=targeted)

                if targeted:
                    for target_class in target_class_values:
                        # Generazione delle immagini avversarie
                        targeted_labels = target_class * torch.ones(test_labels.size, dtype=torch.long)
                        one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
                        test_images_adv = attack.generate(test_images, one_hot_targeted_labels)

                        # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                        accuracy = compute_accuracy(classifier, test_images_adv, test_labels)
                        accuracies.append(accuracy)

                        # Calcolo della perturbazione massima
                        max_perturbation = np.max(np.abs(test_images_adv - test_images))
                        max_perturbations.append(max_perturbation)

                        # Calcolo dell'accuracy sulle immagini modificate rispetto alle label della classe target
                        targeted_attack_accuracy = compute_accuracy(classifier, test_images_adv, targeted_labels)
                        targeted_accuracies.append(targeted_attack_accuracy)
                else:
                    # Generazione delle immagini avversarie
                    test_images_adv = attack.generate(test_images)

                    # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                    accuracy = compute_accuracy(classifier, test_images_adv, test_labels)
                    accuracies.append(accuracy)

                    # Calcolo della perturbazione massima
                    max_perturbation = np.max(np.abs(test_images_adv - test_images))
                    max_perturbations.append(max_perturbation)

    return accuracies, max_perturbations, targeted_accuracies


def pgd(classifier, epsilon_values, epsilon_step_values, max_iter_values, test_images, test_labels, targeted=False, target_class_values=None):
    accuracies = []
    max_perturbations = []
    targeted_accuracies = []

    for epsilon in epsilon_values:
        for epsilon_step in epsilon_step_values:
            for max_iter in max_iter_values:

                # Definizione dell'attacco
                attack = ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=epsilon_step, max_iter=max_iter, random_eps=True, targeted=targeted)

                if targeted:
                    for target_class in target_class_values:
                        # Generazione delle immagini avversarie
                        targeted_labels = target_class * torch.ones(test_labels.size, dtype=torch.long)
                        one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
                        test_images_adv = attack.generate(test_images, one_hot_targeted_labels)

                        # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                        accuracy = compute_accuracy(classifier, test_images_adv, test_labels)
                        accuracies.append(accuracy)

                        # Calcolo della perturbazione massima
                        max_perturbation = np.max(np.abs(test_images_adv - test_images))
                        max_perturbations.append(max_perturbation)

                        # Calcolo dell'accuracy sulle immagini modificate rispetto alle label della classe target
                        targeted_attack_accuracy = compute_accuracy(classifier, test_images_adv, targeted_labels)
                        targeted_accuracies.append(targeted_attack_accuracy)
                else:
                    # Generazione delle immagini avversarie
                    test_images_adv = attack.generate(test_images)

                    # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                    accuracy = compute_accuracy(classifier, test_images_adv, test_labels)
                    accuracies.append(accuracy)

                    # Calcolo della perturbazione massima
                    max_perturbation = np.max(np.abs(test_images_adv - test_images))
                    max_perturbations.append(max_perturbation)

    return accuracies, max_perturbations, targeted_accuracies


def deepfool(classifier, epsilon_values, max_iter_values, test_images, test_labels):
    accuracies = []
    max_perturbations = []

    for epsilon in epsilon_values:
        for max_iter in max_iter_values:

            # Definizione dell'attacco
            attack = DeepFool(classifier=classifier, epsilon=epsilon, max_iter=max_iter)

            # Generazione delle immagini avversarie
            test_images_adv = attack.generate(test_images)

            # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
            accuracy = compute_accuracy(classifier, test_images_adv, test_labels)
            accuracies.append(accuracy)

            # Calcolo della perturbazione massima
            max_perturbation = np.max(np.abs(test_images_adv - test_images))
            max_perturbations.append(max_perturbation)

    return accuracies, max_perturbations


def carlini_wagner(classifier, confidence_values, max_iter_values, learning_rate_values, test_images, test_labels, targeted=False, target_class_values=None):
    accuracies = []
    max_perturbations = []
    targeted_accuracies = []

    for confidence in confidence_values:
        for max_iter in max_iter_values:
            for learning_rate in learning_rate_values:

                # Definizione dell'attacco
                attack = CarliniLInfMethod(classifier=classifier, confidence=confidence, max_iter=max_iter, learning_rate=learning_rate, targeted=targeted)

                if targeted:
                    for target_class in target_class_values:
                        # Generazione delle immagini avversarie
                        targeted_labels = target_class * torch.ones(test_labels.size, dtype=torch.long)
                        one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
                        test_images_adv = attack.generate(test_images, one_hot_targeted_labels)

                        # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                        accuracy = compute_accuracy(classifier, test_images_adv, test_labels)
                        accuracies.append(accuracy)

                        # Calcolo della perturbazione massima
                        max_perturbation = np.max(np.abs(test_images_adv - test_images))
                        max_perturbations.append(max_perturbation)

                        # Calcolo dell'accuracy sulle immagini modificate rispetto alle label della classe target
                        targeted_attack_accuracy = compute_accuracy(classifier, test_images_adv, targeted_labels)
                        targeted_accuracies.append(targeted_attack_accuracy)
                else:
                    # Generazione delle immagini avversarie
                    test_images_adv = attack.generate(test_images)

                    # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                    accuracy = compute_accuracy(classifier, test_images_adv, test_labels)
                    accuracies.append(accuracy)

                    # Calcolo della perturbazione massima
                    max_perturbation = np.max(np.abs(test_images_adv - test_images))
                    max_perturbations.append(max_perturbation)

    return accuracies, max_perturbations, targeted_accuracies
