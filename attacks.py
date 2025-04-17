import numpy as np
import torch
from sklearn.metrics import accuracy_score
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, DeepFool, CarliniLInfMethod

num_classes = 8631

def compute_accuracy(classifier, x_test, y_test):
    # Predizioni del modello (output con le probabilità per ogni classe)
    y_pred = classifier.predict(x_test)  # Shape: (N, 10)

    # Convertiamo da probabilità a etichette (argmax sulle colonne)
    y_pred_labels = np.argmax(y_pred, axis=1)  # Predizioni finali

    # Calcoliamo l'accuratezza
    accuracy = accuracy_score(y_pred_labels, y_test)
    return accuracy


def fgsm(classifier, epsilon_values, test_images, test_labels, targeted, target_class_values):
    accuracies = []
    average_perturbations = []
    targeted_accuracies = []

    for epsilon in epsilon_values:

        # Definizione dell'attacco
        attack = FastGradientMethod(estimator=classifier, eps=epsilon, targeted=targeted)

        if targeted:
            for target_class in target_class_values:
                # Generazione delle immagini avversarie
                targeted_labels = target_class * torch.ones(test_labels.size, dtype=torch.long)
                one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, num_classes=num_classes).numpy()
                test_images_adv = attack.generate(test_images, one_hot_targeted_labels)

                # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                accuracy_test = compute_accuracy(classifier, test_images_adv, test_labels)
                accuracies.append(accuracy_test)

                # Calcolo della perturbazione media
                perturbation = np.mean(np.abs(test_images_adv - test_images))
                average_perturbations.append(perturbation)

                # Calcolo dell'accuracy sulle immagini modificate rispetto alle label della classe target
                targeted_attack_accuracy = compute_accuracy(classifier, test_images_adv, targeted_labels)
                targeted_accuracies.append(targeted_attack_accuracy)
        else:
            # Generazione delle immagini avversarie
            test_images_adv = attack.generate(test_images)

            # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
            accuracy_test = compute_accuracy(classifier, test_images_adv, test_labels)
            accuracies.append(accuracy_test)

            # Calcolo della perturbazione media
            perturbation = np.mean(np.abs(test_images_adv - test_images))
            average_perturbations.append(perturbation)

    return accuracies, average_perturbations, targeted_accuracies


def bim(classifier, epsilon_values, epsilon_step_values, max_iter_values, test_images, test_labels, targeted, target_class_values):
    accuracies = []
    average_perturbations = []
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
                        one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, num_classes=num_classes).numpy()
                        test_images_adv = attack.generate(test_images, one_hot_targeted_labels)

                        # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                        accuracy_test = compute_accuracy(classifier, test_images_adv, test_labels)
                        accuracies.append(accuracy_test)

                        # Calcolo della perturbazione media
                        perturbation = np.mean(np.abs(test_images_adv - test_images))
                        average_perturbations.append(perturbation)

                        # Calcolo dell'accuracy sulle immagini modificate rispetto alle label della classe target
                        targeted_attack_accuracy = compute_accuracy(classifier, test_images_adv, targeted_labels)
                        targeted_accuracies.append(targeted_attack_accuracy)
                else:
                    # Generazione delle immagini avversarie
                    test_images_adv = attack.generate(test_images)

                    # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                    accuracy_test = compute_accuracy(classifier, test_images_adv, test_labels)
                    accuracies.append(accuracy_test)

                    # Calcolo della perturbazione media
                    perturbation = np.mean(np.abs(test_images_adv - test_images))
                    average_perturbations.append(perturbation)

    return accuracies, average_perturbations, targeted_accuracies


def pgd(classifier, epsilon_values, epsilon_step_values, max_iter_values, test_images, test_labels, targeted, target_class_values):
    accuracies = []
    average_perturbations = []
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
                        one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, num_classes=num_classes).numpy()
                        test_images_adv = attack.generate(test_images, one_hot_targeted_labels)

                        # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                        accuracy_test = compute_accuracy(classifier, test_images_adv, test_labels)
                        accuracies.append(accuracy_test)

                        # Calcolo della perturbazione media
                        perturbation = np.mean(np.abs(test_images_adv - test_images))
                        average_perturbations.append(perturbation)

                        # Calcolo dell'accuracy sulle immagini modificate rispetto alle label della classe target
                        targeted_attack_accuracy = compute_accuracy(classifier, test_images_adv, targeted_labels)
                        targeted_accuracies.append(targeted_attack_accuracy)
                else:
                    # Generazione delle immagini avversarie
                    test_images_adv = attack.generate(test_images)

                    # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                    accuracy_test = compute_accuracy(classifier, test_images_adv, test_labels)
                    accuracies.append(accuracy_test)

                    # Calcolo della perturbazione media
                    perturbation = np.mean(np.abs(test_images_adv - test_images))
                    average_perturbations.append(perturbation)

    return accuracies, average_perturbations, targeted_accuracies


def deepfool(classifier, epsilon_values, max_iter_values, test_images, test_labels):
    accuracies = []
    average_perturbations = []

    for epsilon in epsilon_values:
        for max_iter in max_iter_values:

            # Definizione dell'attacco
            attack = DeepFool(classifier=classifier, epsilon=epsilon, max_iter=max_iter)

            # Generazione delle immagini avversarie
            test_images_adv = attack.generate(test_images)

            # Calcolo della massima perturbazione L_inf per ogni immagine
            perturbations = np.abs(test_images_adv - test_images)
            max_perturbation = np.max(perturbations.reshape(perturbations.shape[0], -1), axis=1)

            # Filtraggio delle immagini con L_inf ≤ 0.05
            test_images_adv_filtered = test_images_adv[max_perturbation <= 0.05]
            test_labels_filtered = test_labels[max_perturbation <= 0.05]

            # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
            accuracy_test = compute_accuracy(classifier, test_images_adv_filtered, test_labels_filtered)
            accuracies.append(accuracy_test)

            # Calcolo della perturbazione media
            perturbation = np.mean(np.abs(test_images_adv_filtered - test_labels_filtered))
            average_perturbations.append(perturbation)

    return accuracies, average_perturbations


def carlini_wagner(classifier, confidence_values, max_iter_values, learning_rate_values, test_images, test_labels, targeted, target_class_values):
    accuracies = []
    average_perturbations = []
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
                        one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, num_classes=num_classes).numpy()
                        test_images_adv = attack.generate(test_images, one_hot_targeted_labels)

                        # Calcolo della massima perturbazione L_inf per ogni immagine
                        perturbations = np.abs(test_images_adv - test_images)
                        max_perturbation = np.max(perturbations.reshape(perturbations.shape[0], -1), axis=1)

                        # Filtraggio delle immagini con L_inf ≤ 0.05
                        test_images_adv_filtered = test_images_adv[max_perturbation <= 0.05]
                        test_labels_filtered = test_labels[max_perturbation <= 0.05]

                        # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                        accuracy_test = compute_accuracy(classifier, test_images_adv_filtered, test_labels_filtered)
                        accuracies.append(accuracy_test)

                        # Calcolo della perturbazione media
                        perturbation = np.mean(np.abs(test_images_adv_filtered - test_labels_filtered))
                        average_perturbations.append(perturbation)

                        # Calcolo dell'accuracy sulle immagini modificate rispetto alle label della classe target
                        targeted_attack_accuracy = compute_accuracy(classifier, test_images_adv_filtered, targeted_labels)
                        targeted_accuracies.append(targeted_attack_accuracy)
                else:
                    # Generazione delle immagini avversarie
                    test_images_adv = attack.generate(test_images)

                    # Calcolo della massima perturbazione L_inf per ogni immagine
                    perturbations = np.abs(test_images_adv - test_images)
                    max_perturbation = np.max(perturbations.reshape(perturbations.shape[0], -1), axis=1)

                    # Filtraggio delle immagini con L_inf ≤ 0.05
                    test_images_adv_filtered = test_images_adv[max_perturbation <= 0.05]
                    test_labels_filtered = test_labels[max_perturbation <= 0.05]

                    # Calcolo dell'accuracy sulle immagini modificate rispetto alle label vere
                    accuracy_test = compute_accuracy(classifier, test_images_adv_filtered, test_labels_filtered)
                    accuracies.append(accuracy_test)

                    # Calcolo della perturbazione media
                    perturbation = np.mean(np.abs(test_images_adv_filtered - test_labels_filtered))
                    average_perturbations.append(perturbation)

    return accuracies, average_perturbations, targeted_accuracies
