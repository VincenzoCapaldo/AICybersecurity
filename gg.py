from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import torch
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod
from utils import compute_accuracy, process_images

NUM_CLASSES = 8631

class AdversarialAttack(ABC):
    def __init__(self, test_images, test_labels, classifierNN1, classifierNN2):
        self.classifierNN1 = classifierNN1
        self.classifierNN2 = classifierNN2
        self.test_images = test_images
        self.test_labels = test_labels

    @abstractmethod
    def generate_attack(self, targeted=False, target_class=0):
        pass

    @abstractmethod
    def compute_security_curve(self, targeted=False, target_class=0):
        pass

    def compute_max_perturbation(self, test_images_adv):
        return np.max(np.abs(test_images_adv - self.test_images))
    
    def compute_accuracies(self, accuracies, max_perturbations, targeted_accuracies, test_images_adv, targeted=False, targeted_labels=None):
        # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label vere
        accuracies["nn1"].append(compute_accuracy(self.classifierNN1, test_images_adv, self.test_labels))

        # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label della classe target
        if targeted:
            targeted_attack_accuracy = compute_accuracy(self.classifierNN1, test_images_adv, targeted_labels)
            targeted_accuracies["nn1"].append(targeted_attack_accuracy)
        
        # TRASFERIBILITA' DELL'ATTACCO SUL CLASSIFICATORE NN2
        if self.classifierNN2 is not None:
            # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label vere
            accuracy = compute_accuracy(self.classifierNN2, process_images(test_images_adv, use_padding=False), self.test_labels)
            accuracies["nn2"].append(accuracy)

            if targeted:
            # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label della classe target
                targeted_attack_accuracy = compute_accuracy(self.classifierNN2, process_images(test_images_adv, use_padding=False), targeted_labels)
                targeted_accuracies["nn2"].append(targeted_attack_accuracy)

        # Calcolo della perturbazione massima
        max_perturbation = self.compute_max_perturbation(test_images_adv)
        max_perturbations.append(max_perturbation)
    

class FGSM(AdversarialAttack):
    def __init__(self, test_images, test_labels, classifierNN1, classifierNN2=None):
        super().__init__(test_images, test_labels, classifierNN1, classifierNN2)

    def generate_attack(self, epsilon, targeted=False, targeted_labels=None):
        attack = FastGradientMethod(estimator=self.classifierNN1, eps=epsilon, targeted=targeted)
        if targeted:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(self.test_images, one_hot_targeted_labels)
        else:
            return attack.generate(self.test_images)

    def compute_security_curve(self, epsilon_values, targeted=False, target_class=0):
        accuracies = defaultdict(list)
        max_perturbations = []
        targeted_accuracies = defaultdict(list)
        targeted_labels = None
        if targeted:
            targeted_labels = target_class * torch.ones(self.test_labels.size, dtype=torch.long)

        for epsilon in epsilon_values:
            # Generazione delle immagini avversarie
            test_images_adv = self.generate_attack(epsilon, targeted, targeted_labels)
            self.compute_accuracies(accuracies, max_perturbations, targeted_accuracies, test_images_adv, targeted, targeted_labels)

        return accuracies, max_perturbations, targeted_accuracies
    

class BIM(AdversarialAttack):
    def __init__(self, test_images, test_labels, classifierNN1, classifierNN2=None):
        super().__init__(test_images, test_labels, classifierNN1, classifierNN2)

    def generate_attack(self, epsilon, epsilon_step, max_iter, targeted=False, targeted_labels=None):
        attack = BasicIterativeMethod(estimator=self.classifierNN1, eps=epsilon, eps_step=epsilon_step, max_iter=max_iter, targeted=targeted)
        if targeted:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(self.test_images, one_hot_targeted_labels)
        else:
            return attack.generate(self.test_images)

    def compute_security_curve(self, epsilon_values, epsilon_step_values, max_iter_values, targeted=False, target_class=0):
        accuracies = defaultdict(list)
        max_perturbations = []
        targeted_accuracies = defaultdict(list)

        if targeted:
            targeted_labels = target_class * torch.ones(self.test_labels.size, dtype=torch.long)
        else:
            targeted_labels = None

        for epsilon in epsilon_values:
            for epsilon_step in epsilon_step_values:
                for max_iter in max_iter_values:
                    # Generazione delle immagini avversarie
                    test_images_adv = self.generate_attack(epsilon, epsilon_step, max_iter, targeted, targeted_labels)
                    self.compute_accuracies(accuracies, max_perturbations, targeted_accuracies, test_images_adv, targeted, targeted_labels)
        return accuracies, max_perturbations, targeted_accuracies