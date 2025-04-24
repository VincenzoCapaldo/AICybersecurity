from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import torch
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, DeepFool, CarliniLInfMethod
from utils import compute_accuracy, process_images, compute_accuracy_with_detectors

NUM_CLASSES = 8631

class AdversarialAttack(ABC):
    def __init__(self, test_images, test_labels, classifierNN1, classifierNN2 , detectors):
        self.classifierNN1 = classifierNN1
        self.classifierNN2 = classifierNN2
        self.test_images = test_images
        self.test_labels = test_labels
        self.detectors = detectors
        self.threshold = 0.7  # Soglia per i detector (default)

    @abstractmethod
    def generate_attack(self, targeted=False, target_class=0):
        pass

    @abstractmethod
    def compute_security_curve(self, targeted=False, target_class=0):
        pass

    def compute_max_perturbation(self, test_images_adv):
        return np.max(np.abs(test_images_adv - self.test_images))
    
    def compute_accuracies(self, accuracies, max_perturbations, targeted_accuracies, test_images_adv, targeted=False, targeted_labels=None):
        # Calcolo della perturbazione massima
        max_perturbation = self.compute_max_perturbation(test_images_adv)
        max_perturbations.append(max_perturbation)

        # Calcolo dell'accuracy (sul classificatore NN1 + detectors) sulle immagini modificate rispetto alle label vere
        if self.detectors is not None:
            adv_labels = np.ones(test_images_adv.shape[0], dtype=bool) # Tutti i campioni sono adversarial (classe 1)
            accuracies["nn1"].append(compute_accuracy_with_detectors(self.classifierNN1, test_images_adv, self.test_labels, adv_labels, self.detectors, self.threshold)[0])
            if targeted:
                # Calcolo dell'accuracy (sul classificatore NN1 + detectors) sulle immagini modificate rispetto alle label della classe target
                targeted_accuracies["nn1"].append(compute_accuracy_with_detectors(self.classifierNN1, test_images_adv, targeted_labels, adv_labels, self.detectors, self.threshold, targeted)[0])
        else:
            # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label vere
            accuracies["nn1"].append(compute_accuracy(self.classifierNN1, test_images_adv, self.test_labels))
            if targeted:
                # Calcolo dell'accuracy (sul classificatore NN1) sulle immagini modificate rispetto alle label della classe target
                targeted_accuracies["nn1"].append(compute_accuracy(self.classifierNN1, test_images_adv, targeted_labels))
        
        # TRASFERIBILITA' DELL'ATTACCO SUL CLASSIFICATORE NN2
        if self.classifierNN2 is not None:
            # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label vere
            accuracies["nn2"].append(compute_accuracy(self.classifierNN2, process_images(test_images_adv, use_padding=False), self.test_labels))

            if targeted:
                # Calcolo dell'accuracy (sul classificatore NN2) sulle immagini modificate rispetto alle label della classe target
                targeted_accuracies["nn2"].append(compute_accuracy(self.classifierNN2, process_images(test_images_adv, use_padding=False), targeted_labels))
    

class FGSM(AdversarialAttack):
    def __init__(self, test_images, test_labels, classifierNN1, classifierNN2=None, detectors=None):
        super().__init__(test_images, test_labels, classifierNN1, classifierNN2, detectors)

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
    def __init__(self, test_images, test_labels, classifierNN1, classifierNN2=None, detectors=None):
        super().__init__(test_images, test_labels, classifierNN1, classifierNN2, detectors)

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
    

class PGD(AdversarialAttack):
    def __init__(self, test_images, test_labels, classifierNN1, classifierNN2=None, detectors=None):
        super().__init__(test_images, test_labels, classifierNN1, classifierNN2, detectors)

    def generate_attack(self, epsilon, epsilon_step, max_iter, targeted=False, targeted_labels=None):
        attack = ProjectedGradientDescent(estimator=self.classifierNN1, eps=epsilon, eps_step=epsilon_step, max_iter=max_iter, random_eps=True, targeted=targeted)
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
    

class DF(AdversarialAttack):
    def __init__(self, test_images, test_labels, classifierNN1, classifierNN2=None, detectors=None):
        super().__init__(test_images, test_labels, classifierNN1, classifierNN2, detectors)

    def generate_attack(self, epsilon, max_iter):
        attack = DeepFool(classifier=self.classifierNN1, epsilon=epsilon, max_iter=max_iter, batch_size=16)
        return attack.generate(self.test_images)

    def compute_security_curve(self, epsilon_values, max_iter_values):
        accuracies = defaultdict(list)
        max_perturbations = []
        targeted_accuracies = defaultdict(list)

        for epsilon in epsilon_values:
                for max_iter in max_iter_values:
                    # Generazione delle immagini avversarie
                    test_images_adv = self.generate_attack(epsilon, max_iter)
                    self.compute_accuracies(accuracies, max_perturbations, targeted_accuracies, test_images_adv)

        return accuracies, max_perturbations
    
class CW(AdversarialAttack):
    def __init__(self, test_images, test_labels, classifierNN1, classifierNN2=None, detectors=None):
        super().__init__(test_images, test_labels, classifierNN1, classifierNN2, detectors)

    def generate_attack(self, confidence, max_iter, learning_rate, targeted=False, targeted_labels=None):
        attack = CarliniLInfMethod(classifier=self.classifierNN1, confidence=confidence, max_iter=max_iter, 
                                   learning_rate=learning_rate, batch_size=16, targeted=targeted)
        if targeted:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(self.test_images, one_hot_targeted_labels)
        else:
            return attack.generate(self.test_images)

    def compute_security_curve(self, confidence_values, max_iter_values, learning_rate_values, targeted=False, target_class=0):
        accuracies = defaultdict(list)
        max_perturbations = []
        targeted_accuracies = defaultdict(list)

        if targeted:
            targeted_labels = target_class * torch.ones(self.test_labels.size, dtype=torch.long)
        else:
            targeted_labels = None

        for confidence in confidence_values:
            for max_iter in max_iter_values:
                for learning_rate in learning_rate_values:
                    # Generazione delle immagini avversarie
                    test_images_adv = self.generate_attack(confidence, max_iter, learning_rate, targeted, targeted_labels)
                    self.compute_accuracies(accuracies, max_perturbations, targeted_accuracies, test_images_adv, targeted, targeted_labels)
        
        return accuracies, max_perturbations, targeted_accuracies