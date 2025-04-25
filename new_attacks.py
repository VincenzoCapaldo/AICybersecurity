from abc import ABC, abstractmethod
import numpy as np
import torch
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, DeepFool, CarliniLInfMethod
from utils import *

NUM_CLASSES = 8631

class AdversarialAttack(ABC):
    def __init__(self, classifierNN1, classifierNN2 , detectors, threshold):
        self.classifierNN1 = classifierNN1
        self.classifierNN2 = classifierNN2
        self.detectors = detectors
        self.threshold = threshold  # Soglia per i detector (default)

    @abstractmethod
    def generate_attack(self, images, targeted=False, target_class=0):
        pass

    @abstractmethod
    def generate_test_adv(self, images, save_dir, targeted=False, target_class=0):
        pass

    @abstractmethod
    def generate_train_adv(self, images, values, save_dir, verbose=False):
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
    def __init__(self, classifierNN1, classifierNN2=None, detectors=None, threshold=0.5):
        super().__init__(classifierNN1, classifierNN2, detectors, threshold)

    def generate_attack(self, images, epsilon, targeted=False, targeted_labels=None):
        attack = FastGradientMethod(estimator=self.classifierNN1, eps=epsilon, targeted=targeted)
        if targeted:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            return attack.generate(images)

    def generate_test_adv(self, images, epsilon_values, save_dir, targeted=False, target_class=0, verbose=False):
        targeted_labels = None
        if targeted:
            targeted_labels = target_class * torch.ones(len(images), dtype=torch.long)

        for epsilon in epsilon_values:
            # Generazione delle immagini avversarie
            test_images_adv = self.generate_attack(images, epsilon, targeted, targeted_labels)
            save_images_as_npy(test_images_adv, f"eps_{epsilon}", save_dir)

        if verbose:
            print("Test adversarial examples generated and saved successfully for fgsm.")

    def generate_train_adv(self, images, epsilon_values, save_dir, verbose=False):
        n_samples = images.shape[0]
        split_size = n_samples // len(epsilon_values)

        for i, eps in enumerate(epsilon_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values) - 1 else n_samples
                x_subset = images[start:end]
                adv_examples = self.generate_attack(x_subset, eps)
                save_images_as_npy(adv_examples, f"eps_{eps}", save_dir)
        
        if verbose:
            print("Training adversarial examples generated and saved successfully for fgsm.")
    

class BIM(AdversarialAttack):
    def __init__(self, classifierNN1, classifierNN2=None, detectors=None, threshold=0.5):
        super().__init__(classifierNN1, classifierNN2, detectors, threshold)

    def generate_attack(self, images, epsilon, epsilon_step, max_iter, targeted=False, targeted_labels=None):
        attack = BasicIterativeMethod(estimator=self.classifierNN1, eps=epsilon, eps_step=epsilon_step, max_iter=max_iter, targeted=targeted)
        if targeted:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            return attack.generate(images)

    def generate_test_adv(self, images, epsilon_values, epsilon_step_values, max_iter_values, save_dir, targeted=False, target_class=0, verbose=False):
        targeted_labels = None
        if targeted:
            targeted_labels = target_class * torch.ones(len(images), dtype=torch.long)

        for epsilon in epsilon_values:
            for epsilon_step in epsilon_step_values:
                for max_iter in max_iter_values:
                    # Generazione delle immagini avversarie
                    test_images_adv = self.generate_attack(images, epsilon, epsilon_step, max_iter, targeted, targeted_labels)
                    save_images_as_npy(test_images_adv, f"eps_{epsilon};eps-step_{epsilon_step};max_iter_{max_iter}", save_dir)
        
        if verbose:
            print("Test adversarial examples generated and saved successfully for bim.")

    def generate_train_adv(self, images, epsilon_values, save_dir, verbose=False):
        n_samples = images.shape[0]
        split_size = n_samples // len(epsilon_values)

        for i, eps in enumerate(epsilon_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values) - 1 else n_samples
                x_subset = images[start:end]
                adv_examples = self.generate_attack(x_subset, eps, epsilon_step=0.005, max_iter=10)
                save_images_as_npy(adv_examples, f"eps_{eps}", save_dir)
        
        if verbose:
            print("Training adversarial examples generated and saved successfully for bim.")
    

class PGD(AdversarialAttack):
    def __init__(self, classifierNN1, classifierNN2=None, detectors=None, threshold=0.5):
        super().__init__(classifierNN1, classifierNN2, detectors, threshold)

    def generate_attack(self, images, epsilon, epsilon_step, max_iter, targeted=False, targeted_labels=None):
        attack = ProjectedGradientDescent(estimator=self.classifierNN1, eps=epsilon, eps_step=epsilon_step, max_iter=max_iter, random_eps=True, targeted=targeted)
        if targeted:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            return attack.generate(images)

    def generate_test_adv(self, images, epsilon_values, epsilon_step_values, max_iter_values, save_dir, targeted=False, target_class=0, verbose=False):
        targeted_labels = None
        
        if targeted:
            targeted_labels = target_class * torch.ones(len(images), dtype=torch.long)
        else:
            targeted_labels = None

        for epsilon in epsilon_values:
            for epsilon_step in epsilon_step_values:
                for max_iter in max_iter_values:
                    # Generazione delle immagini avversarie
                    test_images_adv = self.generate_attack(images, epsilon, epsilon_step, max_iter, targeted, targeted_labels)
                    save_images_as_npy(test_images_adv, f"eps_{epsilon};eps-step_{epsilon_step};max_iter_{max_iter}", save_dir)

        if verbose:
            print("Test adversarial examples generated and saved successfully for pgd.")

    def generate_train_adv(self, images, epsilon_values, save_dir, verbose=False):
        n_samples = images.shape[0]
        split_size = n_samples // len(epsilon_values)

        for i, eps in enumerate(epsilon_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values) - 1 else n_samples
                x_subset = images[start:end]
                adv_examples = self.generate_attack(x_subset, eps, epsilon_step=0.005, max_iter=10)
                save_images_as_npy(adv_examples, f"eps_{eps}", save_dir)
        
        if verbose:
            print("Training adversarial examples generated and saved successfully for pgd.")


class DF(AdversarialAttack):
    def __init__(self, classifierNN1, classifierNN2=None, detectors=None, threshold=0.5):
        super().__init__(classifierNN1, classifierNN2, detectors, threshold)

    def generate_attack(self, images, epsilon, max_iter):
        attack = DeepFool(classifier=self.classifierNN1, epsilon=epsilon, max_iter=max_iter, batch_size=16)
        return attack.generate(images)

    def generate_test_adv(self, images, epsilon_values, max_iter_values, save_dir, verbose=False):
        for epsilon in epsilon_values:
                for max_iter in max_iter_values:
                    # Generazione delle immagini avversarie
                    test_images_adv = self.generate_attack(images, epsilon, max_iter)
                    save_images_as_npy(test_images_adv, f"eps_{epsilon};max_iter_{max_iter}", save_dir)

        if verbose:
            print("Test adversarial examples generated and saved successfully for df.")

    def generate_train_adv(self, images, epsilon_values, save_dir, verbose=False):
        n_samples = images.shape[0]
        split_size = n_samples // len(epsilon_values)

        for i, eps in enumerate(epsilon_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(epsilon_values) - 1 else n_samples
                x_subset = images[start:end]
                adv_examples = self.generate_attack(x_subset, eps, max_iter=5)
                save_images_as_npy(adv_examples, f"eps_{eps}", save_dir)
        
        if verbose:
            print("Training adversarial examples generated and saved successfully for df.")


class CW(AdversarialAttack):
    def __init__(self, classifierNN1, classifierNN2=None, detectors=None, threshold=0.5):
        super().__init__(classifierNN1, classifierNN2, detectors, threshold)

    def generate_attack(self, images, confidence, max_iter, learning_rate, targeted=False, targeted_labels=None):
        attack = CarliniLInfMethod(classifier=self.classifierNN1, confidence=confidence, max_iter=max_iter, 
                                   learning_rate=learning_rate, batch_size=16, targeted=targeted)
        if targeted:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            return attack.generate(images)

    def generate_test_adv(self, images, confidence_values, max_iter_values, learning_rate_values, save_dir, targeted=False, target_class=0, verbose=False):
        targeted_labels = None
        
        if targeted:
            targeted_labels = target_class * torch.ones(len(images), dtype=torch.long)
        else:
            targeted_labels = None

        for confidence in confidence_values:
            for max_iter in max_iter_values:
                for learning_rate in learning_rate_values:
                    # Generazione delle immagini avversarie
                    test_images_adv = self.generate_attack(images, confidence, max_iter, learning_rate, targeted, targeted_labels)
                    save_images_as_npy(test_images_adv, f"conf_{confidence};max_iter_{max_iter};lr_{learning_rate}", save_dir)

        if verbose:
            print("Test adversarial examples generated and saved successfully for cw.")

    def generate_train_adv(self, images, confidence_values, save_dir, verbose=False):
        n_samples = images.shape[0]
        split_size = n_samples // len(confidence_values)

        for i, conf in enumerate(confidence_values):
                start, end = i * split_size, (i + 1) * split_size if i < len(confidence_values) - 1 else n_samples
                x_subset = images[start:end]
                adv_examples = self.generate_attack(x_subset, conf, learning_rate=0.01, max_iter=5)
                save_images_as_npy(adv_examples, f"conf_{conf}", save_dir)
        
        if verbose:
            print("Training adversarial examples generated and saved successfully for cw.")