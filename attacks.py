from abc import ABC, abstractmethod
import numpy as np
import torch
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, DeepFool, CarliniLInfMethod
from utils import *
from tqdm import tqdm

NUM_CLASSES = 8631

class AdversarialAttack(ABC):
    def __init__(self, classifierNN1):
        self.classifierNN1 = classifierNN1

    @abstractmethod
    def generate_attack(self, images, targeted=False, target_class=0):
        pass
    
    
class FGSM(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_attack(self, images, epsilon, targeted=False, targeted_labels=None):
        attack = FastGradientMethod(estimator=self.classifierNN1, eps=epsilon, targeted=targeted)
        if targeted:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            return attack.generate(images)    

class BIM(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_attack(self, images, epsilon, epsilon_step, max_iter, targeted=False, targeted_labels=None):
        attack = BasicIterativeMethod(estimator=self.classifierNN1, eps=epsilon, eps_step=epsilon_step, max_iter=max_iter, targeted=targeted)
        if targeted:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            return attack.generate(images)    

class PGD(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_attack(self, images, epsilon, epsilon_step, max_iter, targeted=False, targeted_labels=None):
        attack = ProjectedGradientDescent(estimator=self.classifierNN1, eps=epsilon, eps_step=epsilon_step, max_iter=max_iter, targeted=targeted)
        if targeted:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            return attack.generate(images)

class DF(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_attack(self, images, epsilon, nb_grads, max_iter, verbose):
        test_images_adv = []
        batch_dim = 1 # Numero di immagini da processare nel batch
        attack = DeepFool(classifier=self.classifierNN1, epsilon=epsilon, max_iter=max_iter, nb_grads=nb_grads, verbose=False)
        
        for j in tqdm(range (0, len(images), batch_dim), desc="DeepFool"):
            batch = images[j:j+batch_dim]  # Prendi batch_dim immagini
            test_images_adv.append(attack.generate(batch))

        # Concatenazione immagini adv
        test_images_adv = np.concatenate(test_images_adv, axis=0)
        return test_images_adv

class CW(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_attack(self, images, confidence, max_iter, learning_rate, targeted=False, targeted_labels=None):
        attack = CarliniLInfMethod(classifier=self.classifierNN1, confidence=confidence, max_iter=max_iter, learning_rate=learning_rate, initial_const=0.1, targeted=targeted)
        if targeted:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            return attack.generate(images)