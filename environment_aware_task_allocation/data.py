from dataclasses import dataclass
import numpy as np

@dataclass
class Sample:
    exit_0_confidence: float
    exit_1_confidence: float
    returned_by: int
    answer: int
    remote_answer: int
    label: int
    preprocessing_performance: float
    exit_0_performance: float
    exit_1_performance: float
    transferring_performance: float

class Dataset():

    def __init__(self,
                 scores:np.array,
                 returned_by:np.array,
                 answers:np.array,
                 remote_answers:np.array,
                 labels:np.array,
                 energy_performance:dict,
                 seed:int=42):
        """Construct a dataset given the pre-computed metrics on a test set.

        Keyword arguments:
        returned_by -- for each sample, the stage of the system that has elaborated it
        answers -- for each sample, the answer of the system
        labels -- for each sample, the ground truth
        seed -- the seed to use to randomly pick samples
        """
        # print(f"{energy_performance=}")
        self.rng = np.random.default_rng(seed=seed)
        self.samples = []
        for idx in range(len(returned_by)):
            exit_0_confidence = self.compute_confidence(scores["exit_0"][idx])
            exit_1_confidence = self.compute_confidence(scores["exit_1"][idx])
            if len(energy_performance["wifi"]) != 0:
                energy_performance_wifi = energy_performance["wifi"][idx][1]
            else:
                energy_performance_wifi = 0
            self.samples.append(Sample(exit_0_confidence=exit_0_confidence,
                                       exit_1_confidence=exit_1_confidence,
                                       returned_by = returned_by[idx],
                                       answer = answers[idx],
                                       remote_answer = answers[idx],
                                       label = labels[idx],
                                       preprocessing_performance=energy_performance["ee_cnn"]["exit_0"]["preprocessing"][idx][1],
                                       exit_0_performance=energy_performance["ee_cnn"]["exit_0"]["inference"][idx][1],
                                       exit_1_performance=(energy_performance["ee_cnn"]["exit_1"]["inference"][idx][1] -
                                                           energy_performance["ee_cnn"]["exit_0"]["inference"][idx][1]),
                                       transferring_performance=energy_performance_wifi))
        self.n = len(self.samples)
        """
        print("DATASET")
        for s in self.samples: print(f"{s}")
        """

    def pick_random_samples(self, n:int=1):
        """Pick n samples randomly. Default n=1"""
        if n==1:
            return self.samples[self.rng.integers(0, self.n)]
        else:
            random_samples = []
            for i in range(n):
                random_samples.append(self.samples[self.rng.integers(0, self.n)])
            return random_samples

    def compute_confidence(self, scores):
        if not isinstance(scores, np.ndarray) and isinstance(scores, list):
            scores = np.array(scores)
        confidences = 2 * np.abs(scores - 0.5)
        max_confidence = np.max(confidences)
        return max_confidence

