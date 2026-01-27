import os
from datetime import date
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Annotate numbers
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    return fig

class ReturnedPredictionsQuality():
    def __init__(self, number_of_exits:int, lower_return_threshold:float=0.5, upper_return_threshold:float=0.5):
        # Labels
        self.labels = []
        # Predictions relative to the returned samples.
        # Each exit has its own predictions.
        self.exits_predictions = dict.fromkeys(range(number_of_exits), [])
        self.number_of_exits = number_of_exits
        self.lower_return_threshold = lower_return_threshold
        self.upper_return_threshold = upper_return_threshold

    def update(self, labels, exits_predictions):
        labels = labels.cpu().numpy()
        self.labels = np.append(self.labels, labels)
        if self.number_of_exits == 1:
            exits_predictions.cpu().numpy()
            self.exits_predictions[0] = np.append(self.exits_predictions[0], exits_prediction)
        for idx in self.exits_predictions:
            try:
                predictions = exits_predictions[idx]
            except IndexError:
                raise IndexError(f"Number of exits out of bound: \
                expected {self.number_of_exits} exits, \
                but {len(exits_predictions)} were found.")
            predictions = predictions.cpu().detach().numpy()
            self.exits_predictions[idx] = np.append(self.exits_predictions[idx], predictions)

    def compute_metrics(self, summary_writer:SummaryWriter=None, dataset:str=None, epoch:int=None):
        res = {}
        for idx, predictions in self.exits_predictions.items():
            print(f'-------------------- Metrics for exit at index {idx} --------------------')
            predictions = (predictions > self.upper_return_threshold).astype(float)
            acc = accuracy_score(self.labels, predictions)
            precision = precision_score(self.labels, predictions)
            recall = recall_score(self.labels, predictions)
            f1 = f1_score(self.labels, predictions)
            cm = confusion_matrix(self.labels, predictions)

            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print("Confusion Matrix:\n", cm)

            if summary_writer is not None and dataset is not None and epoch is not None:
                summary_writer.add_scalar(f'Accuracy/{idx}/{dataset}', acc, epoch)
                summary_writer.add_scalar(f'Precision/{idx}/{dataset}', precision, epoch)
                summary_writer.add_scalar(f'Recall/{idx}/{dataset}', recall, epoch)
                summary_writer.add_scalar(f'F1-Score/{idx}/{dataset}', f1, epoch)
                fig = plot_confusion_matrix(cm, ['Free', 'Infested'])
                summary_writer.add_figure(f'Confusion Matrix/{idx}/{dataset}', fig)

            res[idx] = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

        return res

    def extract_scores(self, model_name:str="model", scores_directory:str="./scores/"):
        today = str(date.today())
        scores_directory = os.path.join(scores_directory, today)
        Path(scores_directory).mkdir(parents=True, exist_ok=True)
        print(f'-------------------- Writing labels --------------------')
        labels_file = os.path.join(scores_directory, '{}_labels'.format(model_name))
        np.save(labels_file, self.labels)
        scores = []
        for idx, predictions in self.exits_predictions.items():
            print(f'-------------------- Writing scores for exit at index {idx} --------------------')
            exit_scores = ReturnedPredictionsQuality.props_to_scores(predictions)
            scores_file = os.path.join(scores_directory, '{}_{}_scores'.format(model_name, idx))
            scores.append({
                "exit_idx": idx,
                "scores": exit_scores
            })
            np.save(scores_file, exit_scores)
        return scores, self.labels

    @staticmethod
    def props_to_scores(props):
        return np.stack([1 - props, props], axis=1)
