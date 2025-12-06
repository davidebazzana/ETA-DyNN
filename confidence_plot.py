from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from ee_cnn.experiments_db import EXPERIMENTS_DB

class ConfidencePlot():
    def __init__(self,
                 train_data:np.array,
                 db_path:str,
                 experiment_codename:str,
                 dataset_codename:str,
                 model_codename:str,
                 exit_idx:int|None=None,
                 prev_pred:ConfidencePlot|None=None,
                 next_pred:ConfidencePlot|None=None):
        self.db = EXPERIMENTS_DB(db_path)
        self.test_data = self.db.get_scores(experiment_codename,
                                            dataset_codename,
                                            model_codename,
                                            exit_idx)
        self.labels = np.array(self.db.get_labels(dataset_codename))
        self.train_data = train_data
        self.returned = np.zeros_like(self.labels, dtype=bool)
        
        self.prev_pred = prev_pred
        self.next_pred = next_pred

        self.cm = None
        self.acc = None
        self.prec = None
        self.rec = None
        self.f1 = None
        
        self.fig, (self.ax_hist, self.ax_cm, self.ax_metrics) = plt.subplots(1, 3, figsize=(15, 5))
        counts, _, _ = self.ax_hist.hist(self.train_data, bins=50, edgecolor="black", density=True)
        self.max_hist_y = counts.max()

        # Initial line positions
        x1, x2 = 0, 1
        
        # Create the lines
        self.line1 = self.ax_hist.axvline(x1, color='blue', linewidth=2, picker=5)
        self.line2 = self.ax_hist.axvline(x2, color='red', linewidth=2, picker=5)
        self.lower_threshold = self.line1.get_xdata()[0]
        self.upper_threshold = self.line2.get_xdata()[0]

        self.selected_line = None

        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.update_metrics()

        self.update_plots()

        plt.tight_layout(pad=3)

    def on_pick(self, event):
        """When a line is clicked, mark it as selected."""
        if event.artist in [self.line1, self.line2]:
            self.selected_line = event.artist

    def on_mouse_move(self, event):
        """When dragging with mouse pressed, move the selected line."""
        if self.selected_line is None:
            return
        if event.inaxes != self.ax_hist:
            return

        # Update line position
        self.selected_line.set_xdata([event.xdata, event.xdata])
        self.lower_threshold = self.line1.get_xdata()[0]
        self.upper_threshold = self.line2.get_xdata()[0]
        self.update_metrics()
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        """Release the selected line."""
        self.selected_line = None

    def on_key(self, event):
        """When pressing Enter, print the x-values of the two lines."""
        if event.key == "enter":
            x1 = self.line1.get_xdata()[0]
            x2 = self.line2.get_xdata()[0]
            print(f"Line 1 = {x1},   Line 2 = {x2}")

            plt.close(self.fig)

    def compute_answers(self):
        answers = []
        for sample_preds in self.test_data:
            sample_answers = []
            for pred in sample_preds:
                if pred < self.lower_threshold: 
                    sample_answers.append(0)
                elif pred > self.upper_threshold:
                    sample_answers.append(1)
            if len(sample_answers) > 0:
                # Majority voting
                answers.append(int(sum(sample_answers)/len(sample_answers) > 0.5))
            else:
                answers.append(-1)
        answers = np.array(answers)

        valid_answers = (answers == 0) | (answers == 1)
        if self.prev_pred is not None:
            self.returned = self.prev_pred.returned & valid_answers
            curr_returned = ~self.prev_pred.returned & valid_answers
        else:
            self.returned = valid_answers
            curr_returned = valid_answers

        answers = answers[curr_returned]
        labels = self.labels[curr_returned]

        return labels, answers
            
    def compute_metrics(self):
        """
        if self.prev_pred is not None:
            preds = np.copy(self.test_data[~self.prev_pred.returned])
            labels = np.copy(self.labels[~self.prev_pred.returned])
        else:
            preds = np.copy(self.test_data)
            labels = np.copy(self.labels)
        print("PREDS", preds)
        cond = (preds < self.lower_threshold) | (preds > self.upper_threshold)
        self.returned = cond
        preds = preds[cond]
        labels = labels[cond]
        preds[preds > self.upper_threshold] = 1
        preds[preds < self.lower_threshold] = 0
        """
        labels, answers = self.compute_answers()
        if len(answers) > 0:
            cm = confusion_matrix(labels, answers)

            if cm.shape == (2, 2):
                acc = accuracy_score(labels, answers)
                prec = precision_score(labels, answers)
                rec = recall_score(labels, answers)
                f1 = f1_score(labels, answers)

                return cm, acc, prec, rec, f1

        return None
            
    def update_metrics(self):
        """Recompute and redraw the metrics based on threshold."""
        res = self.compute_metrics()

        if res is not None:
            self.cm, self.acc, self.prec, self.rec, self.f1 = res
            self.update_plots()

        if self.next_pred is not None:
            self.next_pred.update_metrics()

    def show_plot(self):
        plt.show()

    def update_plots(self):
        if self.cm is not None:
            self.ax_cm.clear()
            self.ax_cm.imshow(self.cm, cmap="Blues")
            # Cell values
            for i in range(2):
                for j in range(2):
                    color = "white" if self.cm[i, j] > self.cm.max() / 2 else "black"
                    self.ax_cm.text(j, i, self.cm[i, j], ha='center', va='center', fontsize=14, color=color)
        self.ax_cm.set_title(f"Confusion Matrix\n(lower threshold={self.lower_threshold:.3f}, upper threshold={self.upper_threshold:.3f})")
        self.ax_cm.set_xlabel("Predicted")
        self.ax_cm.set_ylabel("Ground Truth")
        self.ax_cm.set_xticks([0, 1], labels=["0", "1"])
        self.ax_cm.set_yticks([0, 1], labels=["0", "1"])
        
        self.ax_metrics.clear()
        self.ax_metrics.axis('off')
        self.ax_metrics.set_title("Metrics")

        if self.acc is not None and self.prec is not None and self.rec is not None and self.f1 is not None:
            metrics_text = (
                f"Accuracy:  {self.acc:.4f}\n"
                f"Precision: {self.prec:.4f}\n"
                f"Recall:    {self.rec:.4f}\n"
                f"F1 Score:  {self.f1:.4f}\n\n"
                f"Percentage returned: {np.sum(self.returned)/len(self.test_data)*100:.2f}%"
            )
        else:
            metrics_text = (
                f"Accuracy:  N/A\n"
                f"Precision: N/A\n"
                f"Recall:    N/A\n"
                f"F1 Score:  N/A\n\n"
                f"Percentage returned: 0.0%"
            )
        self.ax_metrics.text(0.05, 0.95, metrics_text, va='top', fontsize=13, family="monospace")
            
        self.fig.canvas.draw_idle()

    def get_thresholds(self):
        return self.lower_threshold, self.upper_threshold

    def set_pred_pred(self, pred_pred:ConfidencePlot):
        self.pred_pred = pred_pred

    def set_next_pred(self, next_pred:ConfidencePlot):
        self.next_pred = next_pred

    def set_hist_ylim(self, lower_lim, upper_lim):
        self.ax_hist.set_ylim(lower_lim, upper_lim)
