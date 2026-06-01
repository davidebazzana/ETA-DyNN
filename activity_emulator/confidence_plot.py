from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from ee_cnn.experiments_db import EXPERIMENTS_DB

class ConfidencePlot():
    def __init__(self,
                 train_data:np.array,
                 train_labels:np.array,
                 db_path:str,
                 experiment_codename:str,
                 dataset_codename:str,
                 model_codename:str,
                 exit_idx:int|None=None,
                 prev_pred:ConfidencePlot|None=None,
                 next_pred:ConfidencePlot|None=None):
        self.db = EXPERIMENTS_DB(db_path)
        print(f"{model_codename=}")
        print(f"{exit_idx=}")
        self.test_data = self.db.get_scores(experiment_codename,
                                            dataset_codename,
                                            model_codename,
                                            exit_idx)
        self.labels = np.array(self.db.get_labels(dataset_codename))
        self.train_data = train_data
        self.train_labels = train_labels
        self.returned = np.zeros_like(self.labels, dtype=bool)
        self.curr_returned = np.zeros_like(self.labels, dtype=bool)
        self.complete_answers = None

        self.exit_idx = exit_idx
        
        self.prev_pred = prev_pred
        self.next_pred = next_pred

        self.final_plot = None

        self.cm = None
        self.acc = None
        self.prec = None
        self.rec = None
        self.f1 = None

        self.fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(2, 3)

        self.ax_hist = self.fig.add_subplot(gs[:, 0])
        self.ax_train_cm = self.fig.add_subplot(gs[0, 1])
        self.ax_train_metrics = self.fig.add_subplot(gs[1, 1])
        self.ax_test_cm = self.fig.add_subplot(gs[0, 2])
        self.ax_test_metrics = self.fig.add_subplot(gs[1, 2])
        
        # self.fig, (self.ax_hist, self.ax_cm, self.ax_metrics) = plt.subplots(2, 3, figsize=(15, 5))
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

        self.update_training_testing_plots()

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

    def set_thresholds(self, lower_threshold, upper_threshold):
        self.line1.set_xdata([lower_threshold, lower_threshold])
        self.line2.set_xdata([upper_threshold, upper_threshold])
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
            if (type(sample_preds) is float or type(sample_preds) is int): sample_preds = [sample_preds]
            print(f"{sample_preds=}")
            for pred in sample_preds:
                if pred < self.lower_threshold: 
                    sample_answers.append(0)
                elif pred > self.upper_threshold:
                    sample_answers.append(1)
            
            if len(sample_answers) > 0:
                # Majority voting heuristic
                # answers.append(int(sum(sample_answers)/len(sample_answers) > 0.5))
                # Exist positive heuristic
                answers.append(int(sum(sample_answers) >= 1))
            else:
                answers.append(-1)
        answers = np.array(answers)
        self.complete_answers = np.copy(answers)

        valid_answers = (answers == 0) | (answers == 1)
        if self.prev_pred is not None:
            self.returned = self.prev_pred.returned | valid_answers
            self.curr_returned = ~self.prev_pred.returned & valid_answers
        else:
            self.returned = valid_answers
            self.curr_returned = valid_answers

        answers = answers[self.curr_returned]
        labels = self.labels[self.curr_returned]

        return labels, answers

    def compute_training_answers(self):
        valid_answers = (self.train_data < self.lower_threshold) | (self.train_data > self.upper_threshold)

        answers = self.train_data[valid_answers]
        labels = self.train_labels[valid_answers]
        answers[answers > self.upper_threshold] = 1
        answers[answers < self.lower_threshold] = 0
        
        return labels, answers
    
    def compute_metrics(self):
        self.training_metrics = None
        self.testing_metrics = None
        
        training_labels, training_answers = self.compute_training_answers()
        labels, answers = self.compute_answers()
        if len(training_answers) > 0:
            cm = confusion_matrix(training_labels, training_answers)

            if cm.shape == (2, 2):
                acc = accuracy_score(training_labels, training_answers)
                prec = precision_score(training_labels, training_answers)
                rec = recall_score(training_labels, training_answers)
                f1 = f1_score(training_labels, training_answers)

                self.training_metrics = {
                    "cm": cm,
                    "acc": acc,
                    "prec": prec,
                    "rec": rec,
                    "f1": f1
                }

            self.valid_training_answers = len(training_labels) / len(self.train_labels)
            print(f"Exit {self.exit_idx}: {self.valid_training_answers=}")

        if len(answers) > 0:
            cm = confusion_matrix(labels, answers)

            if cm.shape == (2, 2):
                acc = accuracy_score(labels, answers)
                prec = precision_score(labels, answers)
                rec = recall_score(labels, answers)
                f1 = f1_score(labels, answers)

                self.testing_metrics = {
                    "cm": cm,
                    "acc": acc,
                    "prec": prec,
                    "rec": rec,
                    "f1": f1
                }
            
    def update_metrics(self):
        """Recompute and redraw the metrics based on threshold."""
        self.compute_metrics()

        self.update_training_testing_plots()

        if self.next_pred is not None:
            self.next_pred.update_metrics()

        if self.final_plot is not None:
            self.final_plot.update_plot()

    def show_plot(self):
        plt.show()

    def update_training_testing_plots(self):
        self.update_plots(self.training_metrics, self.ax_train_cm, self.ax_train_metrics, "training")
        self.update_plots(self.testing_metrics, self.ax_test_cm, self.ax_test_metrics, "testing")
        
    def update_plots(self, metrics, ax_cm, ax_metrics, stage):
        if metrics is not None:
            ax_cm.clear()
            ax_cm.imshow(metrics["cm"], cmap="Blues")
            # Cell values
            for i in range(2):
                for j in range(2):
                    color = "white" if metrics["cm"][i, j] > metrics["cm"].max() / 2 else "black"
                    ax_cm.text(j, i, metrics["cm"][i, j], ha='center', va='center', fontsize=9, color=color)
            
            metrics_text = (
                f'Accuracy:  {metrics["acc"]:.4f}\n'
                f'Precision: {metrics["prec"]:.4f}\n'
                f'Recall:    {metrics["rec"]:.4f}\n'
                f'F1 Score:  {metrics["f1"]:.4f}\n\n'           
            )
            if stage == "testing":
                metrics_text += f'Percentage returned: {np.sum(self.returned)/len(self.test_data)*100:.2f}%'
        else:
            metrics_text = (
                f"Accuracy:  N/A\n"
                f"Precision: N/A\n"
                f"Recall:    N/A\n"
                f"F1 Score:  N/A\n\n"
            )
            if stage == "testing":
                metrics_text += f"Percentage returned: 0.0%"
        ax_cm.set_title(f"Confusion Matrix\n(lower threshold={self.lower_threshold:.3f}, upper threshold={self.upper_threshold:.3f})", fontsize=11)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Ground Truth")
        ax_cm.set_xticks([0, 1], labels=["0", "1"])
        ax_cm.set_yticks([0, 1], labels=["0", "1"])
        
        ax_metrics.clear()
        ax_metrics.axis('off')
        ax_metrics.set_title("Metrics", fontsize=11)
        ax_metrics.text(0.05, 0.95, metrics_text, va='top', fontsize=9, family="monospace")
            
        self.fig.canvas.draw_idle()

    def get_thresholds(self):
        return self.lower_threshold, self.upper_threshold

    def set_pred_pred(self, pred_pred:ConfidencePlot):
        self.pred_pred = pred_pred

    def set_next_pred(self, next_pred:ConfidencePlot):
        self.next_pred = next_pred

    def set_hist_ylim(self, lower_lim, upper_lim):
        self.ax_hist.set_ylim(lower_lim, upper_lim)

    def set_final_plot(self, final_plot):
        self.final_plot = final_plot
        
    def close(self):
        self.db.close()
