from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class ConfidencePlot():
    def __init__(self,
                 train_data:np.array,
                 test_data:np.array,
                 labels:np.array,
                 title:str|None=None,
                 prev_pred:ConfidencePlot|None=None,
                 next_pred:ConfidencePlot|None=None):
        self.train_data = train_data
        self.test_data = test_data
        self.labels = labels
        self.returned = np.zeros_like(self.test_data, dtype=bool)
        self.title = title
        
        self.prev_pred = prev_pred
        self.next_pred = next_pred
        
        self.fig, (self.ax_hist, self.ax_cm, self.ax_metrics) = plt.subplots(1, 3, figsize=(15, 5))
        self.ax_hist.hist(self.train_data, bins=50, edgecolor="black")
        
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

        self.update_confusion_matrix()

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
        self.update_confusion_matrix()
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

    def compute_metrics(self, lower_threshold, upper_threshold):
        if self.prev_pred is not None:
            preds = np.copy(self.test_data[~self.prev_pred.returned])
            labels = np.copy(self.labels[~self.prev_pred.returned])
        else:
            preds = np.copy(self.test_data)
            labels = np.copy(self.labels)
        cond = (preds < lower_threshold) | (preds > upper_threshold)
        self.returned = cond
        preds = preds[cond]
        labels = labels[cond]
        preds[preds > upper_threshold] = 1
        preds[preds < lower_threshold] = 0
        if len(preds) > 0:
            cm = confusion_matrix(labels, preds)

            if cm.shape == (2, 2):
                acc = accuracy_score(labels, preds)
                prec = precision_score(labels, preds)
                rec = recall_score(labels, preds)
                f1 = f1_score(labels, preds)

                return cm, acc, prec, rec, f1

        return None
            
    def update_confusion_matrix(self):
        """Recompute and redraw the confusion matrix based on threshold."""
        res = self.compute_metrics(self.lower_threshold, self.upper_threshold)

        if res is not None:
            cm, acc, prec, rec, f1 = res
            self.ax_cm.clear()
            self.ax_cm.imshow(cm, cmap="Blues")
            self.ax_cm.set_title(f"Confusion Matrix\n(lower threshold={self.lower_threshold:.3f}, upper threshold={self.upper_threshold:.3f})")
            self.ax_cm.set_xlabel("Predicted")
            self.ax_cm.set_ylabel("Ground Truth")
            self.ax_cm.set_xticks([0, 1], labels=["0", "1"])
            self.ax_cm.set_yticks([0, 1], labels=["0", "1"])

            # Cell values
            for i in range(2):
                for j in range(2):
                    self.ax_cm.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)

            self.ax_metrics.clear()
            self.ax_metrics.axis('off')
            self.ax_metrics.set_title("Metrics")

            metrics_text = (
                f"Accuracy:  {acc:.4f}\n"
                f"Precision: {prec:.4f}\n"
                f"Recall:    {rec:.4f}\n"
                f"F1 Score:  {f1:.4f}\n\n"
                f"Percentage returned: {np.sum(self.returned)/len(self.test_data)*100:.2f}%"
            )
            self.ax_metrics.text(0.05, 0.95, metrics_text, va='top', fontsize=13, family="monospace")
                    
            self.fig.canvas.draw_idle()

        if self.next_pred is not None:
            self.next_pred.update_confusion_matrix()

    def choose_thresholds(self, show=True):
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
        plt.title("Histogram of Confidence" if self.title is None else self.title)

        plt.tight_layout(pad=3)
        if show:
            plt.show()

    def get_thresholds(self):
        return self.lower_threshold, self.upper_threshold

    def set_pred_pred(self, pred_pred:ConfidencePlot):
        self.pred_pred = pred_pred

    def set_next_pred(self, next_pred:ConfidencePlot):
        self.next_pred = next_pred
