from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from ee_cnn.experiments_db import EXPERIMENTS_DB
from confidence_plot import ConfidencePlot

class SummaryPlot():
    def __init__(self,
                 db_path:str,
                 confidence_plot_exit_0:ConfidencePlot,
                 confidence_plot_exit_1:ConfidencePlot,
                 experiment_codename:str,
                 dataset_codename:str):
        self.db = EXPERIMENTS_DB(db_path)
        performance_exit_0 = self.db.get_performance(experiment_codename,
                                                     dataset_codename,
                                                     "ee_cnn",
                                                     0)
        performance_exit_1 = self.db.get_performance(experiment_codename,
                                                     dataset_codename,
                                                     "ee_cnn",
                                                     1)
        performance_vit4v = self.db.get_performance(experiment_codename,
                                                    dataset_codename,
                                                    "vit4v")
        self.energy_performance = {
            "ee_cnn": {
                "exit_0": {
                    "preprocessing": np.array(performance_exit_0["preprocessing"]),
                    "inference": np.array(performance_exit_0["inference"])
                },
                "exit_1": {
                    "preprocessing": np.array(performance_exit_1["preprocessing"]),
                    "inference": np.array(performance_exit_1["inference"])
                },
            },
            "vit4v": {
                "preprocessing": np.array(performance_vit4v["preprocessing"]),
                "inference": np.array(performance_vit4v["inference"])                
            }
        }
        self.labels = np.array(self.db.get_labels(dataset_codename))
        self.vit4v_answers = np.array(self.db.get_scores(experiment_codename,
                                                         dataset_codename,
                                                         "vit4v"))

        self.cp_e0 = confidence_plot_exit_0
        self.cp_e1 = confidence_plot_exit_1
        self.cp_e0.set_final_plot(self)
        self.cp_e1.set_final_plot(self)

        self.fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(2, 2)

        # self.ax_energy_hist = self.fig.add_subplot(gs[0, 0])
        self.ax_preprocessing_costs = self.fig.add_subplot(gs[0, 0])
        self.ax_inference_costs = self.fig.add_subplot(gs[1, 0])
        self.ax_total_cm = self.fig.add_subplot(gs[0, 1])
        self.ax_total_metrics = self.fig.add_subplot(gs[1, 1])
        
        self.update_plot()

        plt.tight_layout(pad=3)
        
    def retrieve_answers(self):
        self.returned_by_e0 = self.cp_e0.curr_returned
        self.returned_by_e1 = self.cp_e1.curr_returned
        self.returned_by_vit4v = ~(self.returned_by_e0 | self.returned_by_e1)

        self.e0_answers = self.cp_e0.complete_answers
        self.e1_answers = self.cp_e1.complete_answers

        self.global_answers = np.zeros_like(self.vit4v_answers)
        self.global_answers[self.returned_by_e0] = self.e0_answers[self.returned_by_e0]
        self.global_answers[self.returned_by_e1] = self.e1_answers[self.returned_by_e1]
        self.global_answers[self.returned_by_vit4v] = self.vit4v_answers[self.returned_by_vit4v]

    def compute_costs(self, p_metric, phase):
        p_idx = {
            "duration": 0,
            "tot_energy": 1,
            "cpu_energy": 2,
            "gpu_energy": 3,
            "ram_energy": 4
        }
        baseline_cost = np.copy(self.energy_performance["vit4v"][phase][:,p_idx[p_metric]])
        exit_0_cost = np.copy(self.energy_performance["ee_cnn"]["exit_0"][phase][:,p_idx[p_metric]])
        exit_1_cost = np.copy(self.energy_performance["ee_cnn"]["exit_1"][phase][:,p_idx[p_metric]])
        vit4v_cost = np.copy(baseline_cost + exit_1_cost)

        """
        if p_metric == "tot_energy" and phase == "inference":
            print(f"[TEST BEFORE] duration\nBASELINE: {baseline_cost}\nEXIT 0: {exit_0_cost}\nEXIT 1: {exit_1_cost}\nViT4V: {vit4v_cost}")
            print(f"[TEST BEFORE] Returned by e0: {~self.returned_by_e0}\nReturned by e1: {~self.returned_by_e1}\nReturned by ViT4V: {~self.returned_by_vit4v}")
        """
        
        exit_0_cost[~self.returned_by_e0] = 0
        exit_1_cost[~self.returned_by_e1] = 0
        vit4v_cost[~self.returned_by_vit4v] = 0

        # saving_performance = baseline_cost - (exit_0_cost + exit_1_cost + vit4v_cost)
        complete_costs_vector = (exit_0_cost + exit_1_cost + vit4v_cost)

        """
        if p_metric == "tot_energy" and phase == "inference":
            print(f"[TEST AFTER] duration\nBASELINE: {baseline_cost}\nEXIT 0: {exit_0_cost}\nEXIT 1: {exit_1_cost}\nViT4V: {vit4v_cost}")
        """
        
        return np.sum(complete_costs_vector), np.sum(baseline_cost)

    def retrieve_costs(self):
        duration_pre, duration_pre_baseline = self.compute_costs("duration", "preprocessing")
        duration_inf, duration_inf_baseline = self.compute_costs("duration", "inference")
        tot_energy_pre, tot_energy_pre_baseline = self.compute_costs("tot_energy", "preprocessing")
        tot_energy_inf, tot_energy_inf_baseline = self.compute_costs("tot_energy", "inference")
        cpu_energy_pre, cpu_energy_pre_baseline = self.compute_costs("cpu_energy", "preprocessing")
        cpu_energy_inf, cpu_energy_inf_baseline = self.compute_costs("cpu_energy", "inference")
        gpu_energy_pre, gpu_energy_pre_baseline = self.compute_costs("gpu_energy", "preprocessing")
        gpu_energy_inf, gpu_energy_inf_baseline = self.compute_costs("gpu_energy", "inference")
        ram_energy_pre, ram_energy_pre_baseline = self.compute_costs("ram_energy", "preprocessing")
        ram_energy_inf, ram_energy_inf_baseline = self.compute_costs("ram_energy", "inference")
        
        self.costs = {
            "duration": {
                "baseline": {
                    "preprocessing": duration_pre_baseline,
                    "inference": duration_inf_baseline
                },
                "system": {
                    "preprocessing": duration_pre,
                    "inference": duration_inf
                }
            },
            "tot_energy": {
                "baseline": {
                    "preprocessing": tot_energy_pre_baseline,
                    "inference": tot_energy_inf_baseline
                },
                "system": {
                    "preprocessing": tot_energy_pre,
                    "inference": tot_energy_inf
                }
            },
            "cpu_energy": {
                "baseline": {
                    "preprocessing": cpu_energy_pre_baseline,
                    "inference": cpu_energy_inf_baseline
                },
                "system": {
                    "preprocessing": cpu_energy_pre,
                    "inference": cpu_energy_inf
                }
            },
            "gpu_energy": {
                "baseline": {
                    "preprocessing": gpu_energy_pre_baseline,
                    "inference": gpu_energy_inf_baseline
                },
                "system": {
                    "preprocessing": gpu_energy_pre,
                    "inference": gpu_energy_inf
                }
            },
            "ram_energy": {
                "baseline": {
                    "preprocessing": ram_energy_pre_baseline,
                    "inference": ram_energy_inf_baseline
                },
                "system": {
                    "preprocessing": ram_energy_pre,
                    "inference": ram_energy_inf
                }
            }
        }
    
    def compute_metrics(self):
        self.retrieve_answers()
        self.retrieve_costs()

        cm = confusion_matrix(self.labels, self.global_answers)

        if cm.shape == (2, 2):
            acc = accuracy_score(self.labels, self.global_answers)
            prec = precision_score(self.labels, self.global_answers)
            rec = recall_score(self.labels, self.global_answers)
            f1 = f1_score(self.labels, self.global_answers)

            self.metrics = {
                "cm": cm,
                "acc": acc,
                "prec": prec,
                "rec": rec,
                "f1": f1
            }

    def update_plot(self):
        self.compute_metrics()

        """
        self.ax_energy_hist.hist(self.costs["duration"], bins=50, alpha=0.5, label='duraiton')
        self.ax_energy_hist.hist(self.costs["tot_energy"], bins=50, alpha=0.5, label='tot_energy')
        self.ax_energy_hist.hist(self.costs["cpu_energy"], bins=50, alpha=0.5, label='cpu_energy')
        self.ax_energy_hist.hist(self.costs["gpu_energy"], bins=50, alpha=0.5, label='gpu_energy')
        self.ax_energy_hist.hist(self.costs["ram_energy"], bins=50, alpha=0.5, label='ram_energy')
        """
        self.ax_total_cm.clear()
        self.ax_total_cm.imshow(self.metrics["cm"], cmap="Blues")
        # Cell values
        for i in range(2):
            for j in range(2):
                color = "white" if self.metrics["cm"][i, j] > self.metrics["cm"].max() / 2 else "black"
                self.ax_total_cm.text(j, i, self.metrics["cm"][i, j], ha='center', va='center', fontsize=9, color=color)
        metrics_text = (
            f'Accuracy:  {self.metrics["acc"]:.4f}\n'
            f'Precision: {self.metrics["prec"]:.4f}\n'
            f'Recall:    {self.metrics["rec"]:.4f}\n'
            f'F1 Score:  {self.metrics["f1"]:.4f}\n\n'
            f'Percentage returned: {np.sum(self.returned_by_e0|self.returned_by_e1|self.returned_by_vit4v)/len(self.labels)*100:.2f}%'
        )
        """
        sum_duration = np.sum(self.costs["duration"]["system"])
        sum_duration_baseline = np.sum(self.costs["duration"]["baseline"])
        sum_tot_energy = np.sum(self.costs["tot_energy"]["system"])
        sum_tot_energy_baseline = np.sum(self.costs["tot_energy"]["baseline"])
        sum_cpu_energy = np.sum(self.costs["cpu_energy"]["system"])
        sum_cpu_energy_baseline = np.sum(self.costs["cpu_energy"]["baseline"])
        sum_gpu_energy = np.sum(self.costs["gpu_energy"]["system"])
        sum_gpu_energy_baseline = np.sum(self.costs["gpu_energy"]["baseline"])
        sum_ram_energy = np.sum(self.costs["ram_energy"]["system"])
        sum_ram_energy_baseline = np.sum(self.costs["ram_energy"]["baseline"])
        costs_text = (
            f'Duration: {sum_duration:.4f}       Baseline: {sum_duration_baseline:.4f}   Savings: {((sum_duration_baseline - sum_duration)/sum_duration_baseline)*100:.2f}%\n'
            f'Tot energy: {sum_tot_energy:.4f}   Baseline: {sum_tot_energy_baseline:.4f}   Savings: {((sum_tot_energy_baseline - sum_tot_energy)/sum_tot_energy_baseline)*100:.2f}%\n'
            f'CPU energy: {sum_cpu_energy:.4f}   Baseline: {sum_cpu_energy_baseline:.4f}   Savings: {((sum_cpu_energy_baseline - sum_cpu_energy)/sum_cpu_energy_baseline)*100:.2f}%\n'
            f'GPU energy: {sum_gpu_energy:.4f}   Baseline: {sum_gpu_energy_baseline:.4f}   Savings: {((sum_gpu_energy_baseline - sum_gpu_energy)/sum_gpu_energy_baseline)*100:.2f}%\n'
            f'RAM energy: {sum_ram_energy:.4f}   Baseline: {sum_ram_energy_baseline:.4f}   Savings: {((sum_ram_energy_baseline - sum_ram_energy)/sum_ram_energy_baseline)*100:.2f}%'
        )
        """

        self.ax_total_cm.set_title(f"Confusion Matrix", fontsize=11)
        self.ax_total_cm.set_xlabel("Predicted")
        self.ax_total_cm.set_ylabel("Ground Truth")
        self.ax_total_cm.set_xticks([0, 1], labels=["0", "1"])
        self.ax_total_cm.set_yticks([0, 1], labels=["0", "1"])
        
        self.ax_total_metrics.clear()
        self.ax_total_metrics.axis('off')
        self.ax_total_metrics.set_title("Metrics", fontsize=11)
        self.ax_total_metrics.text(0.05, 0.95, metrics_text, va='top', fontsize=9, family="monospace")

        self.ax_preprocessing_costs.clear()
        self.ax_preprocessing_costs.axis('off')
        self.ax_preprocessing_costs.set_title("Preprocessing Costs", fontsize=11)
        # self.ax_preprocessing_costs.text(0.05, 0.95, costs_text, va='top', fontsize=9, family="monospace")
        preprocessing_table = self.ax_preprocessing_costs.table(
            cellText=[[f'{self.costs["duration"]["system"]["preprocessing"]:.4f} s',
                       f'{self.costs["duration"]["baseline"]["preprocessing"]:.4f} s',
                       f'{((self.costs["duration"]["baseline"]["preprocessing"] - self.costs["duration"]["system"]["preprocessing"])/self.costs["duration"]["baseline"]["preprocessing"])*100:.2f}%'],
                      [f'{self.costs["tot_energy"]["system"]["preprocessing"]:.4f} kWh',
                       f'{self.costs["tot_energy"]["baseline"]["preprocessing"]:.4f} kWh',
                       f'{((self.costs["tot_energy"]["baseline"]["preprocessing"] - self.costs["tot_energy"]["system"]["preprocessing"])/self.costs["tot_energy"]["baseline"]["preprocessing"])*100:.2f}%'],
                      [f'{self.costs["cpu_energy"]["system"]["preprocessing"]:.4f} kWh',
                       f'{self.costs["cpu_energy"]["baseline"]["preprocessing"]:.4f} kWh',
                       f'{((self.costs["cpu_energy"]["baseline"]["preprocessing"] - self.costs["cpu_energy"]["system"]["preprocessing"])/self.costs["cpu_energy"]["baseline"]["preprocessing"])*100:.2f}%'],
                      [f'{self.costs["gpu_energy"]["system"]["preprocessing"]:.4f} kWh',
                       f'{self.costs["gpu_energy"]["baseline"]["preprocessing"]:.4f} kWh',
                       f'{((self.costs["gpu_energy"]["baseline"]["preprocessing"] - self.costs["gpu_energy"]["system"]["preprocessing"])/self.costs["gpu_energy"]["baseline"]["preprocessing"])*100:.2f}%'],
                      [f'{self.costs["ram_energy"]["system"]["preprocessing"]:.4f} kWh',
                       f'{self.costs["ram_energy"]["baseline"]["preprocessing"]:.4f} kWh',
                       f'{((self.costs["ram_energy"]["baseline"]["preprocessing"] - self.costs["ram_energy"]["system"]["preprocessing"])/self.costs["ram_energy"]["baseline"]["preprocessing"])*100:.2f}%']],
            colLabels=["System", "Baseline", "Savings"],
            rowLabels=["Duration", "Tot Energy", "CPU Energy", "GPU Energy", "RAM Energy"],
            bbox=[0, 0, 1, 1]
        )
        preprocessing_table.scale(1, 1.9)
        
        self.ax_inference_costs.clear()
        self.ax_inference_costs.axis('off')
        self.ax_inference_costs.set_title("Inference Costs", fontsize=11)
        # self.ax_inference_costs.text(0.05, 0.95, costs_text, va='top', fontsize=9, family="monospace")
        inference_table = self.ax_inference_costs.table(
            cellText=[[f'{self.costs["duration"]["system"]["inference"]:.4f} s',
                       f'{self.costs["duration"]["baseline"]["inference"]:.4f} s',
                       f'{((self.costs["duration"]["baseline"]["inference"] - self.costs["duration"]["system"]["inference"])/self.costs["duration"]["baseline"]["inference"])*100:.2f}%'],
                      [f'{self.costs["tot_energy"]["system"]["inference"]:.4f} kWh',
                       f'{self.costs["tot_energy"]["baseline"]["inference"]:.4f} kWh',
                       f'{((self.costs["tot_energy"]["baseline"]["inference"] - self.costs["tot_energy"]["system"]["inference"])/self.costs["tot_energy"]["baseline"]["inference"])*100:.2f}%'],
                      [f'{self.costs["cpu_energy"]["system"]["inference"]:.4f} kWh',
                       f'{self.costs["cpu_energy"]["baseline"]["inference"]:.4f} kWh',
                       f'{((self.costs["cpu_energy"]["baseline"]["inference"] - self.costs["cpu_energy"]["system"]["inference"])/self.costs["cpu_energy"]["baseline"]["inference"])*100:.2f}%'],
                      [f'{self.costs["gpu_energy"]["system"]["inference"]:.4f}  kWh',
                       f'{self.costs["gpu_energy"]["baseline"]["inference"]:.4f} kWh',
                       f'{((self.costs["gpu_energy"]["baseline"]["inference"] - self.costs["gpu_energy"]["system"]["inference"])/self.costs["gpu_energy"]["baseline"]["inference"])*100:.2f}%'],
                      [f'{self.costs["ram_energy"]["system"]["inference"]:.4f} kWh',
                       f'{self.costs["ram_energy"]["baseline"]["inference"]:.4f} kWh',
                       f'{((self.costs["ram_energy"]["baseline"]["inference"] - self.costs["ram_energy"]["system"]["inference"])/self.costs["ram_energy"]["baseline"]["inference"])*100:.2f}%']],
            colLabels=["System", "Baseline", "Savings"],
            rowLabels=["Duration", "Tot Energy", "CPU Energy", "GPU Energy", "RAM Energy"],
            bbox=[0, 0, 1, 1]
        )
        inference_table.scale(1, 1.9)
        
        self.fig.canvas.draw_idle()
        
    def close(self):
        self.db.close()
