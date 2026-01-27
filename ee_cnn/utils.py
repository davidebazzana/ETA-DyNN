import csv
import os
import json
import random
import argparse
import shutil
import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycalib.visualisations import plot_reliability_diagram
from pycalib.metrics import binary_ECE, binary_MCE
from betacal import BetaCalibration
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from scipy.stats import uniform
from datetime import datetime, date
from pathlib import Path


def fit_calibrator(labels, scores) -> BetaCalibration:
    # Fit three-parameter beta calibration
    bc = BetaCalibration(parameters="abm")
    bc.fit(scores[:, 1].reshape(-1, 1), labels)

    return bc

def save_calibrator(bc: BetaCalibration, directory:str|None = None, name:str|None = "beta_calibrator"):
    if directory is None:
        ts = str(datetime.now())
        directory = os.path.join('./calibrators/', ts)
    Path(directory).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(directory, f'{name}.pkl'), 'wb') as f:
        pickle.dump(bc, f)


def analyse_reliability(labels, scores, plot_rd=False):
    print(f'ECE: {round(binary_ECE(labels, scores[:, 1], bins=2), 8)}')
    print(f'MCE: {round(binary_MCE(labels, scores[:, 1], bins=2), 8)}')

    if plot_rd:
        fig = plot_reliability_diagram(labels=labels,
                                       scores=scores,
                                       show_histogram=True,
                                       show_bars=True,
                                       show_gaps=True)
        plt.figure(fig.number)
        plt.show()


def analyse_calibration(scores_path:str):
    files = os.listdir(scores_path)
    scores_pattern = r'^([\w]*)scores.npy$'
    labels_pattern = r'^[\w]*labels.npy$'

    labels_path = None
    scores_files = []
    for f in files:
        match = re.match(scores_pattern, f)
        if match:
            scores_files.append({
                "name": match.group(1),
                "path": os.path.join(scores_path, match.group(0))
            })
            continue
        match = re.match(labels_pattern, f)
        if match:
            labels_path = os.path.join(scores_path, match.group(0))

    if len(scores_files) == 0 or labels_path is None:
        raise Exception("The scores path should contain valid files")

    labels = np.load(labels_path)

    today = str(date.today())
    calibrators_path = os.path.join('./calibrators/', today)
    Path(calibrators_path).mkdir(parents=True, exist_ok=True)
    for scores_file in scores_files:
        print("==============================")
        print(f'Model: {scores_file["name"]}')
        scores = np.load(scores_file["path"])

        print("Uncalibrated")
        analyse_reliability(labels, scores, plot_rd=True)

        # Fit three-parameter beta calibration
        bc = BetaCalibration(parameters="abm")
        bc.fit(scores[:, 1].reshape(-1, 1), labels)

        # Save to file
        with open(os.path.join(calibrators_path, f'{scores_file["name"]}_beta_calibrator.pkl'), 'wb') as f:
            pickle.dump(bc, f)
        """
        # Load calibrator
        with open(calibrators_path + f'{model}_beta_calibrator.pkl', 'rb') as f:
            bc = pickle.load(f)
        """
        
        calibrated_scores = bc.predict(scores[:, 1])
        calibrated_scores = np.stack([1-calibrated_scores, calibrated_scores], axis=1)

        print("Calibrated")
        analyse_reliability(labels, calibrated_scores, plot_rd=True)
    print("==============================")


class ConfidenceEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, *, lower_ret_threshold=0.2, upper_ret_threshold=0.8):
        self.lower_ret_threshold = lower_ret_threshold
        self.upper_ret_threshold = upper_ret_threshold

    def fit(self, X, y=None):
        self.is_fitted_ = True
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X[X < self.lower_ret_threshold] = 0.
        X[X > self.upper_ret_threshold] = 1.
        X[(X >= self.lower_ret_threshold) & (X <= self.upper_ret_threshold)] = -1.
        # print(f'X: {X}')
        return X
        # return np.full(shape=X.shape[0], fill_value=self.param)
    

def confidence_scorer(y_true, y_pred):
    gamma = 0.8
    beta = 0.2
    
    mask_returned_preds = y_pred != -1
    num_returned = np.sum(mask_returned_preds)
    fraction_returned = (num_returned / y_pred.size)
    y_true_ret = y_true[mask_returned_preds]
    y_pred_ret = y_pred[mask_returned_preds]
    accuracy = accuracy_score(y_true_ret, y_pred_ret)
    precision = precision_score(y_true_ret, y_pred_ret)
    recall = recall_score(y_true_ret, y_pred_ret)
    f1 = f1_score(y_true_ret, y_pred_ret)
    score =  gamma*(beta*f1 + (1 - beta)*accuracy) + (1 - gamma)*fraction_returned
    return score

def print_fitting_thresholds_results(y_pred, y_test, lower_threshold, upper_threshold, model_name):
    mask_returned_preds = y_pred != -1
    num_returned = np.sum(mask_returned_preds)
    num_pred_free_returned = np.sum(y_pred == 0)
    num_pred_infested_returned = np.sum(y_pred == 1)
    percentage_returned = (num_returned / y_pred.size) * 100
    y_true_ret = y_test[mask_returned_preds]
    y_pred_ret = y_pred[mask_returned_preds]
    accuracy = accuracy_score(y_true_ret, y_pred_ret)
    precision = precision_score(y_true_ret, y_pred_ret)
    recall = recall_score(y_true_ret, y_pred_ret)
    f1 = f1_score(y_true_ret, y_pred_ret)
    cm = confusion_matrix(y_true_ret, y_pred_ret)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    res = f'========== THRESHOLDS SEARCH RESULTS {timestamp} ==========\n\n'
    res += "==================================================\n"
    res += f'model: {model_name}\n'
    res += '------------------------------\n'
    res += f'accuracy: {accuracy:.4f}\n'
    res += f'precision: {precision:.4f}\n'
    res += f'recall: {recall:.4f}\n'
    res += f'f1 score: {f1:.4f}\n'
    res += f'confusion matrix:\n{cm}\n'
    res += '------------------------------\n'
    res += f'percentage returned: {percentage_returned:.2f}%\n'
    res += f'number of samples returned: {num_returned}\n'
    res += f'best lower return threshold: {lower_threshold:.4f}\n'
    res += f'best upper return threshold: {upper_threshold:.4f}\n'
    res += "==================================================\n"
    print(res)
    print(f"------------------------------")


def fit_thresholds(X_train, y_train, X_test, y_test, bc: BetaCalibration, verbose:bool=False, model_name:str="N/A"):
    X_train = bc.predict(X_train[:, 1])

    param_dist = {
        'lower_ret_threshold': uniform(0.0, 0.5),
        'upper_ret_threshold': uniform(0.5, 0.5)
    }

    scorer = make_scorer(confidence_scorer)
    
    search = RandomizedSearchCV(
        estimator=ConfidenceEstimator(),
        param_distributions=param_dist,
        n_iter=60,
        scoring=scorer,
        random_state=42
    )
    
    search.fit(X_train, y_train)

    best_lower_threshold = search.best_params_['lower_ret_threshold']
    best_upper_threshold = search.best_params_['upper_ret_threshold']

    if verbose:
        print("Best lower return threshold:", best_lower_threshold)
        print("Best upper return threshold:", best_upper_threshold)

    if X_test is not None and y_test is not None:
        X_test = bc.predict(X_test[:, 1])
        y_pred = search.predict(X_test)

        score = confidence_scorer(y_test, y_pred)
        print("Best score:", score)

        if verbose: print_fitting_thresholds_results(y_pred, y_test, best_lower_threshold, best_upper_threshold, model_name)
    
        return score, best_lower_threshold, best_upper_threshold
    else:
        return best_lower_threshold, best_upper_threshold

def search_thresholds():
    models = [
        'Joint_EE_MobileNetV3_0',
        'Joint_EE_MobileNetV3_1',
        'Joint_EE_MobileNetV3_2',
    ]
    scores_path = './scores/test/'
    val_scores_path = './scores/val/'
    calibrators_path = './calibrators/'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    res = f'========== THRESHOLDS SEARCH RESULTS {timestamp} ==========\n\n'
    for model in models:
        print("==============================")
        print(f'Model: {model}')
        X_train = np.load(scores_path + model + "_scores.npy")
        y_train = np.load(scores_path + model + "_labels.npy")
        X_test = np.load(val_scores_path + model + "_scores.npy")
        y_test = np.load(val_scores_path + model + "_labels.npy")

        # Load calibrator (calibrated on the entire test set)
        with open(calibrators_path + f'{model}_beta_calibrator.pkl', 'rb') as f:
            bc = pickle.load(f)

        # Calibrated scores
        X_train = bc.predict(X_train[:, 1])
        # calibrated_scores = bc.predict(X_train[:, 1])
        X_test = bc.predict(X_test[:, 1])

        # X_train, X_test, y_train, y_test = train_test_split(calibrated_scores, labels, random_state=42, test_size=0.5)

        # Parameter distribution: tuning the constant class
        param_dist = {
            'lower_ret_threshold': uniform(0.0, 0.5),
            'upper_ret_threshold': uniform(0.5, 0.5)
        }

        scorer = make_scorer(confidence_scorer)
        
        # Random search
        search = RandomizedSearchCV(
            estimator=ConfidenceEstimator(),
            param_distributions=param_dist,
            n_iter=60,
            scoring=scorer,
            random_state=42
        )

        print(f"------------------------------")
        print(f'searching...')
        search.fit(X_train, y_train)
        print(f"------------------------------")

        # Evaluate
        print(f"------------------------------")
        print(f'evaluating...')
        y_pred = search.predict(X_test)
        print("Best lower return threshold:", search.best_params_['lower_ret_threshold'])
        print("Best upper return threshold:", search.best_params_['upper_ret_threshold'])
        # print("Accuracy:", accuracy_score(y_test, y_pred))
        print(f"------------------------------")

        print(f"------------------------------")
        print('results:')
        # confidence_scorer(y_test, y_pred)
        mask_returned_preds = y_pred != -1
        num_returned = np.sum(mask_returned_preds)
        num_pred_free_returned = np.sum(y_pred == 0)
        num_pred_infested_returned = np.sum(y_pred == 1)
        percentage_returned = (num_returned / y_pred.size) * 100
        y_true_ret = y_test[mask_returned_preds]
        y_pred_ret = y_pred[mask_returned_preds]
        accuracy = accuracy_score(y_true_ret, y_pred_ret)
        precision = precision_score(y_true_ret, y_pred_ret)
        recall = recall_score(y_true_ret, y_pred_ret)
        f1 = f1_score(y_true_ret, y_pred_ret)
        cm = confusion_matrix(y_true_ret, y_pred_ret)
        res += "==================================================\n"
        res += f'model: {model}\n'
        res += '------------------------------\n'
        res += f'accuracy: {accuracy:.4f}\n'
        res += f'precision: {precision:.4f}\n'
        res += f'recall: {recall:.4f}\n'
        res += f'f1 score: {f1:.4f}\n'
        res += f'confusion matrix:\n{cm}\n'
        res += '------------------------------\n'
        res += f'percentage returned: {percentage_returned:.2f}%\n'
        res += f'number of samples returned: {num_returned}\n'
        """
        res += f'number of predicted free samples returned: {num_pred_free_returned}\n'
        res += f'number of predicted infested samples returned: {num_pred_infested_returned}\n'
        """
        res += f'best lower return threshold: {search.best_params_["lower_ret_threshold"]:.4f}\n'
        res += f'best upper return threshold: {search.best_params_["upper_ret_threshold"]:.4f}\n'
        res += "==================================================\n"
        print(res)
        print(f"------------------------------")
    with open("./logs/thresholds_joint_0.txt", "w") as file:
        file.write(res)
        
def compute_cv_stats():
    with open("cv_folds_exits_metrics.json", "r") as f:
        exits_metrics = json.load(f)
    print("="*30 + f"\n5-FOLD CROSS-VALIDATION\n" + "="*30)
    for i in range(3):
        print("#"*30 + f"\nEXIT {i}\n" + "#"*30)
        for metric_name, metric_value in exits_metrics[str(i)].items():
            mean = np.mean(metric_value)
            std = np.std(metric_value)
            print(f"{metric_name.capitalize()}: mean = {mean:.4f}, std = {std:.4f}")
    with open("cv_final_exits_metrics.json", "r") as f:
        final_metrics = json.load(f)
    print("="*30 + f"\nTEST\n" + "="*30)
    for i in range(3):
        print("#"*30 + f"\nEXIT {i}\n" + "#"*30)
        for metric_name, metric_value in final_metrics[str(i)].items():
            print(f"{metric_name.capitalize()}: {metric_value:.4f}")

def compute_metrics(labels, predictions, title):
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)

    print(f"==================== {title} ====================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)

            
def analyse_comparison():
    with open("comparison.csv") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        data = [row for row in reader]

    stats = {}
    for idx, stat in enumerate(data[0]):
        stats[stat] = [row[idx] for row in data[1:]]
        if stat != "video":
            stats[stat] = np.array(stats[stat], dtype=np.float32)
    
    print("Percentage not returned (because of no frames extracted): {:.2f}%, {} of which free, {} of which infested.".format((stats["batch_res"] == -1).sum()/len(stats["batch_res"]) * 100,
                                                                                                                              (stats["label"][stats["batch_res"] == -1] == 0).sum(),
                                                                                                                              (stats["label"][stats["batch_res"] == -1] == 1).sum()))
    returned_mask = stats["batch_res"] != -1
    compute_metrics(stats["label"][returned_mask], stats["batch_res"][returned_mask], title="ee_cnn")
    compute_metrics(stats["label"], stats["vit4v_pred"], title="vit4v")

    print("Exit 0: {}, Exit 1: {}, Exit 2: {}".format((stats["exit_idx"] == 0).sum(),
                                                      (stats["exit_idx"] == 1).sum(),
                                                      (stats["exit_idx"] == 2).sum()))

    failed_mask = stats["label"] != stats["batch_res"]
    failed_by_vit4v_mask = stats["label"] != stats["vit4v_pred"]
    correct_failed_vit4v = stats["label"][failed_mask] == stats["vit4v_pred"][failed_mask]
    correct_by_ee_cnn_failed_by_vit4v = stats["label"][failed_by_vit4v_mask] == stats["batch_res"][failed_by_vit4v_mask]
    print("Percentage vit4v correct where ee_cnn fails: {:.2f}%".format(correct_failed_vit4v.sum()/len(correct_failed_vit4v) * 100))
    print("Percentage ee_cnn correct where vit4v fails: {:.2f}%".format(correct_by_ee_cnn_failed_by_vit4v.sum()/len(correct_by_ee_cnn_failed_by_vit4v) * 100))

def map_video_to_images_indexes(videos_dataset, images_dataset, video_indexes, dataset_base_path):
    
    images = []
    images_indexes = []
    dataset_base_path = Path(dataset_base_path)

    map_label = {
        0: "free",
        1: "infested"
    }
    for index in video_indexes:
        label = videos_dataset[index]["label"]
        bee_id = videos_dataset[index]["id"]
        pattern = f'^{label}_{bee_id}_[0-9]*.png$'
        for image in images_dataset:
            res = re.match(pattern, image)
            if res:
                images.append((label, dataset_base_path/map_label[label]/res.group(0)))

    return images

def get_images(folder:Path):
    images = [f for f in sorted(os.listdir(folder))
              if (os.path.isfile(os.path.join(folder, f)) and
                  f.endswith('.png'))]
    return images
