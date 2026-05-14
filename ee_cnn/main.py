import sys
import os
import re
import argparse
import json
import pickle
import csv
import logging
import sqlite3
import random
from pathlib import Path
from datetime import datetime
import cv2 as cv
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2
from torchvision import datasets, transforms
from torchmetrics.classification import MulticlassAccuracy
from ee_mobilenetv3.joint_ee_mobilenetv3 import Joint_EE_MobileNetV3
from ee_mobilenetv3.utils import joint_train, joint_test
from utils import analyse_reliability, fit_calibrator, save_calibrator, fit_thresholds, map_video_to_images_indexes, get_images
from key_frame_extraction.key_frame_extraction.key_frame_extractor import KeyFrameExtractor
from codecarbon import EmissionsTracker
import cProfile
import pstats
from pstats import SortKey
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from data.image_dataset import ImageDataset
from experiments_db import EXPERIMENTS_DB
from remote import upload

transform_test = torchvision.transforms.v2.Compose(
    [
        torchvision.transforms.v2.ToTensor(),
        torchvision.transforms.v2.Resize((256,256)),
        torchvision.transforms.v2.ToDtype(torch.float32),
        torchvision.transforms.v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ]
)

transform_train = torchvision.transforms.v2.Compose(
    [
        torchvision.transforms.v2.ToTensor(),
        torchvision.transforms.v2.Resize((256,256)),
        torchvision.transforms.Lambda(lambda img: F.rotate(img, float(random.randint(-180, 180)))),
        torchvision.transforms.Lambda(lambda img: img if random.random()>0.5 else F.hflip(img)),
        torchvision.transforms.Lambda(lambda img: img if random.random()>0.5 else F.vflip(img)),
        torchvision.transforms.ColorJitter(
            brightness=float(random.randint(30, 80)) / 100,
            saturation=float(random.randint(30, 80)) / 100,
            hue=float(random.randint(0, 30)) / 100,
            contrast=float(random.randint(30, 80)) / 100,
        ),
        torchvision.transforms.v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Early-Exiting CNN',
                                     description='Implementation of an Early-Exiting architecture',
                                     epilog='')
    parser.add_argument('--joint-model-path', type=str, help='Path to joint model trained parameters')
    parser.add_argument('--epochs', type=int, help='Training number of epochs')
    parser.add_argument('--extract-scores', action=argparse.BooleanOptionalAction, default=False, help='Extract scores when testing')
    parser.add_argument('--joint-cross-validation-auto', action=argparse.BooleanOptionalAction, help='Joint cross-validation')
    parser.add_argument('--extract-preds', action=argparse.BooleanOptionalAction, help='Calibrate the classifier and search for the best thresholds')
    parser.add_argument('--models-directory', type=str, help='Path to directory containing models')
    parser.add_argument('--thresholds-data', type=str, help='Path to thresholds data')
    parser.add_argument('--inference', action=argparse.BooleanOptionalAction, help='Inference')
    parser.add_argument('--experiment', action=argparse.BooleanOptionalAction, help='Experiment')
    parser.add_argument('--video-file', type=str, help='Path to video to classify')
    parser.add_argument('--codename', type=str, help='Name to assign to the running task. All files/folders produced during the task, will be assigned a derivative name.')
    parser.add_argument('--dataset', type=str, help='Path to the dataset to be used.')
    parser.add_argument('--dataset-year', type=str, help='2024 or 2025 dataset?')
    parser.add_argument('--model-hist', type=str, help='Path to the model histogram to be used by the key frame selection algorithm.')
    args = parser.parse_args()

    logging.getLogger("codecarbon").disabled = True
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.codename is not None:
        codename = args.codename
    else:
        raise RuntimeError("Provide --codename")
    
    if args.joint_cross_validation_auto:
        if args.dataset is None:
            raise ValueError("Provide the path to the dataset with --dataset")

        print('Launching cross-validation...')

        num_epochs = args.epochs if args.epochs is not None else 5
        exit_points = (8,)

        models_directory = os.path.join("./models/", codename)
        Path(models_directory).mkdir(parents=True, exist_ok=True)
        calibrators_directory = os.path.join("./calibrators/", codename)
        Path(calibrators_directory).mkdir(parents=True, exist_ok=True)
        thresholds_file = f"./thresholds/{codename}_thresholds.json"
        
        folds_rpq = {}

        with open("./dataset_metadata.json", 'r') as f:
            dataset_data = json.load(f)

        dataset = [data for data in dataset_data["train"]["free"]] + [data for data in dataset_data["train"]["infested"]]

        free_dir = Path(args.dataset)/"train"/"free"
        free_images = get_images(free_dir)
        infested_dir = Path(args.dataset)/"train"/"infested"
        infested_images = get_images(infested_dir)
        
        extracted_images_dataset = free_images + infested_images
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        calibration_scores = [
            {
                "exit_idx": 0,
                "scores": None
            },
            {
                "exit_idx": 1,
                "scores": None
            }            
        ]
        calibration_labels = None
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print("#"*30 + f"\nFOLD {fold}\n" + "#"*30)
            print(f"Fold {fold}")
            print("Train size:", len(train_idx), "Val size:", len(val_idx))
            
            model = Joint_EE_MobileNetV3(exit_points=exit_points, thresholds=(0.9, 0.9))
            model.classifier[3] = nn.Linear(in_features=1280, out_features=1)
        
            if args.joint_model_path is not None:
                print(f'Loading model parameters from {args.joint_model_path}')
                model.load_state_dict(torch.load(args.joint_model_path,
                                                 map_location=torch.device('cpu'),
                                                 weights_only=True))
            model.to(device)

            train_dataset_data = map_video_to_images_indexes(dataset,
                                                             extracted_images_dataset,
                                                             train_idx,
                                                             dataset_base_path=Path(args.dataset)/"train")
            val_dataset_data = map_video_to_images_indexes(dataset,
                                                           extracted_images_dataset,
                                                           val_idx,
                                                           dataset_base_path=Path(args.dataset)/"train")
            train_dataset = ImageDataset(train_dataset_data,
                                         transform=transform_train)
            val_dataset = ImageDataset(val_dataset_data,
                                       transform=transform_test)

            labels = np.array([label for label, _ in train_dataset.data])

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
            
            rpq_res, scores, labels = joint_train(model=model,
                                                  train_data_loader=train_loader,
                                                  test_data_loader=val_loader,
                                                  num_epochs=num_epochs,
                                                  device=device,
                                                  save_model=True,
                                                  extract_scores=True,
                                                  model_path=os.path.join(models_directory, f"fold_{fold}.pt"),
                                                  codename=codename + f"_fold_{fold}")

            assert scores[0]["scores"].shape[0] == len(labels), f"Length of scores does not match length of labels for exit {scores[0]['exit_idx']}"
            assert scores[1]["scores"].shape[0] == len(labels), f"Length of scores does not match length of labels for exit {scores[1]['exit_idx']}"
            if calibration_scores[0]["scores"] is None and calibration_scores[1]["scores"] is None and calibration_labels is None:
                calibration_scores[0]["scores"] = np.copy(scores[0]["scores"])
                calibration_scores[1]["scores"] = np.copy(scores[1]["scores"])
                calibration_labels = np.copy(labels)
            else:
                calibration_scores[0]["scores"] = np.concatenate((calibration_scores[0]["scores"], scores[0]["scores"]), axis=0)
                calibration_scores[1]["scores"] = np.concatenate((calibration_scores[1]["scores"], scores[1]["scores"]), axis=0)
                calibration_labels = np.concatenate((calibration_labels, labels))
            np.set_printoptions(threshold=sys.maxsize)
            assert calibration_scores[0]["scores"].shape[0] == len(calibration_labels), f"Length of scores does not match length of labels for exit {calibration_scores[0]['exit_idx']}"
            assert calibration_scores[1]["scores"].shape[0] == len(calibration_labels), f"Length of scores does not match length of labels for exit {calibration_scores[1]['exit_idx']}"

            folds_rpq[fold] = rpq_res

        # Calibrate and search for best thresholds
        best_thresholds = []
        for exit_idx in range(len(calibration_scores)):
            scores = calibration_scores[exit_idx]["scores"]
            analyse_reliability(calibration_labels, scores, plot_rd=False)
            bc = fit_calibrator(calibration_labels, scores)
            save_calibrator(bc, directory=calibrators_directory, name=f'exit_{exit_idx}')

            lower_threshold, upper_threshold = fit_thresholds(X_train=scores,
                                                              y_train=calibration_labels,
                                                              X_test=None,
                                                              y_test=None,
                                                              bc=bc,
                                                              verbose=True,
                                                              model_name=f"fold_{fold}_exit_{exit_idx}")
            best_thresholds.append({
                "exit": exit_idx,
                "lower_threshold": lower_threshold,
                "upper_threshold": upper_threshold
            })

        print("best_thresholds:", best_thresholds)

        with open(thresholds_file, "w") as f:
            json.dump(best_thresholds, f)

        exits_metrics = {}
        
        for i in range(len(exit_points)):
            for fold in folds_rpq.values():
                for exit_idx, metrics in fold.items():
                    if i == exit_idx:
                        for metric_name, metric_value in metrics.items():
                            if exits_metrics.get(exit_idx) is None:
                                exits_metrics[exit_idx] = {
                                    "accuracy": [],
                                    "precision": [],
                                    "recall": [],
                                    "f1_score": []
                                }
                            exits_metrics[exit_idx][metric_name].append(metric_value)

        print(f"exits_metrics: {exits_metrics}")          
        # Compute mean and std
        for i in range(len(exit_points)):
            print("#"*30 + f"\nEXIT {i}\n" + "#"*30)
            for metric_name, metric_value in exits_metrics[i].items():
                mean = np.mean(metric_value)
                std = np.std(metric_value)
                print(f"{metric_name.capitalize()}: mean = {mean:.4f}, std = {std:.4f}")
        with open("cv_auto_folds_exits_metrics_auto.json", "w") as f:
            json.dump(exits_metrics, f)

        model = Joint_EE_MobileNetV3(exit_points=exit_points, thresholds=(0.9, 0.9))
        model.classifier[3] = nn.Linear(in_features=1280, out_features=1)
        
        if args.joint_model_path is not None:
            print(f'Loading model parameters from {args.joint_model_path}')
            model.load_state_dict(torch.load(args.joint_model_path,
                                             map_location=torch.device('cpu'),
                                             weights_only=True))
        model.to(device)
        
        train_dataset = datasets.ImageFolder(root=Path(args.dataset)/"train",
                                             transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=64,
                                                   shuffle=True)
    
        test_dataset = datasets.ImageFolder(root=Path(args.dataset)/"val",
                                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=64,
                                                  shuffle=True)

        rpq_res = joint_train(model=model,
                              train_data_loader=train_loader,
                              test_data_loader=test_loader,
                              num_epochs=num_epochs,
                              device=device,
                              save_model=True,
                              model_path=os.path.join(models_directory, f"final.pt"),
                              codename=codename + "_final")

        with open("cv_auto_final_exits_metrics.json", "w") as f:
            json.dump(rpq_res, f)

    if args.extract_preds:
        if args.dataset is None:
            raise ValueError("Provide the path to the dataset with --dataset")

        # Extract scores
        models_files = [f for f in sorted(os.listdir(args.models_directory))
                        if (os.path.isfile(os.path.join(args.models_directory, f)) and
                            f.endswith('.pt'))]
        fold_pattern = r'^fold_([0-9]+).pt$'
        final_pattern = r'^final.pt$'
        best_thresholds = []
        with open("./mixed_dataset.json", 'r') as f:
            dataset_data = json.load(f)

        # dataset = [data for data in dataset_data["train"]["free"]] + [data for data in dataset_data["train"]["infested"]]
        dataset = [data for data in dataset_data["train"]["free"] if data["year"] == 2024] + [data for data in dataset_data["train"]["infested"] if data["year"] == 2024]

        free_dir = Path(args.dataset)/"train"/"free"
        free_images = get_images(free_dir, args.dataset_year)
        infested_dir = Path(args.dataset)/"train"/"infested"
        infested_images = get_images(infested_dir, args.dataset_year)
        
        extracted_images_dataset = free_images + infested_images
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        train_scores = [
            {"exit_idx": 0,
             "scores": None},
            {"exit_idx": 1,
             "scores": None},
        ]

        train_labels = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print("#"*30 + f"\nFOLD {fold}\n" + "#"*30)
            
            for model_file in models_files:
                match = re.match(fold_pattern, model_file)
                if match and int(match.group(1)) == fold:
                    model = Joint_EE_MobileNetV3(exit_points=(8,), thresholds=(0.9, 0.9))
                    model.classifier[3] = nn.Linear(in_features=1280, out_features=1)
                    model_path = str(Path(args.models_directory)/model_file)
                    print(f'Loading model parameters from {model_path}')
                    model.load_state_dict(torch.load(model_path,
                                                     map_location=torch.device('cpu'),
                                                     weights_only=True))
                    model.to(device)

                    train_dataset_data = map_video_to_images_indexes(dataset,
                                                                     extracted_images_dataset,
                                                                     train_idx,
                                                                     dataset_base_path=Path(args.dataset)/"train")
                    val_dataset_data = map_video_to_images_indexes(dataset,
                                                                   extracted_images_dataset,
                                                                   val_idx,
                                                                   dataset_base_path=Path(args.dataset)/"train")
                    val_dataset = ImageDataset(val_dataset_data,
                                               transform=transform_test)

                    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

                    _, _, exits_scores, labels = joint_test(model=model,
                                                            data_loader=val_loader,
                                                            device=device,
                                                            extract_scores=True,
                                                            scores_path="./scores/")
                    if train_scores[0]["scores"] is None:
                        train_scores[0]["scores"] = exits_scores[0]["scores"]
                    else:
                        train_scores[0]["scores"] = np.concatenate((train_scores[0]["scores"], exits_scores[0]["scores"]))
                    if train_scores[1]["scores"] is None:
                        train_scores[1]["scores"] = exits_scores[1]["scores"]
                    else:
                        train_scores[1]["scores"] = np.concatenate((train_scores[1]["scores"], exits_scores[1]["scores"]))

                    if len(train_labels) == 0:
                        train_labels = labels
                    else:
                        train_labels = np.concatenate((train_labels, labels))

        test_dataset = datasets.ImageFolder(root=Path(args.dataset)/"val",
                                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=64,
                                                  shuffle=True)

        for model_file in models_files:
            match = re.match(final_pattern, model_file)
            if match:
                model = Joint_EE_MobileNetV3(exit_points=(8,), thresholds=(0.9, 0.9))
                model.classifier[3] = nn.Linear(in_features=1280, out_features=1)
                model_path = str(Path(args.models_directory)/model_file)
                print(f'Loading model parameters from {model_path}')
                model.load_state_dict(torch.load(model_path,
                                                 map_location=torch.device('cpu'),
                                                 weights_only=True))
                model.to(device)

                _, _, test_scores, test_labels = joint_test(model=model,
                                                            data_loader=test_loader,
                                                            device=device,
                                                            extract_scores=True,
                                                            scores_path="./scores/")

        data = {
            "train": {
                "scores": train_scores,
                "labels": train_labels
            },
            "test": {
                "scores": test_scores,
                "labels": test_labels
            }
        }
        os.makedirs(f"./scores/{args.codename}/", exist_ok=True)
        with open(Path(f"./scores/{args.codename}/")/"scores.pkl", "wb") as f:
            pickle.dump(data, f)

    if args.experiment:
        if args.model_hist is None:
            model_hist = "../key_frame_extraction/assets/model_histograms/model_hist.npz"
        else:
            model_hist = args.model_hist

        # Set up experiment database
        db = EXPERIMENTS_DB(path=f"{codename}.db")
        db.reset_database()
        db.load_vit_validation(dataset_codename="vit_validation",
                               video_directory="/usr/src/ETA-DyNN/dataset/")

        db.add_experiment(codename)
        db.add_dataset_to_experiment(codename, "vit_validation")

        # Load the model parameters
        print(f"Experiment {codename}")
        print("[ee_cnn] Loading the model...")
        models_directory = f"./models/{codename}/"
        models_files = [f for f in sorted(os.listdir(models_directory))
                        if (os.path.isfile(os.path.join(models_directory, f)) and
                            f.endswith('.pt'))]
        final_pattern = r'^final.pt$'
        for model_file in models_files:
            match = re.match(final_pattern, model_file)
            if match:
                model_path = os.path.join(models_directory, match.group(0))

                model = Joint_EE_MobileNetV3(exit_points=(8,), thresholds=(0.9,0.9))
                model.classifier[3] = nn.Linear(in_features=1280, out_features=1)
                model.load_state_dict(torch.load(model_path,
                                                 map_location=torch.device('cpu'),
                                                 weights_only=True))
                model.to(device)

        # Iterate over the dataset of videos
        for label, video_file in tqdm(db.get_dataset_samples("vit_validation")):
            print(f"{video_file=}")
            success = True

            with EmissionsTracker() as kfe_tracker:
                kfe = KeyFrameExtractor(model_data=model_hist)
                file_name = os.path.basename(video_file)
                pattern = r'^([0-9]*) [0-9- ]*.mkv$'
                res = re.match(pattern, file_name)
                if res:
                    video_id = res.group(1)
                else:
                    raise RuntimeError(f"Video {video_file}: does not respect pattern")
                try:
                    cropped = kfe.extract_frames(video_file,
                                                 squared=True,
                                                 num_key_frames=5)
                except Exception as e:
                    print("[ee_cnn] Key frame extraction failed", e)
                    success = False

            if success and len(cropped) > 0:
                frames = []
                for frame_idx, img in enumerate(cropped):
                    frame = transform_test(img)
                    frames.append(frame)
                frames = torch.stack(frames)

                with EmissionsTracker() as ee_cnn_exit_0_tracker:
                    model.eval()
                    with torch.no_grad():
                        frames = frames.to(device)
                        out = model(frames, force_exit=8)
                        exit_0_props = np.array([nn.functional.sigmoid(exit_predictions).cpu().detach().numpy() for exit_predictions in out]).squeeze()
                with EmissionsTracker() as ee_cnn_exit_1_tracker:
                    model.eval()
                    with torch.no_grad():
                        frames = frames.to(device)
                        out = model(frames)
                        exit_1_props = np.array([nn.functional.sigmoid(exit_predictions).cpu().detach().numpy() for exit_predictions in out]).squeeze()
            else:
                success = False

            if success:
                db.add_modelrun_with_performance(
                    video_file,
                    codename,
                    "ee_cnn",
                    0,
                    scores=exit_0_props.tolist(),
                    preprocessing_perf=(
                        kfe_tracker.final_emissions_data.duration,
                        kfe_tracker.final_emissions_data.energy_consumed,
                        kfe_tracker.final_emissions_data.cpu_energy,
                        kfe_tracker.final_emissions_data.gpu_energy,
                        kfe_tracker.final_emissions_data.ram_energy,
                    ),
                    inference_perf=(
                        ee_cnn_exit_0_tracker.final_emissions_data.duration,
                        ee_cnn_exit_0_tracker.final_emissions_data.energy_consumed,
                        ee_cnn_exit_0_tracker.final_emissions_data.cpu_energy,
                        ee_cnn_exit_0_tracker.final_emissions_data.gpu_energy,
                        ee_cnn_exit_0_tracker.final_emissions_data.ram_energy,
                    )
                )
                db.add_modelrun_with_performance(
                    video_file,
                    codename,
                    "ee_cnn",
                    1,
                    scores=exit_1_props[-1].tolist(),
                    preprocessing_perf=(
                        kfe_tracker.final_emissions_data.duration,
                        kfe_tracker.final_emissions_data.energy_consumed,
                        kfe_tracker.final_emissions_data.cpu_energy,
                        kfe_tracker.final_emissions_data.gpu_energy,
                        kfe_tracker.final_emissions_data.ram_energy,
                    ),
                    inference_perf=(
                        ee_cnn_exit_1_tracker.final_emissions_data.duration,
                        ee_cnn_exit_1_tracker.final_emissions_data.energy_consumed,
                        ee_cnn_exit_1_tracker.final_emissions_data.cpu_energy,
                        ee_cnn_exit_1_tracker.final_emissions_data.gpu_energy,
                        ee_cnn_exit_1_tracker.final_emissions_data.ram_energy,
                    )
                )
            else:
                db.add_modelrun_with_performance(
                    video_file,
                    codename,
                    "ee_cnn",
                    0,
                    scores=-1,
                    preprocessing_perf=(
                        kfe_tracker.final_emissions_data.duration,
                        kfe_tracker.final_emissions_data.energy_consumed,
                        kfe_tracker.final_emissions_data.cpu_energy,
                        kfe_tracker.final_emissions_data.gpu_energy,
                        kfe_tracker.final_emissions_data.ram_energy,
                    ),
                    inference_perf=(
                        ee_cnn_exit_0_tracker.final_emissions_data.duration,
                        ee_cnn_exit_0_tracker.final_emissions_data.energy_consumed,
                        ee_cnn_exit_0_tracker.final_emissions_data.cpu_energy,
                        ee_cnn_exit_0_tracker.final_emissions_data.gpu_energy,
                        ee_cnn_exit_0_tracker.final_emissions_data.ram_energy,
                    )
                )
                db.add_modelrun_with_performance(
                    video_file,
                    codename,
                    "ee_cnn",
                    1,
                    scores=-1,
                    preprocessing_perf=(
                        kfe_tracker.final_emissions_data.duration,
                        kfe_tracker.final_emissions_data.energy_consumed,
                        kfe_tracker.final_emissions_data.cpu_energy,
                        kfe_tracker.final_emissions_data.gpu_energy,
                        kfe_tracker.final_emissions_data.ram_energy,
                    ),
                    inference_perf=(
                        ee_cnn_exit_1_tracker.final_emissions_data.duration,
                        ee_cnn_exit_1_tracker.final_emissions_data.energy_consumed,
                        ee_cnn_exit_1_tracker.final_emissions_data.cpu_energy,
                        ee_cnn_exit_1_tracker.final_emissions_data.gpu_energy,
                        ee_cnn_exit_1_tracker.final_emissions_data.ram_energy,
                    )
                )
            
        db.close()
