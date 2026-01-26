from key_frame_extraction.key_frame_extractor import KeyFrameExtractor
from key_frame_extraction.utils import resize_to_fit, compute_iou
import argparse
import cv2 as cv
import numpy as np
import os
import re
import ast
import pickle
import math
import time
from datetime import datetime
import csv
import json
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
from .utils import find_file_by_integer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Key Frame Extraction by Motion Analysis",
                                     description="An implementation of 'Key Frame Selection by Motion Analysis' by Wayne Wolf",
                                     epilog="")
    parser.add_argument("--video", type=str, help="path to the video to extract the key frames from")
    parser.add_argument("--model-image", type=str, help="path to the image to use to extract the model histogram from")
    parser.add_argument("--model-histogram", type=str, default="./assets/model_histograms/model_hist.npz", help="path to the model histogram data")
    parser.add_argument("--video-dataset", type=str, help="path to the video dataset. It expects the folder structure: /varroa_free/*.mkv, /varroa_infested/*.mkv")
    parser.add_argument("--extracted-dataset", type=str, help="path to be used where to construct the dataset with extracted frames")
    parser.add_argument("--codename", type=str, default=str(datetime.now()), help="codename of the run")
    parser.add_argument("--compute-model-histogram", action=argparse.BooleanOptionalAction, help="Compute the model histogram to use in back-projection. You must provide the model image. If you provide a path through --model-histogram, the model histogram will be dumped in an .npy file with that path and file name.")
    parser.add_argument("--extract-frames", action=argparse.BooleanOptionalAction, help="Compute Lucas-Kanade Optical Flow on a video. The path of the video must be passed through the argument --video.")
    parser.add_argument("--extract-dataset", action=argparse.BooleanOptionalAction, help="Extract frames of an entire dataset. The path of the dataset must be passed through the argument --video-dataset.")
    args = parser.parse_args()

    kfe = KeyFrameExtractor()

    model_histogram_path = Path(args.model_histogram)
    codename = args.codename

    if args.compute_model_histogram:
        try:
            kfe.compute_model_histogram(model_image=args.model_image)
        except TypeError:
            TypeError("Provide the image to use as model image through --model-image")
        kfe.dump_model_histogram_data(model_histogram_path)

    if (args.extract_frames or
        args.extract_dataset or
        args.construct_motion_metric_dataset):
        kfe.load_model_histogram_data(model_histogram_path)
    
    if args.extract_frames:
        if args.video is None:
            raise TypeError("Provide the video through --video")
        
        if args.model_histogram is not None:
            kfe.load_model_histogram_data(args.model_histogram)

        cropped_frames = kfe.extract_frames(args.video,
                                            squared=True,
                                            num_key_frames=5,
                                            show_optical_flow=True,
                                            show_bounding_box=True,
                                            show_metric=True)
        print(cropped_frames)
        for frame in cropped_frames:
            cv.imshow("CROPPED FRAME", frame)
            k = cv.waitKey() & 0xff
            if k == 27:
                break

    if args.extract_dataset:
        if args.video_dataset is None:
            raise ValueError("Provide the path to the video dataset through --video-dataset")
        if args.extracted_dataset is None:
            raise ValueError("Provide the path where to save the extracted frames through --extracted-dataset")

        with open('dataset_metadata.json', 'r') as f:
            dataset = json.load(f)

        path_free = f"{args.video_dataset}/varroa_free"
        path_infested = f"{args.video_dataset}/varroa_infested"

        for partition in dataset:
            print("Partition:", partition)
            for infestation_class in dataset[partition]:
                print("Class:", infestation_class)
                for video in tqdm(dataset[partition][infestation_class]):
                    video_file = find_file_by_integer(folder, video["id"])
                    try:
                        cropped_frames = kfe.extract_frames(video_file,
                                                            squared=True,
                                                            num_key_frames=5)
                    except Exception as e:
                        print(f"Frames extraction failed on video {video['file']}.\nError: {e}")
                        continue

                    for frame_idx, frame in enumerate(cropped_frames):
                        if frame is None:
                            continue
                        if video["label"] == 0:
                            label = "free"
                        elif video["label"] == 1:
                            label = "infested"
                        os.makedirs(f"{args.extracted_dataset}/{codename}/{partition}/{label}/", exist_ok=True)
                        cv.imwrite(f"{args.extracted_dataset}/{codename}/{partition}/{label}/{video['label']}_{video['id']}_{frame_idx}.png", frame)

