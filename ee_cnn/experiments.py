import subprocess
import argparse

def train(codename:str, dataset:str, epochs:int):
    script = "main.py"

    common_args = ["--joint-cross-validation-auto",
                   "--dataset", dataset,
                   "--epochs", epochs]

    augmix = common_args + ["--codename", codename]

    subprocess.run(["python", script] + augmix)

def extract(codename:str, dataset:str, models_directory:str):
    script = "main.py"

    args = ["--extract-preds",
            "--dataset", dataset,
            "--models-directory", models_directory,
            "--codename", codename]

    subprocess.run(["python", script] + args)

def experiment(codename:str):
    script = "main.py"

    args = ["--experiment",
            "--codename", codename]

    subprocess.run(["python", script] + args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, help="Train the ee-cnn")
    parser.add_argument("--extract", action=argparse.BooleanOptionalAction, help="Extract the ee-cnn model scores")
    parser.add_argument("--experiment", action=argparse.BooleanOptionalAction, help="Run an experiment on the test set with both ee-cnn and ViT4V models")
    parser.add_argument('--codename', type=str, help='Codename of the experiment')
    parser.add_argument('--dataset', type=str, help='Path to the dataset to be used')
    parser.add_argument('--models-directory', type=str, help='Path to directory containing models')
    parser.add_argument('--epochs', type=str, help='Training number of epochs')
    args = parser.parse_args()

    if args.codename is None:
        raise ValueError("Provide --codename")
    else:
        codename = args.codename
    if args.epochs is None:
        epochs = 30
    else:
        epochs = args.epochs
    if args.dataset is None:
        raise ValueError("Provide --dataset")
    else:
        dataset = args.dataset
    if args.train:
        train(codename=codename,
              dataset=dataset,
              epochs=epochs)

    if args.compare:
        compare(codename=codename)
    
    if args.extract:
        if args.models_directory is None:
            raise ValueError("Provide the --models-directory")
        extract(codename=codename, dataset=dataset, models_directory=args.models_directory)

    if args.experiment:
        experiment(codename)
