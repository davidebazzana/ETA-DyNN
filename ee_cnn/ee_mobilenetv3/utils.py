from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pycalib.visualisations import plot_reliability_diagram
from torch.utils.tensorboard import SummaryWriter
from .joint_ee_mobilenetv3 import multi_exit_loss
from misc import ReturnedPredictionsQuality
from codecarbon import track_emissions
from torchvision.utils import draw_bounding_boxes

def joint_train(model, train_data_loader, test_data_loader, num_epochs, device, save_model:bool=True, extract_scores:bool=False, model_path:str|None=None, codename:str|None=None):
    writer = SummaryWriter() if codename is None else SummaryWriter(log_dir=f"../logs/runs/{codename}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_loss = 1_000_000

    lower_return_threshold = 0.5
    upper_return_threshold = 0.5

    optimizer = torch.optim.Adam([
        {"params": model.features.parameters(), "lr": 1e-5},
        {"params": model.exits.parameters(), "lr": 1e-3},
        {"params": model.classifier[:-1].parameters(), "lr": 1e-5},
        {"params": model.classifier[-1].parameters(), "lr": 1e-3}
    ])

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(1, num_epochs+1):
        torch.set_printoptions(precision=5)
        print("=========================================")
        print(f'Epoch {epoch}')

        n = 0
        loss_cumulative_avg = 0
        # Number of exits: model exit points + final classification head 
        rpq = ReturnedPredictionsQuality(number_of_exits=len(model.exit_points)+1)
        model.train()
        for samples, labels in tqdm(train_data_loader):
            samples = samples.to(device)
            labels = labels.to(device)

            out = model(samples)
            
            loss = multi_exit_loss(out, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            props = [nn.functional.sigmoid(exit_predictions) for exit_predictions in out]
            rpq.update(labels, props)
            if n == 0:
                loss_cumulative_avg = loss.item()
            else:
                loss_cumulative_avg = (loss.item() + n*loss_cumulative_avg)/(n+1)
            n += 1

        rpq.compute_metrics(summary_writer=writer, dataset="train", epoch=epoch)
        writer.add_scalar('Loss/train', loss_cumulative_avg, epoch)
        print(f'Train loss: {loss_cumulative_avg}')

        loss_cumulative_avg, rpq_test_res = joint_test(model=model,
                                                       data_loader=test_data_loader,
                                                       device=device,
                                                       epoch=epoch,
                                                       writer=writer)
        writer.add_scalar('Loss/test', loss_cumulative_avg, epoch)
        print(f'Test loss: {loss_cumulative_avg}')

        if loss_cumulative_avg < best_loss:
            best_loss = loss_cumulative_avg

        if save_model and epoch == num_epochs:
            if model_path is None:
                model_path = './models/{}_{}_{}.pt'.format(model.__class__.__name__, timestamp, epoch)
            torch.save(model.state_dict(), model_path)

        scheduler.step()

    writer.close()

    if extract_scores:
        _, _, scores, labels = joint_test(model=model,
                                          data_loader=test_data_loader,
                                          device=device,
                                          epoch=epoch,
                                          writer=writer,
                                          extract_scores=True)

        return rpq_test_res, scores, labels
    
    return rpq_test_res

def joint_test(model, data_loader, device:str, epoch:int=None, writer:SummaryWriter=None, extract_scores:bool=False, scores_path:str="./scores/", calibrate:bool=False):
    n = 0
    loss_cumulative_avg = 0
    model.eval()

    rpq = ReturnedPredictionsQuality(number_of_exits=len(model.exit_points)+1)
    with torch.no_grad():
        for samples, labels in tqdm(data_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            
            out = model(samples)
            # loss = multi_exit_loss(out, labels.float(), weights=[0.2, 0.3, 0.5])
            loss = multi_exit_loss(out, labels.float())
            
            props = [nn.functional.sigmoid(exit_predictions) for exit_predictions in out]

            rpq.update(labels, props)
            
            if n == 0:
                loss_cumulative_avg = loss.item()
            else:
                loss_cumulative_avg = (loss.item() + n*loss_cumulative_avg)/(n+1)
            n += 1

    rpq_res = rpq.compute_metrics(summary_writer=writer, dataset="val", epoch=epoch)
    
    if extract_scores:
        scores, labels = rpq.extract_scores(model_name=model.__class__.__name__,
                                            scores_directory=scores_path)
        return loss_cumulative_avg, rpq_res, scores, labels

    return loss_cumulative_avg, rpq_res
