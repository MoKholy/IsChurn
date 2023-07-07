import copy
import os
import torch
from transformer import get_model
import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pytorch_warmup as warmup
import argparse
import pandas as pd
import numpy as np
from accelerate import Accelerator

# class weights for the loss function
CLASS_WEIGHTS = [0.5296650705183834, 8.927419710500093]

accelerator = Accelerator()

# create custom dataset class
class CustomDataset(Dataset):
    def __init__(self, file, nrows=None):
        if nrows is not None:
            self.data = pd.read_csv(file, nrows=nrows)
        else:
            self.data = pd.read_csv(file)
        
        # features are everything but last column
        self.features = self.data.iloc[:, :-1]
        # labels are last column
        self.labels = self.data.iloc[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.features.iloc[index].values, dtype=torch.float32)
        y = torch.tensor(self.labels.iloc[index], dtype=torch.float32)
        return x, y


# load date and trainloader from csv file
def load_data(file, nrows=None):

    # load data from csv file 
    dataset = CustomDataset(file, nrows=nrows)

    return dataset


def train_one_epoch(model, optimizer, criterion, generator, len_dataset,
                    lr_scheduler, warmup_scheduler, epoch, batch_size, device="cpu"):
    
    model.train()
    tot_loss = 0.0
    tot_acc = 0.0
    train_total = 0
    train_correct = 0
                          
    for batch, labels in tqdm(generator):

        # adjust for accelerator 
        labels = labels.unsqueeze(dim=1)
        optimizer.zero_grad()
        logits = model(batch)
        lr_scheduler.step(epoch)
        warmup_scheduler.dampen()
        loss = criterion(logits, labels)

        accelerator.backward(loss)
        # loss.backward()

        # squeeze labels again
        labels = labels.squeeze()

        # Calculate accuracy
        train_total += labels.size(0)
        predictions = torch.round(torch.sigmoid(logits))
        train_correct += torch.sum(predictions == labels).item()

        optimizer.step()
        tot_loss += loss.item()
    
    # Calculate accuracy
    tot_acc = train_correct / train_total
    tot_loss /= len_dataset // batch_size
    return tot_loss, tot_acc



def get_metrics_dict(val_predicted, val_true):

    from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
    # get recall, precision, f1 score, and accuracy of binary classification using sklearn
    recall = recall_score(val_true, val_predicted)
    precision = precision_score(val_true, val_predicted)
    f1 = f1_score(val_true, val_predicted)
    accuracy = accuracy_score(val_true, val_predicted)

    # get confusion matrix using sklearn
    cm = confusion_matrix(val_true, val_predicted)

    # get true positive, true negative, false positive, and false negative from confusion matrix
    tn, fp, fn, tp = cm.ravel()

    # calculate true positive rate, false positive rate, true negative rate, and false negative rate
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tnr = tn / (tn + fp)
    fnr = fn / (fn + tp)

    # calculate balanced accuracy
    balanced_accuracy = (tpr + tnr) / 2

    # return metrics dictionary

    return {"recall": recall, "precision": precision, "f1": f1, "accuracy": accuracy, "balanced_accuracy": balanced_accuracy,
            "tpr": tpr, "fpr": fpr, "tnr": tnr, "fnr": fnr}    
    
def print_metrics_dict(metrics_dict):
    print("recall:", metrics_dict["recall"])
    print("precision:", metrics_dict["precision"])
    print("f1:", metrics_dict["f1"])
    print("accuracy:", metrics_dict["accuracy"])
    print("balanced_accuracy:", metrics_dict["balanced_accuracy"])
    print("tpr:", metrics_dict["tpr"])
    print("fpr:", metrics_dict["fpr"])
    print("tnr:", metrics_dict["tnr"])
    print("fnr:", metrics_dict["fnr"])

def evaluate_one_epoch(model, criterion, generator, len_dataset, batch_size):
    
    model.eval()
    val_loss = 0.0

    val_output_pred = []
    val_output_true = []
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            # batch, labels = batch.to(device), labels.to(device)
            labels = labels.unsqueeze(dim=1)
            logits = model(batch)
            loss = criterion(logits, labels)

            # squeeze labels again
            labels = labels.squeeze()
            val_loss += loss.item()
            val_output_pred += torch.round(torch.sigmoid(logits)).cpu().tolist()
            val_output_true += labels.cpu().tolist()
    
    # Calculate accuracy
    val_loss /= len_dataset // batch_size

    # get metrics dictionary
    metrics_dict = get_metrics_dict(val_output_pred, val_output_true)

    return val_loss, metrics_dict


def train_pipeline(args):

    # load dataset from csv file
    train_dataset = load_data(args.train_dataset, nrows=args.limit_rows)
    test_dataset = load_data(args.test_dataset, nrows=args.limit_rows)


    # get dataset size
    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)


    # get batch size
    batch_size = args.batch_size

    # create dataloader
    train_generator = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_generator = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )




    if args.load_weights:

        # put models in accelerator
        

        model = get_model(args.n_items, args.d_model, args.heads, args.dropout,
                          args.n_layers, args.hidden_size, args.weights_path)
        print("model loaded from", args.weights_path)
    else:
        model = get_model(args.n_items, args.d_model, args.heads, args.dropout,
                          args.n_layers, args.hidden_size, None)

    # pos_weight=torch.tensor(CLASS_WEIGHTS[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(CLASS_WEIGHTS[1])) # weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, pos_weight=torch.tensor(CLASS_WEIGHTS[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-9)
    # warmup lr
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.warmup_epochs], gamma=0.1)
    if args.warmup_type == 'linear':
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    elif args.warmup_type == 'exponential':
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    elif args.warmup_type == 'radam':
        warmup_scheduler = warmup.RAdamWarmup(optimizer)
    else:
        warmup_scheduler = warmup.LinearWarmup(optimizer, 1)
    warmup_scheduler.last_step = -1  # initialize the step counter

    best_model = copy.deepcopy(model.state_dict())
    best_test_loss = np.inf

    model, optimizer, lr_scheduler, warmup_scheduler, train_generator, test_generator = accelerator.prepare(
            model, optimizer, lr_scheduler, warmup_scheduler, train_generator, test_generator
    )

    start = time.time()
    if not args.no_train:
        print("training...")
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_generator, 
                                                    train_dataset_size,lr_scheduler, warmup_scheduler, 
                                                    epoch, args.batch_size)
            print("epoch {} | train loss: {} | train acc: {}".format(epoch + 1, train_loss, train_acc))

            
            if args.save_weights_epoch is not None and epoch % args.save_weights_epoch == 0:
                torch.save(model.state_dict(), os.path.join(args.weights_path, "weights_{}.pth".format(epoch)))
                print("model saved at", os.path.join(args.weights_path, "weights_{}.pth".format(epoch)))

        print("finished training in", time.time() - start)

    

    print("testing...")
    # get metrics dict
    
    test_loss, test_metrics = evaluate_one_epoch(model, criterion, test_generator, test_dataset_size, args.batch_size)
    print("test loss: {}".format(test_loss))

    # print metrics dict
    print_metrics_dict(test_metrics)

    if args.save_weights > 0:
        torch.save(model.state_dict(), os.path.join(args.weights_path, "weights.pth"))
        print("model saved at", os.path.join(args.weights_path, "weights.pth"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default="final_data/train.csv")
    parser.add_argument('--test_dataset', type=str, default="final_data/test.csv")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--n_items', type=int, default=1,
                        help='number of different items that can be recommended')
    parser.add_argument('--no_train', default=False, action='store_true',
                        help='if True skip training')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_type', type=str, default="radam",
                        help='choose from "linear", "exponential", "radam", "none"')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--d_model', type=int, default=16,
                        help='dimension of the model')
    parser.add_argument('--heads', type=int, default=2,
                        help='number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of Transformer layer')
    parser.add_argument('--load_weights', default=False, action='store_true')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--save_weights', default=False, action='store_true')
    parser.add_argument('--weights_path', type=str, default="weights",
                        help='path where to store the model weights')
    parser.add_argument('--limit_rows', type=int, default=None,
                        help='if not None limit the size of the dataset')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='hidden size of the encoders forward layer')
    parser.add_argument('--save_weights_epoch', type=int, default=None,
                        help='during training save a copy of the model weights every "save_weights_epoch" epochs')
    args = parser.parse_known_args()[0]
    return args


"""
Some usage examples
use a small dataset to make sure the code can run:
- python model/transformer_model.py --limit_rows 100 --epochs 10 --warmup_epochs 2
process the data without training
- python model/transformer_model.py --save_data --no_load_data --no_train --dataset "data/train_reduced.csv"
train the model for 100 epochs with 10 warmup epochs and save final weights
- python model/transformer_model.py --save_weights --epochs 100 --warmup_epochs 10
load pretrained weights and test without training
- python model/transformer_model.py --load_weights --no_train
"""
if __name__ == '__main__':
    args = get_args()
    train_pipeline(args)
