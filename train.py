import torch
from tqdm.auto import tqdm
from model import BertWithClassifierMixUp
from transformers import AdamW
from torch import nn
from dataset import train_dataloader, eval_dataloader

from helpers import training_step_mixup, validation_step_mixup, device


if __name__ == "__main__":
    # parameters
    num_of_epochs = 13
    learning_rate = 1e-5
    batch_size = 16
    hidden_layers = 8

    bert_on = 3
    mixup_on = 6

    model_mixup = BertWithClassifierMixUp(linear_size=hidden_layers)
    model_mixup.to(device)
    optimizer = AdamW(model_mixup.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    if_freeze_bert = True
    lam = 1
    best_acc = 0

    path = "./best_model.pt"

    tqdm.pandas()

    for i in tqdm(range(num_of_epochs)):
        print("Epoch: #{}".format(i + 1))

        if i < bert_on:
            if_freeze_bert = True
            print("Bert is freezed")
            lam = 1
        else:
            if_freeze_bert = False
            print("Bert is not freezed")

        if i >= mixup_on:
            print("Mix Up")
            lam = 0.5

        training_step_mixup(
            train_dataloader, model_mixup, optimizer, loss_fn, if_freeze_bert, lam
        )
        train_acc, train_f1 = validation_step_mixup(train_dataloader, model_mixup)
        val_acc, val_f1 = validation_step_mixup(eval_dataloader, model_mixup)

        print("Training results: ")
        print("Acc: {:.3f}, f1: {:.3f}".format(train_acc, train_f1))

        print("Validation results: ")
        print("Acc: {:.3f}, f1: {:.3f}".format(val_acc, val_f1))

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model_mixup, path)
