import torch
from model import BertWithClassifierMixUp
from transformers import BertTokenizer, BertModel
from dataset import test_dataloader
from helpers import validation_step_mixup, device


if __name__ == "__main__":

    path = "./best_model.pt"

    model_mixup = torch.load(path)
    # model_mixup.to(device)

    val_acc, val_f1 = validation_step_mixup(test_dataloader, model_mixup)

    print("Validation results: ")
    print("Acc: {:.3f}, f1: {:.3f}".format(val_acc, val_f1))
