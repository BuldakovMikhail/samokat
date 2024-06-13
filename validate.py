import torch
from model import BertWithClassifierMixUp
from dataset import eval_dataloader
from helpers import validation_step_mixup, device


if __name__ == "__main__":
    hidden_layers = 8
    model_mixup = BertWithClassifierMixUp(hidden_layers)
    model_mixup.to(device)

    path = "./best_model.pt"

    model_mixup.load_state_dict(torch.load(path))

    val_acc, val_f1 = validation_step_mixup(eval_dataloader, model_mixup)

    print("Validation results: ")
    print("Acc: {:.3f}, f1: {:.3f}".format(val_acc, val_f1))
