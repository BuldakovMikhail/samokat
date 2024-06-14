import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def eval_prediction(y_batch_actual, y_batch_predicted):
    """Return batches of accuracy and f1 scores."""
    y_batch_actual_np = y_batch_actual.cpu().detach().numpy()
    y_batch_predicted_np = np.round(y_batch_predicted.cpu().detach().numpy())

    acc = accuracy_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np)
    f1 = f1_score(
        y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average="weighted"
    )

    return acc, f1


def training_step_mixup(dataloader, model, optimizer, loss_fn, if_freeze_bert, lam):
    """Method to train the model"""

    model.train()
    model.freeze_bert() if if_freeze_bert else model.unfreeze_bert()

    epoch_loss = 0

    for _, batch in enumerate(dataloader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        indexes = torch.randperm(len(input_ids)).to(device)

        model_answer = model(
            input_ids, attention_mask, input_ids[indexes], attention_mask[indexes], lam
        )

        outputs = torch.flatten(model_answer)

        mixup_labels = lam * labels.float() + (1.0 - lam) * labels.float()[indexes]

        optimizer.zero_grad()
        loss = loss_fn(outputs, mixup_labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()


def validation_step_mixup(dataloader, model):
    """Method to test accuracy"""

    model.eval()
    model.freeze_bert()

    size = len(dataloader)
    f1, acc = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            X = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            pred = model(X, attention_mask, X, attention_mask, 1)

            acc_batch, f1_batch = eval_prediction(y.float(), pred)
            acc += acc_batch
            f1 += f1_batch

        acc = acc / size
        f1 = f1 / size

    return acc, f1
