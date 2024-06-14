from datasets import load_dataset
from transformers import BertTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(example):
    return tokenizer(example["text"])


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")


train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
test_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=True, batch_size=8, collate_fn=data_collator
)
