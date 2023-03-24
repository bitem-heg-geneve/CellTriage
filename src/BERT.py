import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

from BaseModel import BaseModel
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_examples_to_inputs(
    self,
    example_texts,
    example_labels,
    label2idx,
    max_seq_length,
    tokenizer,
    verbose=0,
):
    """Loads a data file into a list of `InputBatch`s."""

    input_items = []
    examples = zip(example_texts, example_labels)
    for ex_index, (text, label) in enumerate(examples):
        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label2idx[label]

        input_items.append(
            BertInputItem(
                text=text,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
            )
        )

    return input_items


def get_data_loader(features, max_seq_length, batch_size, shuffle=True):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader


class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class BertBaseUncased(BaseModel, max_seq_length=512):
    def __init__(self):
        BaseModel.__init__(self)
        assert max_seq_length <= 512
        self.max_seq_length = max_seq_length
        self.gradient_accumulation_steps = 1
        self.num_train_epochs = 1
        self.learning_rate = 5e-5
        self.warmup_proportion = 0.1
        self.max_grad_norm = 5
        self.batch_size = 16

        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.LEARNING_RATE, correct_bias=False
        )

    def train(self, X_train, y_train, X_dev, y_dev):
        labels = list(set(y_train))
        self.label2idx = {label: idx for idx, label in enumerate(labels)}

        self.model = BertForSequenceClassification.from_pretrained(
            self.bert_model, num_label=len(self.label2idx)
        )
        self.model.to(device)

        train_features = convert_examples_to_inputs(X_train, y_train)
        dev_features = convert_examples_to_inputs(X_dev, y_dev)
        train_dataloader = get_data_loader(
            train_features, self.max_seq_length, self.batch_size
        )
        dev_dataloader = get_data_loader(
            dev_features, self.max_seq_length, self.batch_size
        )

        # scheduler
        n_train_steps = int(
            len(train_dataloader.dataset)
            / self.batch_size
            / self.gradient_accumulation_steps
            * self.num_train_epochs
        )
        n_warmup_steps = int(self.warmup_proportion * n_train_steps)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=n_warmup_steps,
            num_training_steps=num_train_steps,
        )

        return self

    def predict_proba(self, X, normalize=False):
        if len(X) > 0:
            y_pred = None  # TODO

            if normalize:
                y_pred = (y_pred - y_pred.min()) / (
                    y_pred.max() - y_pred.min() + 0.0000001
                )  # min-max
        return y_pred

    def predict(self, X, normalize=False):
        if len(X) > 0:
            y_pred = None  # TODO
        return y_pred

    def train_predict(self, X_train, y_train, X_val):
        return BERT().train(X_train, y_train).predict(X_val)


def test():
    from pprint import pprint
    from time import time

    from TestData import TestData

    DATA_FP = "data/20230201/test/TestData_100_50.pkl"
    MODEL_FP = "models/test/LogReg_100_50.pkl"

    # data = TestData()
    # data.create(n_train=100, n_test=50)
    # data.save(DATA_FP)
    print("\nLoad data")
    data = TestData.load(DATA_FP)
    print(f"n_train: {len(data.y_train)}, n_test: {len(data.y_test)}")

    print("\nCrossvalidate model")
    t0 = time()
    pprint(LogReg().cv(data.X_train, data.y_train, kfold=5, pool=3))
    print(f"Done in {time() - t0:.3f}s")

    print("\nTrain model and save to disk")
    t0 = time()
    model = LogReg().train(data.X_train, data.y_train)
    model.save(MODEL_FP)
    print(f"Done in {time() - t0:.3f}s")

    print("\nLoad model from disk and evaluate on test data")
    t0 = time()
    m2 = LogReg.load(MODEL_FP)
    pprint(m2.eval(data.X_test, data.y_test))
    print(f"Done in {time() - t0:.3f}s")


if __name__ == "__main__":
    test()
