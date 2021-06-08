import argparse
import logging
from functools import partial

import datasets
import numpy as np
import trainer
import transformers
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
    set_seed,
)

logger = logging.getLogger(__name__)


def tokenize_data(examples, tokenizer, label2id):
    res = tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    if type(examples["label"][0]) == int:
        res["labels"] = examples["label"]
    elif type(examples["label"][0]) == float:
        res["labels"] = [int(label) for label in examples["label"]]
    else:
        res["labels"] = [label2id[label] for label in examples["label"]]
    return res


def prepare_optimizer(optimizer, model, weight_decay, lr):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    return optimizer


def main(args):
    set_seed(args.seed)
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    dataset = load_dataset(
        "json",
        cache_dir=args.cache,
        data_files={"train": args.train_data, "validation": args.dev_data},
    )
    labels = dataset["train"].unique("label")
    label2id = None
    if type(labels[0]) == str:
        label2id = {label: i for i, label in enumerate(sorted(labels))}
    elif type(labels[0]) == int:
        label2id = {f"label_{i}": i for i in sorted(labels)}
    elif type(labels[0]) == float:
        labels = [int(label) for label in labels]
        label2id = {f"label_{i}": i for i in sorted(labels)}
    num_labels = len(labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        label2id=label2id,
        id2label={v: k for k, v in label2id.items()},
        cache_dir=args.cache,
    )

    cols_to_remove = [
        k for k in list(dataset["train"][0].keys()) if k not in ["labels"]
    ]
    dataset = dataset.map(
        partial(tokenize_data, tokenizer=tokenizer, label2id=label2id),
        batched=True,
        remove_columns=cols_to_remove,
    )
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
    )

    train_dataloader = DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.train_batch_size,
    )
    eval_dataloader = DataLoader(
        dataset["validation"],
        collate_fn=data_collator,
        batch_size=args.eval_batch_size,
    )

    optimizer = prepare_optimizer(
        optimizer="adamw", model=model, weight_decay=args.weight_decay, lr=args.lr
    )
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    metric_f1 = load_metric("f1")
    metric_accuracy = load_metric("accuracy")

    def compute_metrics(predictions, labels):
        predictions = np.argmax(predictions, axis=1)

        result = metric_accuracy.compute(predictions=predictions, references=labels)

        for typ in ["macro", "micro", "weighted"]:
            result[f"f1_{typ}"] = metric_f1.compute(
                predictions=predictions, references=labels, average=typ
            ).pop("f1")

        return result

    trainer_obj = trainer.Trainer(
        model=model,
        tokenizer=tokenizer,
        model_dir=args.model_dir,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        accelerator=accelerator,
        dataloader_train=train_dataloader,
        dataloader_eval=eval_dataloader,
        batch_size_train=args.train_batch_size,
        batch_size_eval=args.eval_batch_size,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        metric_name=args.metric_name,
        metric_fn=compute_metrics,
    )

    trainer_obj.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT.")
    parser.add_argument(
        "--seed", default=42, required=False, help="Random seed for reproducibility"
    )

    md_args = parser.add_argument_group("Model and data")
    md_args.add_argument("--model", default="bert-base-cased")
    md_args.add_argument(
        "--train_data", type=str, help="Path to training data (json-format)"
    )
    md_args.add_argument(
        "--dev_data", type=str, help="Path to development data (json-format)"
    )
    md_args.add_argument(
        "--model_dir", type=str, required=True, help="model saving dir"
    )
    md_args.add_argument("--cache", default="./tmp")

    train_args = parser.add_argument_group("Training")
    train_args.add_argument("--lr", type=float, default=5e-05, help="Learning rate")
    train_args.add_argument(
        "--num_epochs",
        type=int,
        required=False,
        default=-1,
        help="Number of training epochs",
    )
    train_args.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="number of training steps, overrides 'epochs'",
    )
    train_args.add_argument(
        "--patience",
        type=int,
        default=-1,
        help="Number of stalled steps until early stopping, -1 for no early stopping",
    )
    train_args.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size for training batches",
    )
    train_args.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for dev/test batches",
    )
    train_args.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay regularization parameter",
    )
    train_args.add_argument(
        "--eval_steps", type=int, default=500, help="Evaluate on dev every n steps"
    )
    train_args.add_argument(
        "--save_steps", type=int, default=500, help="Save model every n steps"
    )
    train_args.add_argument(
        "--log_steps", type=int, default=500, help="Save model every n steps"
    )
    train_args.add_argument(
        "--metric_name",
        type=str,
        required=False,
        default="f1_micro",
        help="Metric name for evaluation",
        choices=["f1_samples", "f1_micro", "f1_macro", "f1_binary", "accuracy"],
    )
    train_args.add_argument(
        "--lr_scheduler",
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    train_args.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps in the lr scheduler.",
    )
    args = parser.parse_args()

    main(args)
