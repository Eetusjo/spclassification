import argparse
import logging
import os
from collections import deque
from functools import partial

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    get_scheduler,
    set_seed,
)


class Trainer:
    def __init__(
        self,
        model,
        model_dir,
        tokenizer,
        optimizer,
        accelerator,
        dataloader_train,
        dataloader_eval,
        batch_size_train,
        batch_size_eval,
        max_steps,
        eval_steps,
        save_steps,
        log_steps,
        metric_name,
    ):
        self.model = model
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.accelerator = accelerator

        self.dataloader_train = dataloader_train
        self.dataloader_eval = dataloader_eval
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.gradient_accumulation_steps = 1
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.log_steps = log_steps

        self.best_model_checkpoint = None
        self.best_model_score = None

        self.metric_name = metric_name
        self.metric = load_metric(self.metric_name)

    def compute_metrics(self, predictions, labels):
        if self.metric_name in ["accuracy", "f1"]:
            predictions = (predictions > 0.5).astype(int)

        if self.metric_name == "accuracy":
            result = self.metric.compute(predictions=predictions, references=labels)
        elif self.metric_name == "f1":
            result = self.metric.compute(
                predictions=predictions, references=labels, average="macro"
            )

        return result

    def evaluate(self, dataloader):
        self.model.eval()
        predictions, labels = [], []
        for step, batch in enumerate(self.dataloader_eval):
            outputs = self.model(**batch)
            predictions.extend(
                torch.sigmoid(self.accelerator.gather(outputs.logits))
                .detach()
                .view(-1)
                .tolist()
            )
            labels.extend(batch["labels"].tolist())
        self.model.train()
        eval_metric = self.compute_metrics(np.array(predictions), labels)
        return eval_metric

    def train(self):
        total_batch_size = self.batch_size_train * self.accelerator.num_processes

        num_samples = len(self.dataloader_train.dataset)
        logging.info("***** Running training *****")
        logging.info(f"  Num samples = {num_samples}")
        logging.info(f"  Batch size per device = {self.batch_size_train}")
        logging.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) "
            f"= {total_batch_size}"
        )
        logging.info(
            f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}"
        )
        logging.info(f"  Total optimization steps = {self.max_steps}")
        logging.info(
            f"  Total epochs approx. = "
            f"{(self.max_steps*total_batch_size)/num_samples:.2f}"
        )
        progress_bar = tqdm(
            range(self.max_steps), disable=not self.accelerator.is_local_main_process
        )
        losses = deque(maxlen=args.log_steps // 2)
        continue_training, completed_steps = True, 0
        while continue_training:
            self.model.train()
            for step, batch in enumerate(self.dataloader_train):
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                losses.append(loss.item())

                if completed_steps >= self.max_steps:
                    continue_training = False
                    break

                if completed_steps % self.log_steps == 0:
                    logging.info(
                        f"Step {completed_steps}: train loss {sum(losses)/len(losses):.5f}"
                    )

                eval_metric = None
                if (completed_steps > 0) and (completed_steps % self.eval_steps == 0):
                    logging.info("Evaluating.")
                    eval_metric = self.evaluate(self.dataloader_eval)
                    logging.info(f"Step {completed_steps}: {eval_metric}")

                if completed_steps % self.save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    chkpt_dir = os.path.join(
                        self.model_dir, f"checkpoint-{completed_steps}"
                    )
                    unwrapped_model.save_pretrained(
                        chkpt_dir, save_function=self.accelerator.save
                    )
                    self.accelerator.save(
                        self.optimizer, os.path.join(chkpt_dir, "optimizer.pt")
                    )
                    self.accelerator.save(
                        self.tokenizer, os.path.join(chkpt_dir, "tokenizer.pt")
                    )
                    # If we did not evaluate at this step, we must get the
                    # evaluate to get the metric for comparison to earlier checkpoints
                    if eval_metric is None:
                        eval_metric = self.evaluate(self.dataloader_eval)

                    if (self.best_model_score is None) or (
                        eval_metric[self.metric_name] > self.best_model_score
                    ):
                        self.best_model_score = eval_metric[self.metric_name]
                        self.best_model_checkpoint = chkpt_dir

                    # FIXME: USE MLFLOW ARTIFACT LOGGING
                    # if (self.mlflow_tracking_uri is not None) and (
                    #     self.mlflow_save_models == "all"
                    # ):
                    #     # FIXME: Save model to mlflow artifact store
                    #     pass


def tokenize_data(examples, tokenizer):
    res = tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    res["labels"] = examples["label"]
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
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.info(accelerator.state)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, cache_dir=args.cache
    )

    dataset = load_dataset(
        "json",
        cache_dir=args.cache,
        data_files={"train": args.train_data, "validation": args.dev_data},
    )

    cols_to_remove = [
        k for k in list(dataset["train"][0].keys()) if k not in ["labels"]
    ]
    dataset = dataset.map(
        partial(tokenize_data, tokenizer=tokenizer),
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

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        model_dir=args.model_dir,
        optimizer=optimizer,
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
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT.")
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
    # train_args.add_argument("--log_freq", type=int, default=250,
    #                         help="log training every N steps")
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
        default="f1",
        help="Metric name for evaluation",
    )
    args = parser.parse_args()

    main(args)
