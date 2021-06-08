import logging
import os
from collections import deque

import numpy as np
import torch
from tqdm import tqdm


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        model_dir,
        tokenizer,
        optimizer,
        scheduler,
        accelerator,
        dataloader_train,
        dataloader_eval,
        batch_size_train,
        batch_size_eval,
        max_steps,
        eval_steps,
        save_steps,
        log_steps,
        patience,
        metric_name,
        metric_fn,
    ):
        self.model = model
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
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

        self.patience = patience

        self.best_model_checkpoint = None
        self.best_model_score = None

        self.metric_name = metric_name
        self.metric_fn = metric_fn

    def evaluate(self, dataloader):
        self.model.eval()
        predictions, labels = [], []
        eval_bar = tqdm(
            range(len(self.dataloader_eval)),
            disable=not self.accelerator.is_local_main_process,
        )
        for step, batch in enumerate(self.dataloader_eval):
            outputs = self.model(**batch)
            predictions.extend(
                self.accelerator.gather(outputs.logits).detach().tolist()
            )
            labels.extend(batch["labels"].tolist())
            eval_bar.update(1)
        self.model.train()
        eval_metric = self.metric_fn(np.array(predictions), labels)
        return eval_metric

    def train(self):
        total_batch_size = self.batch_size_train * self.accelerator.num_processes

        num_samples = len(self.dataloader_train.dataset)
        logger.info("***** Running training *****")
        logger.info(f"  Num samples = {num_samples}")
        logger.info(f"  Batch size per device = {self.batch_size_train}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) "
            f"= {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self.max_steps}")
        logger.info(
            f"  Total epochs approx. = "
            f"{(self.max_steps*total_batch_size)/num_samples:.2f}"
        )
        progress_bar = tqdm(
            range(self.max_steps), disable=not self.accelerator.is_local_main_process
        )
        losses = deque(maxlen=self.log_steps // 2)
        continue_training, completed_steps = True, 0
        best_metric, no_improvement = None, 0
        while continue_training:
            self.model.train()
            for step, batch in enumerate(self.dataloader_train):
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                losses.append(loss.item())

                if completed_steps >= self.max_steps:
                    continue_training = False
                    break

                if completed_steps % self.log_steps == 0:
                    logger.info(
                        f"Step {completed_steps} | "
                        f"train loss {sum(losses)/len(losses):.5f} | "
                        f"lr {self.scheduler.get_last_lr()[0]:.3E}"
                    )

                eval_metric = None
                if (completed_steps > 0) and (completed_steps % self.eval_steps == 0):
                    logger.info("Evaluating.")
                    eval_metric = self.evaluate(self.dataloader_eval)
                    logger.info(f"Step {completed_steps}: {eval_metric}")

                    if (best_metric is None) or (
                        eval_metric[self.metric_name] > best_metric
                    ):
                        best_metric = eval_metric[self.metric_name]
                        no_improvement = 0
                    else:
                        no_improvement += 1
                        if no_improvement >= self.patience:
                            logging.info(
                                f"Stalled for {no_improvement} times. Exiting training."
                            )
                            continue_training = False
                            break

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

                    # FIXME: USE MLFLOW ARTIFACT
                    # if (self.mlflow_tracking_uri is not None) and (
                    #     self.mlflow_save_models == "all"
                    # ):
                    #     # FIXME: Save model to mlflow artifact store
                    #     pass
