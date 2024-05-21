import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import model
import data_process
import transformer.model


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
                self.criterion(
                    x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
                )
                / norm
        )
        return sloss.data * norm, sloss


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def run_epoch(data_iter, model, loss_compute, mode):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].dim, factor=1.0, warmup=400
        ),
    )
    start_time = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss, loss_node = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if (mode == "train"):
            loss_node.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
        if i % 50 == 1:
            elapsed = time.time() - start_time
            print(f"Epoch Step: {i} Loss: {loss / batch.ntokens:.6f} Tokens per Sec: {elapsed / tokens :.6f}")
            start_time = time.time()
            tokens = 0
    return total_loss / total_tokens


MAX_LEN = 50000
ens, zhs = data_process.data_load("./data/news-commentary-v16.en-zh.tsv")
en_dic = data_process.create_dic(ens, MAX_LEN)
zh_dic = data_process.create_dic(zhs, MAX_LEN)

## word2id
ens, zhs = data_process.word2id(ens, zhs, en_dic, True)

batches = data_process.split_batch(ens, zhs, 32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.make_model(50002, 50002, N=3)
criterion = transformer.model.LabelSmoothing(50002, padding_idx=1)
loss_computer = SimpleLossCompute(model.generator, criterion)
run_epoch(batches, model, loss_computer, "train")
