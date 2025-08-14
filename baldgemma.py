from argparse import ArgumentParser
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithFlattening, TrainingArguments
from trl import SFTTrainer
import torch
import torch.nn.functional as F
import deepspeed

class Embeddings(torch.nn.Module):
    def __init__(self, low_rank_embed: torch.nn.Module):
        super().__init__()
        self.low_rank_embed = low_rank_embed

    def forward(self, x):
        return self.low_rank_embed(x)

class BaldHead(torch.nn.Module):
    def __init__(self, low_rank_embed: torch.nn.Module):
        super().__init__()
        self.low_rank_embed = low_rank_embed

    def forward(self, x):
        intermediate = F.linear(x, self.low_rank_embed[1].weight.T)
        return F.linear(intermediate, self.low_rank_embed[0].weight)
    
@torch.no_grad()
def decompose(w, rank: int):
    """ Init low rank embeddings with svd."""
    r = rank
    u, s, v = torch.linalg.svd(w.to(dtype=torch.float32), full_matrices=False)
    w1 = u @ torch.sqrt(torch.diag(s)[:, :r])
    w2 = torch.sqrt(torch.diag(s)[:r, :]) @ v
    w1 = w1.to(dtype=w.dtype)
    w2 = w2.to(dtype=w.dtype)
    embedding = torch.nn.Embedding(w1.shape[0], w1.shape[1])
    embedding.weight = torch.nn.Parameter(w1)
    proj = torch.nn.Linear(w2.shape[0], w2.shape[1], bias=False, dtype=w.dtype)
    proj.weight = torch.nn.Parameter(w2.T.contiguous())
    return torch.nn.Sequential(embedding, proj)

def load_model(model_name: str, rank: int = 256):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if rank != 0:
        # Since the input/output embeddings are tied, we only need to decompose one of them.
        low_rank_embed = decompose(model.model.embed_tokens.weight, rank)
        model.model.embed_tokens = Embeddings(low_rank_embed)
        model.lm_head = BaldHead(low_rank_embed)
        logging.info(f"Low Rank Parameters: {sum(p.numel() for p in model.model.embed_tokens.parameters()):,}")
    print(model)
    return model, tokenizer

def train(model: torch.nn.Module, dataset, tokenizer, output_dir: str):
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=100,
        save_strategy="steps",
        learning_rate=2e-5,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        push_to_hub=False,
        report_to="none",
        save_steps=4_000,
        deepspeed={
                    "train_batch_size" : "auto",
                    "train_micro_batch_size_per_gpu": "auto",
                    "steps_per_print": 1,
                    "zero_optimization": {
                        "stage": 1
                    },
                    "bf16": {
                        "enabled": True
                    }
                    }
        )
    trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    data_collator=DataCollatorWithFlattening(return_tensors="pt"),
    args=args
    )
    trainer.train()
    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--rank", type=int, default=160)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceTB/smol-smoltalk")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument("--output_dir", type=str, default="./bald_qwen")
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = load_model(model_name=args.model_name, rank=args.rank)
    print(model)
    print(f"{model.num_parameters():,}")
    dataset = load_dataset(args.dataset_name, split="train")
    train(model, dataset, tokenizer, args.output_dir)
