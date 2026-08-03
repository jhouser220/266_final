"""
Stage 1 - contrastive learning training script for ESM-LLAMA modality alignment 
on Esm2LlamaInstructForCausalLM model. 

Without LoRA. 

DistributedDataParallel training script implemented from scratch. 

The script currently supports gradient accumulation, AutoMixedPrecision, 
inter-epoch evaluation. 

The script currently does not support save/load pretrained, gradient checkpointing 
or generation under FSDP. 

reference for AMP: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html 

* The script is designed for multi-GPU parallelism on single node.
* On the cluster, print(...) will go to stdout and tqdm(...) will go to stderr.
"""

import argparse
from datetime import datetime
import os
from typing import Any, Dict, Literal, Union

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel
from transformers import EsmModel, LlamaModel, LlamaForCausalLM

from dataset import Prot2TextInstructDataset, Prot2TextInstructDataLoader
from models import (
    ModalityAdapter, 
    ModalityAdapterConfig, 
    Esm2LlamaInstructForCausalLM
)
import scripts.utils_argparse as utils_argparse


argParser = argparse.ArgumentParser()

argParser.add_argument("--esm_path", type=str)
argParser.add_argument("--llama_path", type=str)
argParser.add_argument("--root_dataset_dir", type=str)
argParser.add_argument("--root_csv_dir", type=str)
argParser.add_argument("--save_checkpoint_dir", type=str)
argParser.add_argument("--load_model_checkpoint_path", type=str, default="")
argParser.add_argument("--load_optimizer_scheduler_checkpoint_path", type=str, default="")

argParser.add_argument("--torch_dtype", type=utils_argparse.str2dtype)
argParser.add_argument("--batch_size_per_device", type=int)
argParser.add_argument("--num_epochs", type=int)
argParser.add_argument("--save_every_epochs", type=int)
argParser.add_argument("--gradient_accumulation_steps", type=int)
argParser.add_argument("--learning_rate", type=float)
argParser.add_argument("--gradient_clipping", type=float, default=None)
argParser.add_argument("--scheduler_gamma", type=float)
argParser.add_argument("--random_seed", type=int)
argParser.add_argument("--contrastive_num_segments", type=int)

argParser.add_argument("--train_split", type=str)
argParser.add_argument("--eval_split", type=str)
argParser.add_argument("--debug_trim_train_split", type=int, default=None)
argParser.add_argument("--debug_trim_eval_split", type=int, default=None)


class BatchInfoNCELoss(torch.nn.Module):
    r"""
    Batch version of InfoNCE loss for contrastive learning. Positive and negative 
    pairs are picked within every batch.

    $L_{\text{InfoNCE}} = -\log \frac{\exp(s(x, x^+)/\tau)}{\sum_{x'} \exp(s(x, x')/\tau)}$

    temperature: float, temperature parameter for the softmax function. The smaller
    the temperature, the sharper the softmax distribution. 
    """
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, batch_output1: torch.Tensor, batch_output2: torch.Tensor):
        similarity_matrix = torch.mm(batch_output1, batch_output2.t())
        logits = similarity_matrix / self.temperature  # (bsz, bsz)
        numerator = torch.exp(torch.diag(logits)).unsqueeze(1)  # (bsz, 1)
        denominator = torch.sum(torch.exp(logits), dim=1, keepdim=True)  # (bsz, 1)
        return - torch.log(numerator / denominator).mean()


class SegmentedBatchInfoNCELoss(torch.nn.Module):
    """Segmented version of BatchInfoNCELoss for contrastive learning."""
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
            self, 
            segment_output1: torch.Tensor,  # (segment_size, hidden_dim)
            batch_output2: torch.Tensor,  # (bsz, hidden_dim)
            labels: torch.Tensor  # (segment_size,)
        ):
        segment_size = segment_output1.size(0)

        similarity_matrix = torch.mm(segment_output1, batch_output2.t())
        logits = similarity_matrix / self.temperature  # (segment_size, bsz)
        
        numerator = torch.exp(logits[torch.arange(segment_size), labels]).unsqueeze(1)  # (segment_size, 1)
        denominator = torch.sum(torch.exp(logits), dim=1, keepdim=True)  # (segment_size, 1)
        
        return - torch.log(numerator / denominator).mean()


def load_model(args: Dict[str, Any]) -> PreTrainedModel:
    """
    Standard API for different models. Used in both `train` and `generate`.
    Load base model of the given name, and load weights from the checkpoint path 
    if provided.

    Returned model should be on CPU and under default data type.
    A general checkpoint shall contain the model state dict, optimizer state dict,
    and scheduler state dict.
    """
    esm_encoder = EsmModel.from_pretrained(
        args["esm_path"], 
        add_pooling_layer=False,
        torch_dtype=args["torch_dtype"], 
        device_map="cpu"
    )
    llama_decoder = LlamaForCausalLM.from_pretrained(
        args["llama_path"], 
        torch_dtype=args["torch_dtype"], 
        device_map="cpu"
        )

    adapter_config = ModalityAdapterConfig(
        input_dim=esm_encoder.config.hidden_size,
        intermediate_dim=2048,
        output_dim=llama_decoder.config.hidden_size,
    )
    adapter = ModalityAdapter(adapter_config)
    adapter.to(args["torch_dtype"])
    
    model = Esm2LlamaInstructForCausalLM(
        esm_encoder=esm_encoder,
        adapter=adapter,
        llama_decoder=llama_decoder,
    )

    if args["load_model_checkpoint_path"]:
        print(f"Loading {args['load_model_checkpoint_path']}")
        model_state_dict = torch.load(
            args["load_model_checkpoint_path"], 
            weights_only=True, 
            map_location="cpu"  # load to CPU first
            # will be loaded to where the weights were saved from if not specified
        )
        model.load_state_dict(model_state_dict)

    # WARNING: esm and llama weights are fixed
    model.esm_encoder.requires_grad_(False)
    model.llama_decoder.requires_grad_(False)

    return model


def readout_embeddings(
        embeddings: torch.Tensor,  # (bsz, seq_len, hidden_dim)
        attention_mask: torch.Tensor,  # (bsz, text_len)
        readout_fn: Literal["last", "mean", "std", "mix"], 
) -> torch.Tensor:
    """
    Perform a readout operation on the output sequence embeddings of the forward 
    pass, given the attention mask. 
    """
    if readout_fn == "last":
        # inputs must be right padded
        # for left padding simply take the last token and do not use this function
        last_token_indices = attention_mask.sum(dim=1) - 1  # (bsz,)
        batch_indices = torch.arange(
            attention_mask.size(0), 
            device=attention_mask.device
        )  # (bsz,)
        return embeddings[batch_indices, last_token_indices, :]  # (bsz, hidden_dim)

    elif readout_fn == "mean":
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        sum_embeddings = masked_embeddings.sum(dim=1)  # (bsz, hidden_dim)
        count_attn_mask = attention_mask.sum(dim=1, keepdim=True)  # (bsz, 1)
        return sum_embeddings / count_attn_mask  # (bsz, hidden_dim)
    
    elif readout_fn == "std":
        mean_embeddings = readout_embeddings(
            embeddings=embeddings, 
            attention_mask=attention_mask, 
            readout_fn="mean"
        )  # (bsz, hidden_dim)
        diff_embeddings = embeddings - mean_embeddings.unsqueeze(1)
            # (bsz, text_len, hidden_dim)
        diff_embeddings_2 = diff_embeddings.pow(2) 
        masked_diff_embeddings_2 = diff_embeddings_2 * attention_mask.unsqueeze(-1)
        sum_diff_embeddings_2 = masked_diff_embeddings_2.sum(dim=1)  # (bsz, hidden_dim)
        count_attn_mask = attention_mask.sum(dim=1, keepdim=True)  # (bsz, 1)
        return (sum_diff_embeddings_2 / count_attn_mask).sqrt()  # (bsz, hidden_dim)

    elif readout_fn == "mix": 
        mean_embeddings = readout_embeddings(
            embeddings=embeddings, 
            attention_mask=attention_mask, 
            readout_fn="mean"
        )
        std_embeddings = readout_embeddings(
            embeddings=embeddings, 
            attention_mask=attention_mask, 
            readout_fn="std"
        )
        return torch.cat([mean_embeddings, std_embeddings], dim=1)  # (bsz, 2 * hidden_dim)


def get_sequence_embeddings(
        model: Esm2LlamaInstructForCausalLM, 
        sequence_input_ids: torch.Tensor, 
        sequence_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Take mean pooling and the std pooling of the adapter outputs for each valid 
    token in the sequence as the sequence embedding for contrastive learning. 

    sequence_input_ids: (bsz, max_seq_len)
    sequence_attention_mask: (bsz, max_seq_len)
    return: (bsz, decoder_hidden_size)
    """
    with torch.no_grad():  # WARNING: esm encoder fixed
        encoder_output = model.forward(
            protein_input_ids=sequence_input_ids,
            protein_attention_mask=sequence_attention_mask,
            return_encoder_outputs=True,
        )

    adapter_output = model.adapter(encoder_output[0])
    protein_attention_mask = sequence_attention_mask
    # adapter_output: (bsz, max_seq_len, decoder_hidden_size)

    return readout_embeddings(
        embeddings=adapter_output, 
        attention_mask=protein_attention_mask, 
        readout_fn="mix"
    )  # (bsz, decoder_hidden_size)


def get_description_embeddings(
        model: Esm2LlamaInstructForCausalLM,
        description_input_ids: torch.Tensor,
        description_attention_mask: torch.Tensor,
        output_llama_layer: int = 16,
) -> torch.Tensor:
    """Take output corresponding to eot_token in the description. """
    llama_model: LlamaModel = model.llama_decoder.model
    hidden_states = llama_model(
        input_ids=description_input_ids,
        attention_mask=description_attention_mask,
        use_cache=False, 
        output_attentions=False, 
        output_hidden_states=True, 
        return_dict=False,
    )[1]  # (bsz, max_desc_len, hidden_dim)

    return readout_embeddings(
        embeddings=hidden_states[output_llama_layer],
        attention_mask=description_attention_mask,
        readout_fn="mix"
    )  # (bsz, decoder_hidden_size)


def teacher_forcing_forward_pass(
        rank: int,
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        data_batch: Dict[str, Any],
        contrastive_num_segments: int, 
) -> torch.Tensor:  # loss
    """
    Standard API for different models. Used in both `train_epoch` and `eval_epoch`.
    Prepare inputs from dataloader, migrate variable to the same device as the model, 
    and execute the forward pass with teacher forcing.

    Returned loss is not scaled with gradient accumulation steps.

    Due to the memory limit on GPUs, the similarity matrix will be computed in 
    segments, and the loss will be averaged over the segments.
    """
    protein_input_ids = data_batch["protein_input_ids"].to(rank)
    protein_attention_mask = data_batch["protein_attention_mask"].to(rank)
    description_input_ids = data_batch["description_input_ids"].to(rank)
    description_attention_mask = data_batch["description_attention_mask"].to(rank)

    base_model = model
    if isinstance(model, DistributedDataParallel):
        base_model = model.module

    batch_size = protein_input_ids.size(0)
    segment_size = batch_size // contrastive_num_segments
    if segment_size * contrastive_num_segments != batch_size:
        print(
            "WARNING: Given batch size is not divisible by the number of segments "
            "for contrastive learning."
        )
    
    acc_loss = torch.zeros([]).to(rank)
    loss_fn = SegmentedBatchInfoNCELoss()

    with torch.no_grad():  # WARNING: llama decoder fixed
        description_output = get_description_embeddings(
            base_model,  
            description_input_ids, 
            description_attention_mask
        )
        description_output = torch.nn.functional.normalize(description_output, p=2, dim=-1)

    for segment_id in range(contrastive_num_segments):
        segment_protein_input_ids = protein_input_ids[
            segment_id * segment_size:(segment_id + 1) * segment_size
        ]
        segment_protein_attention_mask = protein_attention_mask[
            segment_id * segment_size:(segment_id + 1) * segment_size
        ]

        segment_protein_output = get_sequence_embeddings(
            base_model, 
            segment_protein_input_ids, 
            segment_protein_attention_mask
        )
        labels = torch.arange(
            segment_id * segment_size, 
            (segment_id + 1) * segment_size, 
            device=rank
        )

        acc_loss += loss_fn(
            segment_output1=segment_protein_output, 
            batch_output2=description_output, 
            labels=labels
        )

    return acc_loss / contrastive_num_segments


def setup(rank: int, world_size: int):
    """
    Initialize processes for distributed training before first epoch. 
    Fetch from job script or launcher to set the IP address and the port of the 
    master node. 
    """
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '9901')
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    """End processes for distributed training after last epoch"""
    dist.destroy_process_group()


def train_epoch(
        rank: int,
        current_epoch: int,
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        dataloader: Prot2TextInstructDataLoader,
        optimizer: Optimizer,
        args: Dict[str, Any]
):
    """Iterate over all batches for one epoch in training with teacher forcing"""
    model.train()
    ddp_loss = torch.zeros(2).to(rank)  
        # [0] for acc. loss and [1] for num. of seen batches
    ddp_gradnorm = torch.zeros(2).to(rank)  
        # [0] for acc. gradnorm and [1] for num. of passed steps
    optimizer.zero_grad()  # erase accumulated gradients from last epoch

    t = tqdm(iter(dataloader))
    for batch_idx, data_batch in enumerate(t):
        # with autocast, logits will be in AUTOCAST_DTYPE 
        # but loss will be re-casted to torch.float32
        # and model weights will stay in torch.float32
        loss = teacher_forcing_forward_pass(
            rank=rank, 
            model=model, 
            data_batch=data_batch, 
            contrastive_num_segments=args["contrastive_num_segments"]
        )

        # rescale loss for consistency with different gradient accumulation steps
        loss = loss / args["gradient_accumulation_steps"]

        # summary current batch
        t.set_postfix({
            "mode": "train",
            "epoch": f"{current_epoch}/{args['num_epochs']}",
            "batch_loss": loss.item() * args["gradient_accumulation_steps"],
            "device": f"rank:{rank}"
        })
        ddp_loss[0] += loss.item() * args["gradient_accumulation_steps"]
        ddp_loss[1] += 1  # the loss is the weighted mean of the output of every batch

        # scale the loss up by a large factor to prevent them from becoming too small
        # then accumulate the scaled grads
        loss.backward()  
            # backward out of autocast, but still uses same dtype as for forward

        # update weights by loss if accumulation step is reached
        if (batch_idx + 1) % args["gradient_accumulation_steps"] == 0: 
            gradnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    float("inf") 
                    if args["gradient_clipping"] is None 
                    else args["gradient_clipping"]
                )
            )
            ddp_gradnorm[0] += gradnorm
            ddp_gradnorm[1] += 1

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # summary current epoch
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            f"[epoch={current_epoch}/{args['num_epochs']}, "
            f"train_loss={ddp_loss[0] / ddp_loss[1]}, "
            f"epoch_lr={optimizer.param_groups[0]['lr']}, "
            f"epoch_gradnorm={ddp_gradnorm[0] / ddp_gradnorm[1]}]"
        )
        # NaN detection
        if ddp_loss[0] != ddp_loss[0]:
            raise ValueError(
                "NaN detected in the training loss of the epoch, training interrupted."
            )


def eval_epoch(
        rank: int,
        current_epoch: int, 
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        dataloader: Prot2TextInstructDataLoader,
        args: Dict[str, Any]
):
    """Iterate over all batches in evaluation with teacher forcing"""
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)  
        # [0] for acc. loss and [1] for num. of seen batches

    t = tqdm(iter(dataloader))
    for data_batch in t:
        with torch.no_grad():
            loss = teacher_forcing_forward_pass(
                rank=rank,
                model=model,
                data_batch=data_batch,
                contrastive_num_segments=args["contrastive_num_segments"]
            )

            t.set_postfix({
                "mode": "eval",
                "epoch": f"{current_epoch}/{args['num_epochs']}",
                "batch_loss": loss.item(),
                "device": f"rank:{rank}"
            })
            ddp_loss[0] += loss.item()
            ddp_loss[1] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            f"[epoch={current_epoch}/{args['num_epochs']}, "
            f"eval_loss={ddp_loss[0] / ddp_loss[1]}]"
        )


def train_on_device(
        rank: int,
        world_size: int,
        args: Dict[str, Any]
):
    """
    Training and evaluation process for each device, including epochs of training 
    with teacher forcing. 
    """
    setup(rank, world_size)

    # prepare datasets and dataloaders
    esm_tokenizer = AutoTokenizer.from_pretrained(args["esm_path"])
    llama_tokenizer = AutoTokenizer.from_pretrained(
        args["llama_path"], 
        pad_token='<|reserved_special_token_0|>'
    )

    train_dataset = Prot2TextInstructDataset(
        root_dir=os.path.join(args["root_dataset_dir"], f"{args['train_split']}"),
        csv_path=os.path.join(args["root_csv_dir"], f"{args['train_split']}.csv"),
        sequence_tokenizer=esm_tokenizer,
        description_tokenizer=llama_tokenizer,
        skip_reload=True,
        skip_download=True,
        ignore_graph_features=True,
    )
    if args["debug_trim_train_split"]:
        train_dataset.usable_file_names = train_dataset.usable_file_names[
            :args["debug_trim_train_split"]
        ]
    train_sampler = DistributedSampler(
        train_dataset, 
        rank=rank, 
        num_replicas=world_size, 
        shuffle=True
        )
    train_loader = Prot2TextInstructDataLoader(
        train_dataset,
        batch_size=args["batch_size_per_device"],
        sampler=train_sampler,
        num_workers=4,  # parallel CPU cores used for data loading
        pin_memory=True,  # enable page-locked memory allocation for faster data transfer to GPUs
        shuffle=False,  # avoid shuffling twice with DistributedSampler
        drop_last=True,  # avoid incomplete batch at the end
    )
    print(f"Train dataset loaded on rank:{rank}")

    eval_dataset = Prot2TextInstructDataset(
        root_dir=os.path.join(args["root_dataset_dir"], f"{args['eval_split']}"),
        csv_path=os.path.join(args["root_csv_dir"], f"{args['eval_split']}.csv"),
        sequence_tokenizer=esm_tokenizer,
        description_tokenizer=llama_tokenizer,
        skip_reload=True,
        skip_download=True,
        ignore_graph_features=True,
    )
    if args["debug_trim_eval_split"]:
        eval_dataset.usable_file_names = eval_dataset.usable_file_names[
            :args["debug_trim_eval_split"]
        ]
    eval_sampler = DistributedSampler(
        eval_dataset, 
        rank=rank, 
        num_replicas=world_size, 
        shuffle=False
    )
    eval_loader = Prot2TextInstructDataLoader(
        eval_dataset,
        batch_size=args["batch_size_per_device"],
        sampler=eval_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    print(f"Eval dataset loaded on rank:{rank}")

    torch.cuda.set_device(rank)

    model = load_model(args=args)
    model = model.to(rank)

    model = DistributedDataParallel(
        model, 
        find_unused_parameters=True  # suppress error for unused parameters in wrapped model
    )
    print(f"DDP model loaded on rank:{rank}")

    # initialization of the optimizer after wrapping the model
    optimizer = Adam(model.parameters(), lr=args["learning_rate"])
    scheduler = StepLR(optimizer, step_size=1, gamma=args["scheduler_gamma"])
    if args["load_optimizer_scheduler_checkpoint_path"]:
        print(f"Loading {args['load_optimizer_scheduler_checkpoint_path']}")
        checkpoint_state_dicts = torch.load(
            args["load_optimizer_scheduler_checkpoint_path"], 
            weights_only=True
        )
        optimizer_state_dict = checkpoint_state_dicts["optimizer_state_dict"]
        scheduler_state_dict = checkpoint_state_dicts["scheduler_state_dict"]
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)

    # core loop of epochs
    for epoch_idx in range(1, args["num_epochs"] + 1):
        # shuffle data differently at each epoch across all processes
        train_sampler.set_epoch(epoch=epoch_idx)

        train_epoch(
            rank=rank,
            current_epoch=epoch_idx,
            model=model,    
            dataloader=train_loader,
            optimizer=optimizer,
            args=args
        )
        scheduler.step()
        dist.barrier()  # use a barrier to make sure training is done on all ranks
        
        eval_epoch(
            rank=rank,
            model=model,
            current_epoch=epoch_idx,
            dataloader=eval_loader,
            args=args
        )
        dist.barrier()

        if (
            epoch_idx == 1 
            or epoch_idx == args["num_epochs"] 
            or epoch_idx % args["save_every_epochs"] == 0
        ):
            model_state_dict = model.module.state_dict()
            if rank == 0:
                model_checkpoint_path = os.path.join(
                    args["save_checkpoint_dir"], 
                    f"model_checkpoint_{epoch_idx}.pt"
                )
                torch.save(model_state_dict, model_checkpoint_path)
                print(f"Saving {model_checkpoint_path}")

                optimizer_scheduler_checkpoint_path = os.path.join(
                    args["save_checkpoint_dir"], 
                    f"optimizer_scheduler_checkpoint_{epoch_idx}.pt"
                )
                torch.save(
                    {
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }, 
                    optimizer_scheduler_checkpoint_path
                )
                print(f"Saving {optimizer_scheduler_checkpoint_path}")

            dist.barrier()

    cleanup()


def train_distributed(
        args: Dict[str, Any]  # replace **kwargs for compatibility with spawn
):
    """
    Core training process across multiple devices with epochs of training and 
    inter-epoch evaluation.
    """
    torch.multiprocessing.spawn(
        train_on_device, 
        args=(args["world_size"], args),
        nprocs=args["world_size"],
        join=True
    )


if __name__ == '__main__':
    # suppress messages from AutoTokenizer parallelism and Graphein respectively
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["LOGURU_LEVEL"] = "INFO"

    parsed_args = argParser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # restrict GPU visibility
    parsed_args.world_size = torch.cuda.device_count()  # use up all visible GPUs

    torch.manual_seed(parsed_args.random_seed)
    torch.cuda.manual_seed(parsed_args.random_seed)
    
    # initialize checkpoint directory
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    parsed_args.save_checkpoint_dir = os.path.join(
        parsed_args.save_checkpoint_dir, 
        f"checkpoints_{timestamp}"
    )
    if not os.path.exists(parsed_args.save_checkpoint_dir):
        os.mkdir(parsed_args.save_checkpoint_dir)
    
    print("####################")
    for key, value in parsed_args.__dict__.items(): 
        print(f"{key}: {value}")
    print("####################")

    train_distributed(parsed_args.__dict__)
