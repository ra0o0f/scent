import os

import schedulefree
from models.scent_model import ScentEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.utils import get_lm_model, move_batch_to_device
from data.scent_data import (
    END_OF_TITLE_TOKEN,
    MENTION_START_TOKEN,
    NODE_END_TOKEN,
    NODE_START_TOKEN,
    ScentDataCollator,
    ScentDataset,
)
from data.yago import (
    load_graph_as_dataframe,
    load_pyg_data,
    parse_aida_dataset,
    postproces_pyg_data,
    prepare_graph_indices,
)
from data.wiki import load_yagograph_with_wikidata, read_wikidata
from utils.logging import TrainLog
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


def get_grouped_params(model, weight_decay):

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def run_evaluation(model, trainlog, eval_dataloader, device, graph_training, epoch):

    model.eval()
    num_batches = 0

    total_graph_loss = 0.0
    total_text_loss = 0.0
    total_mlm_loss = 0.0
    total_entity_loss = 0.0

    eval_pbar = tqdm(eval_dataloader, desc="Running Validation..", leave=False)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for batch in eval_pbar:
                batch = move_batch_to_device(batch, device, graph_training)

                if graph_training:
                    graph_out = model.forward_graph(batch)
                    graph_loss = graph_out["graph_loss"]

                total_graph_loss += graph_loss.item()

                text_out = model.forward_text(batch)

                total_mlm_loss += text_out["mlm_loss"].item()
                total_entity_loss += text_out["entity_loss"].item()
                total_text_loss += text_out["text_loss"].item()

                num_batches += 1

    avg_graph_loss = total_graph_loss / num_batches if num_batches > 0 else 0.0
    avg_mlm_loss = total_mlm_loss / num_batches if num_batches > 0 else 0.0
    avg_entity_loss = total_entity_loss / num_batches if num_batches > 0 else 0.0
    avg_text_loss = total_text_loss / num_batches if num_batches > 0 else 0.0

    trainlog.track(avg_graph_loss, "val/graph_loss", force_step=epoch)
    trainlog.track(avg_mlm_loss, "val/mlm_loss", force_step=epoch)
    trainlog.track(avg_entity_loss, "val/entity_loss", force_step=epoch)
    trainlog.track(avg_text_loss, "val/text_loss", force_step=epoch)

    val_loss = {
        "text_loss": avg_text_loss,
        "mlm_loss": avg_mlm_loss,
        "entity_loss": avg_entity_loss,
    }
    if graph_training:
        val_loss["graph_loss"] = avg_graph_loss

    return val_loss


def train(args, config):

    graph_training = args.graph_training

    ### prepare data

    train_set, testa_set, testb_set = parse_aida_dataset(
        config["data"]["aida_yago2_path"]
    )
    df_nodes, df_edges, df_rels = load_graph_as_dataframe(
        dataset_dir=config["data"]["dataset_path"],
        node_path=config["data"]["node_path"],
        edge_path=config["data"]["edge_path"],
        rel_path=config["data"]["rel_path"],
    )
    pyg_data = load_pyg_data(
        dataset_dir=config["data"]["dataset_path"],
        df_nodes=df_nodes,
        pyg_path=config["data"]["pyg_path"],
    )

    postproces_pyg_data(pyg_data)

    node_to_idx, idx_to_node, rel_to_idx, idx_to_rel = prepare_graph_indices(
        df_nodes=df_nodes, df_rels=df_rels
    )

    wikidata = read_wikidata(
        os.path.join(
            config["data"]["dataset_path"], config["data"]["aida_wikidata_path"]
        )
    )
    yagograph_wikidata = load_yagograph_with_wikidata(
        os.path.join(
            config["data"]["dataset_path"], config["data"]["graph_yago_wiki_path"]
        )
    )

    ### model and dataset

    lm_model, tokenizer, lm_config = get_lm_model(config["model"]["lm_model_name"])

    # make sure the num_entity_embeddings inside scent encoder is updated accordingly
    special_tokens = {
        "additional_special_tokens": [
            NODE_START_TOKEN,
            NODE_END_TOKEN,
            END_OF_TITLE_TOKEN,
            MENTION_START_TOKEN,
        ]
    }
    tokenizer.add_special_tokens(special_tokens)

    mask_buffer_len = config["model"]["mask_buffer_len"]
    max_seq_len = config["model"]["max_seq_len"]
    graph_num_neighbors = [10, 3]

    batch_size = config["train"]["batch_size"]
    num_workers = config["train"]["num_workers"]
    device = torch.device(config["train"]["device"])

    whitening_stats = torch.load(
        os.path.join(
            config["data"]["dataset_path"], config["data"]["aida_whitening_params_path"]
        )
    )

    dataset = ScentDataset(
        aida_data=train_set,
        node_to_idx=node_to_idx,
        tokenizer=tokenizer,
        mask_buffer_len=mask_buffer_len,
        max_seq_len=max_seq_len,
    )

    collate_fn = ScentDataCollator(
        tokenizer,
        sample_graph_nodes=graph_training,
        pyg_data=pyg_data,
        idx_to_node=idx_to_node,
        idx_to_rel=idx_to_rel,
        graph_num_neighbors=graph_num_neighbors,
        yagograph_wikidata=yagograph_wikidata,
        wikidata=wikidata,  # AIDA wikidata
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    scent_encoder = ScentEncoder(
        config=lm_config,
        lm_model=lm_model,
        tokenizer=tokenizer,
        graph_training=graph_training,
        whitening_stats=whitening_stats,
    ).to(device)

    ### train

    exp_name = config["train"]["exp_name"]

    trainlog = TrainLog(
        disabled=False, project=f"{exp_name}", batch_log=True, log_step=15
    )

    NUM_EPOCHS = config["train"]["num_epochs"]
    LEARNING_RATE = config["train"]["lr"]
    WEIGHT_DECAY = config["train"]["wd"]
    GRADIENT_ACCUMULATION_STEPS = config["train"]["grad_accumulation_steps"]
    MAX_GRAD_NORM = config["train"]["max_grad_norm"]
    WARMUP_RATIO = config["train"]["warmup_ratio"]

    RESUME_FROM_CHECKPOINT = config["train"]["resume_from_checkpoint"]

    optimizer_type = config["train"]["optimizer_type"]

    SAVE_DIR = f"scent_checkpoints/{exp_name}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    total_steps = (len(dataloader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    grouped_params = get_grouped_params(scent_encoder, WEIGHT_DECAY)

    if optimizer_type == "nonschedulefree":
        optimizer = torch.optim.AdamW(
            grouped_params, lr=LEARNING_RATE, betas=(0.9, 0.999)
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    elif optimizer_type == "schedulefree":
        optimizer = schedulefree.AdamWScheduleFree(
            grouped_params, lr=LEARNING_RATE, warmup_steps=warmup_steps
        )
        scheduler = {}

    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    global_step = 0
    resume_step_in_epoch = 0
    batches_per_epoch = len(dataloader)

    ### resume

    if RESUME_FROM_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT):
        print(f"Resuming training from {RESUME_FROM_CHECKPOINT}")
        checkpoint = torch.load(RESUME_FROM_CHECKPOINT, map_location=device)

        scent_encoder.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if optimizer_type != "schedulefree":
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]

        total_batches_processed = global_step * GRADIENT_ACCUMULATION_STEPS

        resume_step_in_epoch = total_batches_processed % batches_per_epoch

        print(
            f"Resumed at epoch {start_epoch}, global Step {global_step}. skipping first {resume_step_in_epoch} batches."
        )

    print(
        f"Starting training | epochs: {NUM_EPOCHS} | steps: {total_steps} | accumulation: {GRADIENT_ACCUMULATION_STEPS}"
    )

    global_step = 0

    for epoch in range(start_epoch, NUM_EPOCHS):

        scent_encoder.train()

        if optimizer_type == "schedulefree":
            optimizer.train()

        progress_bar = tqdm(
            dataloader, desc=f"epoch {epoch+1}/{NUM_EPOCHS}", leave=True
        )

        for i, batch in enumerate(progress_bar):

            if epoch == start_epoch and i < resume_step_in_epoch:
                continue

            trainlog.start_step(global_step=epoch * batches_per_epoch + i)

            batch = move_batch_to_device(batch, device, graph_training=graph_training)

            if graph_training:

                scent_encoder.peft_roberta.set_adapter("graph_adapter")
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    graph_out = scent_encoder.forward_graph(batch)
                    graph_loss = graph_out["graph_loss"]

                scaler.scale(graph_loss / GRADIENT_ACCUMULATION_STEPS).backward()

            scent_encoder.peft_roberta.set_adapter("text_adapter")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                text_out = scent_encoder.forward_text(batch)
                text_loss = text_out["text_loss"]
                mlm_loss = text_out["mlm_loss"]
                entity_loss = text_out["entity_loss"]

            scaler.scale(text_loss / GRADIENT_ACCUMULATION_STEPS).backward()

            with torch.no_grad():
                text_loss_am = trainlog.add_averagemeter("text_loss", text_loss.item())
                mlm_loss_am = trainlog.add_averagemeter("mlm_loss", mlm_loss.item())
                entity_loss_am = trainlog.add_averagemeter(
                    "entity_loss", entity_loss.item()
                )

                if graph_training:
                    graph_loss_am = trainlog.add_averagemeter(
                        "graph_loss", graph_loss.item()
                    )

            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    scent_encoder.parameters(), MAX_GRAD_NORM
                )

                scaler.step(optimizer)
                scaler.update()

                if optimizer_type != "schedulefree":
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = LEARNING_RATE

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                progress_info = {
                    "lr": f"{current_lr:.6f}",
                    "text_loss": text_loss_am.avg,
                    "mlm_loss": mlm_loss_am.avg,
                    "entity_loss": entity_loss_am.avg,
                }

                if graph_training:
                    progress_info["graph_loss"] = graph_loss_am.avg

                progress_bar.set_postfix(progress_info)

                if global_step > 0 and global_step % 500 == 0:
                    ckpt_path = os.path.join(SAVE_DIR, f"checkpoint-{global_step}.pt")

                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model_state_dict": scent_encoder.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": (
                                scheduler.state_dict()
                                if optimizer_type != "schedulefree"
                                else {}
                            ),
                            "scaler_state_dict": scaler.state_dict(),
                        },
                        ckpt_path,
                    )

            trainlog.end_step()

    epoch_save_path = os.path.join(SAVE_DIR, f"scent_epoch_{epoch+1}.pt")
    torch.save(
        {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": scent_encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if optimizer_type != "schedulefree" else {}
            ),
            "scaler_state_dict": scaler.state_dict(),
        },
        epoch_save_path,
    )
    print(f"saved checkpoint to {epoch_save_path}\n")

    print("Training complete.")
