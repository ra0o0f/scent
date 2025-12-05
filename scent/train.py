import os
from scent.models.scent_model import ScentEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scent.utils.logging import TrainLog
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

def move_batch_to_device(batch, device, graph_training=True):
    batch['input_ids'] = batch['input_ids'].to(device)
    batch['attention_mask'] = batch['attention_mask'].to(device)
    batch['labels'] = batch['labels'].to(device)
    batch['node_ids'] = batch['node_ids'].to(device)

    if graph_training:
        for key in ['node_num', 'edge_num', 'edge_index', 'edge_graph_idx', 
                    'n_id', 'e_id', 'graph_offsets']:
            if key in batch:
                batch[key] = batch[key].to(device)

        if 'node_feature_batch' in batch:
            batch['node_feature_batch']['input_ids'] = batch['node_feature_batch']['input_ids'].to(device)
            batch['node_feature_batch']['attention_mask'] = batch['node_feature_batch']['attention_mask'].to(device)
            
        if 'edge_feature_batch' in batch:
            batch['edge_feature_batch']['input_ids'] = batch['edge_feature_batch']['input_ids'].to(device)
            batch['edge_feature_batch']['attention_mask'] = batch['edge_feature_batch']['attention_mask'].to(device)
            
    return batch

def get_grouped_params(model, weight_decay):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

def run_evaluation(model, eval_dataloader, device, graph_training, epoch, trainlog):
    model.eval()
    num_batches = 0

    total_graph_loss = 0.0
    total_text_loss = 0.0
    total_mlm_loss = 0.0
    total_entity_loss = 0.0
    
    eval_pbar = tqdm(eval_dataloader, desc="running validation", leave=False)
    
    with torch.no_grad():
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for batch in eval_pbar:
                batch = move_batch_to_device(batch, device, graph_training)

                if graph_training:
                    graph_out = model.forward_graph(batch)
                    graph_loss = graph_out['graph_loss']

                total_graph_loss += graph_loss.item()

                text_out = model.forward_text(batch)

                total_mlm_loss += text_out['mlm_loss'].item()
                total_entity_loss += text_out['entity_loss'].item()
                total_text_loss += text_out['text_loss'].item()

                num_batches += 1

    avg_graph_loss = total_graph_loss / num_batches if num_batches > 0 else 0.0
    avg_mlm_loss = total_mlm_loss / num_batches if num_batches > 0 else 0.0
    avg_entity_loss = total_entity_loss / num_batches if num_batches > 0 else 0.0
    avg_text_loss = total_text_loss / num_batches if num_batches > 0 else 0.0
    
    trainlog.track(avg_graph_loss, 'val/graph_loss', force_step=epoch)
    trainlog.track(avg_mlm_loss, 'val/mlm_loss', force_step=epoch)
    trainlog.track(avg_entity_loss, 'val/entity_loss', force_step=epoch)
    trainlog.track(avg_text_loss, 'val/text_loss', force_step=epoch)

    val_loss = {
        'text_loss': avg_text_loss, 
        'mlm_loss': avg_mlm_loss, 
        'entity_loss': avg_entity_loss
    }
    if graph_training:
        val_loss['graph_loss'] = avg_graph_loss

    return val_loss


def train(exp_name, scent_encoder, dataloader, collate_fn, test_dataset, graph_training):

    trainlog = TrainLog(disabled=False, project=f'{exp_name}', batch_log=True, log_step=15)

    NUM_EPOCHS = 3
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 4
    MAX_GRAD_NORM = 1.0
    WARMUP_RATIO = 0.05

    RESUME_FROM_CHECKPOINT = ''

    device = torch.device('cuda')
    SAVE_DIR = f"scent_checkpoints/{exp_name}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    scent_encoder = scent_encoder.to(device)

    #optimizer_type = 'schedulefree'
    optimizer_type = 'nonschedulefree'

    total_steps = (len(dataloader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    grouped_params = get_grouped_params(scent_encoder, WEIGHT_DECAY)

    optimizer = torch.optim.AdamW(grouped_params, lr=LEARNING_RATE, betas=(0.9, 0.999))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # optimizer = schedulefree.AdamWScheduleFree(
    #     grouped_params, 
    #     lr=LEARNING_RATE, 
    #     warmup_steps=warmup_steps
    # )
    # scheduler = {}

    # C. Scaler
    scaler = torch.amp.GradScaler('cuda')


    start_epoch = 0
    global_step = 0
    resume_step_in_epoch = 0
    batches_per_epoch = len(dataloader)

    if RESUME_FROM_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT):
        print(f"resuming training: {RESUME_FROM_CHECKPOINT}")
        checkpoint = torch.load(RESUME_FROM_CHECKPOINT, map_location=device)
        
        scent_encoder.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if optimizer_type != 'schedulefree':
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        
        total_batches_processed = global_step * GRADIENT_ACCUMULATION_STEPS
        
        resume_step_in_epoch = total_batches_processed % batches_per_epoch
        
        print(f"resumed at epoch {start_epoch}, global step {global_step}. skipping first {resume_step_in_epoch} batches.")

    print(f"starting training | epochs: {NUM_EPOCHS} | steps: {total_steps} | accumulation: {GRADIENT_ACCUMULATION_STEPS}")

    global_step = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        
        scent_encoder.train()

        if optimizer_type == 'schedulefree':
            optimizer.train()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
        
        for i, batch in enumerate(progress_bar):

            if epoch == start_epoch and i < resume_step_in_epoch:
                continue

            trainlog.start_step(global_step=epoch * batches_per_epoch + i)

            
            batch = move_batch_to_device(batch, device, graph_training=graph_training)

            if graph_training:

                scent_encoder.peft_roberta.set_adapter("graph_adapter")
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    graph_out = scent_encoder.forward_graph(batch)
                    graph_loss = graph_out['graph_loss']

                scaler.scale(graph_loss / GRADIENT_ACCUMULATION_STEPS).backward()

            scent_encoder.peft_roberta.set_adapter("text_adapter")
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                text_out = scent_encoder.forward_text(batch)
                text_loss = text_out['text_loss']
                mlm_loss = text_out['mlm_loss']
                entity_loss = text_out['entity_loss']

            scaler.scale(text_loss / GRADIENT_ACCUMULATION_STEPS).backward()
            
            with torch.no_grad():
                text_loss_am = trainlog.add_averagemeter('text_loss', text_loss.item())
                mlm_loss_am = trainlog.add_averagemeter('mlm_loss', mlm_loss.item())
                entity_loss_am = trainlog.add_averagemeter('entity_loss', entity_loss.item())

                if graph_training:
                    graph_loss_am = trainlog.add_averagemeter('graph_loss', graph_loss.item())

            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                
                scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(scent_encoder.parameters(), MAX_GRAD_NORM)
                
                scaler.step(optimizer)
                scaler.update()

                if optimizer_type != 'schedulefree':
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = LEARNING_RATE
                
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                progress_info = {
                    'lr': f"{current_lr:.6f}",
                    'text_loss': text_loss_am.avg,
                    'mlm_loss': mlm_loss_am.avg,
                    'entity_loss': entity_loss_am.avg   
                }

                if graph_training:
                    progress_info['graph_loss'] = graph_loss_am.avg
                
                progress_bar.set_postfix(progress_info)
                
                if global_step > 0 and global_step % 500 == 0:
                    ckpt_path = os.path.join(SAVE_DIR, f"checkpoint-{global_step}.pt")
                    # torch.save(scent_encoder.state_dict(), ckpt_path)


                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': scent_encoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if optimizer_type != 'schedulefree' else {},
                        'scaler_state_dict': scaler.state_dict(),
                    }, ckpt_path)

                    
            trainlog.end_step()
        
        print(f"\nepoch {epoch+1} finished. starting evaluation...")
        
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=16, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=4
        )
        
        if optimizer_type == 'schedulefree':
            optimizer.eval() 
        val_loss = run_evaluation(scent_encoder, test_dataloader, device, graph_training=graph_training, epoch=epoch, trainlog=trainlog)
        if optimizer_type == 'schedulefree':
            optimizer.train() 

        print(f"epoch:{epoch+1} text_loss: {val_loss['text_loss']:.4f}, mlm_loss: {val_loss['mlm_loss']:.4f}, entity_loss: {val_loss['entity_loss']:.4f}")
        
        if graph_training:
            print(f"graph_loss: {val_loss['graph_loss']:.4f}")

        epoch_save_path = os.path.join(SAVE_DIR, f"scent_epoch_{epoch+1}.pt")
        #torch.save(scent_encoder.state_dict(), epoch_save_path)
        torch.save({
            'epoch': epoch + 1, # Save as next epoch start
            'global_step': global_step,
            'model_state_dict': scent_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if optimizer_type != 'schedulefree' else {},
            'scaler_state_dict': scaler.state_dict(),
        }, epoch_save_path)
        print(f"saved checkpoint to {epoch_save_path}\n")

    print("training complete.")