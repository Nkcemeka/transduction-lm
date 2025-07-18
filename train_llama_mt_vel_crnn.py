import torch
import torch.nn.functional as F
import time
import random
import librosa
import numpy as np
import soundfile
import matplotlib.pyplot as plt
from pathlib import Path
import torch.optim as optim
from data.maestro import MaestroMultiTask
from data.collate import collate_fn
from data.io import events_to_notes
from models.crnn import CRnn
from tqdm import tqdm
import museval
import argparse
import wandb
import os

from data.tokenizers import Tokenizer
from models.enc_dec import EncDecConfig, EncDecPos


def train(args):

    # Arguments
    # model_name = args.model_name

    # Default parameters
    device = "cuda"
    batch_size = 4
    # num_workers = 0
    num_workers = 32
    evaluate_step_frequency = 1000
    save_step_frequency = 10000
    training_steps = 300000
    #training_steps = 68000
    debug = False
    filename = Path(__file__).stem
    segment_seconds = 10.
    lr = 1e-4
    frames_num = 1001
    max_token_len = 1536
    wandb_log = True

    model_name = "AudioLlama"
    checkpoints_dir = Path("./checkpoints", filename, model_name)
    
    root = "/home/nkcemeka/Documents/Datasets/maestro-v3.0.0"

    if wandb_log:
        wandb.init(
            project="mini_piano_transcription",
            name=filename
        )

    tokenizer = Tokenizer()
    
    # Dataset
    train_dataset = MaestroMultiTask(
        root=root,
        split="train",
        segment_seconds=segment_seconds,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        task="velocity"
    )

    test_dataset = MaestroMultiTask(
        root=root,
        split="test",
        segment_seconds=segment_seconds,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        task="velocity"
    )

    # Sampler
    train_sampler = Sampler(dataset_size=len(train_dataset))
    eval_train_sampler = Sampler(dataset_size=len(train_dataset))
    eval_test_sampler = Sampler(dataset_size=len(test_dataset))

    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True
    )

    eval_train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=eval_train_sampler,
        collate_fn=collate_fn,
        num_workers=0, 
        pin_memory=True
    )

    eval_test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        sampler=eval_test_sampler,
        collate_fn=collate_fn,
        num_workers=0, 
        pin_memory=True
    )


    # adsf
    # Load checkpoint
    enc_model_name = "CRnn"
    checkpoint_path = Path("Note_pedal.pth")
    enc_model = get_model(enc_model_name)
    enc_model.load_state_dict2(torch.load(checkpoint_path)["model"])
    enc_model.to(device)

    config = EncDecConfig(
        block_size=max_token_len + 1, 
        vocab_size=tokenizer.vocab_size, 
        padded_vocab_size=tokenizer.vocab_size, 
        n_layer=6, 
        n_head=16, 
        n_embd=1024, 
        audio_n_embd=1536
    )

    model = EncDecPos(config)
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(list(enc_model.parameters()) + list(model.parameters()), lr=lr)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    tmp = []

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):
        
        audio = data["audio"].to(device)
        input_token = data["token"][:, 0 : -1].to(device)
        target_token = data["token"][:, 1 :].to(device)
        target_mask = data["mask"][:, 1 :].to(device)

        optimizer.zero_grad()

        enc_model.train()
        model.train()
        audio_emb = enc_model(audio)["onoffvel_emb_h"]
        logits, loss = model(audio_emb=audio_emb, idx=input_token, target=target_token, target_mask=target_mask)
        
        loss.backward()

        optimizer.step()

        # from IPython import embed; embed(using=False); os._exit(0)
        
        if step % evaluate_step_frequency == 0:
            print("Evaluating ...")
            train_loss = validate(enc_model, model, eval_train_dataloader)
            test_loss = validate(enc_model, model, eval_test_dataloader)
            print("--- step: {} ---".format(step))
            print("Train loss: {:.4f}".format(train_loss))
            print("Test loss: {:.4f}".format(test_loss))

            if wandb_log:
                wandb.log({
                    "train loss": train_loss,
                    "test loss": test_loss
                })

        # Save model
        if step % save_step_frequency == 0:
            checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))

            #
            checkpoint_path = Path(checkpoints_dir, "step={}_encoder.pth".format(step))
            torch.save(enc_model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))
        
        # tmp.extend(data["answer_tokens_num"])
        # from IPython import embed; embed(using=False); os._exit(0)

        if step == training_steps:
            break

        # if step == 1000:
        #     from IPython import embed; embed(using=False); os._exit(0)
        #     hist, bin_edges = np.histogram(tmp)


def get_model(model_name):
    if model_name == "CRnn":
        return CRnn()
    elif model_name == "CRnn2":
        from models.crnn2 import CRnn2
        return CRnn2()
    elif model_name == "CRnn3":
        from models.crnn3 import CRnn3
        return CRnn3()
    elif model_name == "CRnn3_onset_offset_vel":
        from models.crnn3_onset_offset_vel import CRnn3_onset_offset_vel
        return CRnn3_onset_offset_vel()
    elif model_name == "Note_pedal":
        pass
    elif model_name == "AudioLlamaQA":
        from models.audiollama_qa import AudioLlamaQA
    else:
        raise NotImplementedError


class Sampler:
    def __init__(self, dataset_size):
        self.indexes = list(range(dataset_size))
        random.shuffle(self.indexes)
        
    def __iter__(self):

        pointer = 0

        while True:

            if pointer == len(self.indexes):
                random.shuffle(self.indexes)
                pointer = 0
                
            # print(pointer)
            index = self.indexes[pointer]
            pointer += 1

            yield index


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


def play_audio(mixture, target):
    soundfile.write(file="tmp_mixture.wav", data=mixture[0].cpu().numpy().T, samplerate=44100)
    soundfile.write(file="tmp_target.wav", data=target[0].cpu().numpy().T, samplerate=44100)
    from IPython import embed; embed(using=False); os._exit(0)


def validate(enc_model, model, dataloader): 

    pred_ids = []
    target_ids = []
    device = next(model.parameters()).device
    losses = []

    for step, data in tqdm(enumerate(dataloader)):

        if step == 5:
            break

        audio = data["audio"].to(device)
        input_token = data["token"][:, 0 : -1].to(device)
        target_token = data["token"][:, 1 :].to(device)
        target_mask = data["mask"][:, 1 :].to(device)

        with torch.no_grad():
            enc_model.eval()
            audio_emb = enc_model(audio)["onoffvel_emb_h"]

        with torch.no_grad():
            model.eval()
            logits, loss = model(audio_emb=audio_emb, idx=input_token, target=target_token, target_mask=target_mask)

        losses.append(loss.item())

    return np.mean(losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default="AudioLlama")
    args = parser.parse_args()

    train(args)

