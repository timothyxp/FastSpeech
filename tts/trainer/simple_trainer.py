from tqdm.autonotebook import tqdm
import torch


def train_epoch(model, optimizer, loader, scheduler, loss_fn, config, featurizer, aligner, wandb=None):
    model.train()

    for i, batch in enumerate(tqdm(iter(loader))):
        batch = batch.to(config['device'])

        batch.melspec = featurizer(batch.waveform)

        batch.durations = aligner(
            batch.waveform, batch.waveforn_length, batch.transcript
        )

        optimizer.zero_grad()

        batch = model(batch)

        length_loss, melspec_loss = loss_fn(batch)

        loss = length_loss + melspec_loss

        loss.backward()
        optimizer.step()

        np_length_loss = length_loss.detach().cpu().numpy()
        np_melspec_loss = melspec_loss.detach().cpu().numpy()
        print(np_length_loss, np_melspec_loss)

        if config['use_wandb']:
            wandb.log({
                "train/melspec_loss": np_melspec_loss,
                "train/length_loss": np_length_loss
            })

        #   scaler.scale(loss).backward()

        if i % config['grad_accum_steps'] == 0:
            optimizer.step()

        if i > config['len_epoch']:
            break

    scheduler.step()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses = 0

    for batch in tqdm(iter(loader)):
     #   batch: Seq2SeqBatch = batch.to(device)
  #      batch = batch.generate_masks(device)

        logits = model(
            batch.lns_ids['de'],
            batch.lns_ids['en'][:-1, :],
            batch.src_mask, batch.tgt_mask,
            batch.src_padding_mask, batch.tgt_padding_mask, batch.src_padding_mask
        )

        # loss = loss_fn(
        #     logits.reshape(-1, logits.shape[-1]),
        #     batch.lns_ids['en'][1:, :].reshape(-1)
        # )

   #     losses += loss.detach().cpu().numpy()

    val_loss = losses / len(loader)

  #  blue = calc_blue(val_data)

    # if config['use_wandb']:
    #     wandb.log({
    #         "eval/loss": val_loss,
    #         "eval/blue": blue
    #     })
    #
    # return val_loss, blue
