from tqdm.autonotebook import tqdm
import torch
from tts.logger.wandb import WanDBWriter


def train_epoch(model, optimizer, loader, scheduler, loss_fn, config, featurizer, aligner, logger: WanDBWriter, vocoder=None):
    model.train()

    for i, batch in enumerate(tqdm(iter(loader))):
        logger.set_step(logger.step + 1, mode='train')
        batch = batch.to(config['device'])
        batch.melspec = featurizer(batch.waveform)

        batch.melspec_length = batch.melspec.shape[-1] - batch.melspec.eq(-11.5129251)[:, 0, :].sum(dim=-1)
        batch.melspec_length = batch.melspec_length
        
        with torch.no_grad():
            durations = aligner(
                batch.waveform, batch.waveforn_length, batch.transcript
            ).to(config['device'])

            durations = durations * batch.melspec_length.unsqueeze(-1)

            batch.durations = durations

        optimizer.zero_grad()

        batch = model(batch)

        length_loss, melspec_loss = loss_fn(batch)

        loss = length_loss + melspec_loss

        loss.backward()
        optimizer.step()

        np_length_loss = length_loss.detach().cpu().numpy()
        np_melspec_loss = melspec_loss.detach().cpu().numpy()

        logger.add_scalar("melspec_loss", np_melspec_loss)
        logger.add_scalar("length_loss", np_length_loss)

        #   scaler.scale(loss).backward()

        if i % config['grad_accum_steps'] == 0:
            optimizer.step()

        if vocoder is not None and i % config['log_train_step'] == 0:
            reconstructed_wav = vocoder.inference(batch.melspec_prediction[0:1].transpose(-1, -2)).cpu()

            logger.add_text("text", batch.transcript[0])
            logger.add_audio("audio", reconstructed_wav, sample_rate=22050)
            
        if i > config['len_epoch']:
            break

        scheduler.step()


@torch.no_grad()
def evaluate(model, loader, config, vocoder, logger: WanDBWriter):
    model.eval()

    for batch in tqdm(iter(loader)):
        batch = batch.to(config['device'])

        batch = model(batch)

        for i in range(batch.melspec_prediction.shape[0]):
            logger.set_step(logger.step + 1, "val")

            reconstructed_wav = vocoder.inference(batch.melspec_prediction[i:i + 1].transpose(-1, -2)).cpu()

            logger.add_text("text", batch.transcript[i])
            logger.add_audio("audio", reconstructed_wav, sample_rate=22050)

