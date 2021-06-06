
import torch
from torchtext.legacy.data import BucketIterator
from load_data import get_dataset, split_data, _len_sort_key, save_vocab
from config import read_training_pipeline_params
from loguru import logger
import click
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, Trainer, TrainingArguments, EarlyStoppingCallback



SEED = 2021

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True




@click.command(name="main")
@click.argument("config_path")
def train_model(config_path: str):
    writer = SummaryWriter()
    config = read_training_pipeline_params(config_path)
    logger.info("pretrained_emb {b}", b=config.net_params.pretrained_emb)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device is {device}", device=device)
    SRC, TRG, dataset = get_dataset(config.dataset_path, False)
    train_data, valid_data, test_data = split_data(dataset, **config.split_ration.__dict__)
    SRC.build_vocab(train_data, min_freq=3)
    TRG.build_vocab(train_data, min_freq=3)
    torch.save(SRC.vocab, config.src_vocab_name)
    torch.save(TRG.vocab, config.trg_vocab_name)
    logger.info("Vocab saved")
    print(f"Unique tokens in source (ru) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=config.BATCH_SIZE,
        device=device,
        sort_key=_len_sort_key,
    )
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    config_encoder = BertConfig(vocab_size=INPUT_DIM)
    config_decoder = BertConfig(vocab_size=OUTPUT_DIM)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    model = EncoderDecoderModel(config=config)
    config_encoder = model.config.encoder
    config_decoder = model.config.decoder
    config_decoder.is_decoder = True
    config_decoder.add_cross_attention = True
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    model = EncoderDecoderModel(config=config)
    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
        eval_steps=500,
        place_model_on_device=device,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=10,
        save_steps=3000,
        seed=0,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    model.save_pretrained("bert2bert")
if __name__ == "__main__":
    train_model()
