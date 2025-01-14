import sys
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from machine_translation import MachineTranslationModel
from machine_translation.data import MachineTranslationDataModule
from attention_smithy.numeric_embeddings import SinusoidalPositionEmbedding, NumericEmbeddingFacade
from attention_smithy.components import MultiheadAttention, FeedForwardNetwork
from attention_smithy.attention import StandardAttentionMethod
from attention_smithy.utils import seed_everything
from attention_smithy.generators import GeneratorContext
from transformers import AutoTokenizer
from sacrebleu.metrics import BLEU

def train_model(
        loss_type: str,
        label_smoothing: float,
        scheduler_warmup_steps: int,
        maximum_length: int,
        batch_size: int,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        number_of_layers: int,
        dropout: float,
        random_seed: int,
        num_training_samples: int,
):
    seed_everything(random_seed)
    torch.set_float32_matmul_precision('medium')

    data_module = MachineTranslationDataModule(
        en_filepath_suffix='_en.txt',
        de_filepath_suffix='_de.txt',
        maximum_length=maximum_length,
        batch_size=batch_size,
        num_training_samples=num_training_samples,
    )
    data_module.setup()

    run_name_prefix = f'{loss_type}-{label_smoothing}-{embed_dim}-{dim_feedforward}-{number_of_layers}-{scheduler_warmup_steps}__allLines'
    logger = WandbLogger(project=f'mt-embed_dim_{embed_dim}-dim_ff_{dim_feedforward}-num_layers_{number_of_layers}-num_samples_{num_training_samples}', name=run_name_prefix)

    train_loss_checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/",
        every_n_train_steps=10_000,
        #every_n_epochs=1,
        filename=run_name_prefix + "-{epoch:02d}-{step:08d}",
    )

    trainer = pl.Trainer(
        max_epochs=40,
        logger=logger,
        callbacks=[
            train_loss_checkpoint_callback,
            ValidateAtCheckpoints(list(range(0, 100_000, 1_000))[1:]),
        ],
        log_every_n_steps=500,
        #strategy='ddp',
        #devices = 1,
        #use_distributed_sampler=False,
    )

    sinusoidal_position_embedding = SinusoidalPositionEmbedding(embed_dim)
    numeric_embedding_facade = NumericEmbeddingFacade(sinusoidal_position=sinusoidal_position_embedding)
    generic_attention = MultiheadAttention(embedding_dimension = embed_dim, number_of_heads = num_heads, attention_method = StandardAttentionMethod(dropout))
    decoder_self_attention = MultiheadAttention(embedding_dimension = embed_dim, number_of_heads = num_heads, attention_method = StandardAttentionMethod(dropout, is_causal_masking=True))
    feedforward_network = FeedForwardNetwork(embed_dim, dim_feedforward, 'relu', dropout)

    model = MachineTranslationModel(
        src_vocab_size=data_module.de_vocab_size,
        tgt_vocab_size=data_module.en_vocab_size,
        encoder_self_attention=generic_attention,
        decoder_self_attention=decoder_self_attention,
        decoder_cross_attention=generic_attention,
        feedforward_network=feedforward_network,
        numeric_embedding_facade=numeric_embedding_facade,
        tgt_padding_token=data_module.en_pad_token,
        embedding_dimension=embed_dim,
        num_encoder_layers=number_of_layers,
        num_decoder_layers=number_of_layers,
        scheduler_warmup_steps = scheduler_warmup_steps,
        loss_type= loss_type,
        label_smoothing = label_smoothing,
    )

    trainer.fit(model, data_module)
    torch.save(model, 'model.pth')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--loss_type', type=str, required=True, help='Type of loss function')
    parser.add_argument('--label_smoothing', type=float, required=True, help='Label smoothing value')
    parser.add_argument('--scheduler_warmup_steps', type=int, default=4000, help='Number of warmup steps for scheduler')
    parser.add_argument('--maximum_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--embed_dim', type=int, help='Embedding dimension. Original model used 512')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads. Original model used 8')
    parser.add_argument('--dim_feedforward', type=int, help='Feedforward dimension. Original model used 2048.')
    parser.add_argument('--number_of_layers', type=int, help='Number of layers. Original model used 6')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_training_samples', type=int, default=None, help='number of training samples to use. Can reduce to speed up program at cost of accuracy.')
    return parser.parse_args()

class ValidateAtCheckpoints(pl.Callback):
    def __init__(self, checkpoints):
        self.checkpoints = checkpoints
        self.generator = GeneratorContext(method='beam_batch')
        self.de_tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
        self.en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def on_train_batch_end(self, trainer, pl_module, outputs, train_batch, batch_idx, **kwargs):
        end_token = self.en_tokenizer.convert_tokens_to_ids(self.en_tokenizer.sep_token)
        start_token = self.en_tokenizer.convert_tokens_to_ids(self.en_tokenizer.cls_token)
        if batch_idx in self.checkpoints:
            reference_translations = []
            output_translations = []
            with torch.no_grad():
                count = 0
                max_num_batches = 10_000
                for batch in trainer.val_dataloaders:
                    if count > max_num_batches:
                        break
                    count += 1
                    batch = tuple(x.to(pl_module.device) for x in batch)
                    src_input_tensor, tgt_input_tensor, expected_output_tensor, src_padding_mask, tgt_padding_mask = batch
                    output_tensor = torch.cat((tgt_input_tensor, expected_output_tensor[:, -1:]), dim=1)
                    batch_size = src_input_tensor.shape[0]
                    tgt_starting_input = torch.full((batch_size, 1), start_token).to(pl_module.device)
                    src_encoded = pl_module.forward_encode(src_input_tensor.to(pl_module.device),
                                                           src_padding_mask=None)
                    generated_batch_tensor = self.generator.generate_sequence(pl_module,
                                                                        end_token,
                                                                        tgt_starting_input,
                                                                        src_encoded=src_encoded,
                                                                        src_padding_mask=src_padding_mask,
                                                                        tgt_padding_mask=None,
                                                                        )
                    print(f'batch number: {count}')
                    for i in range(batch_size):
                        generated_tgt_tensor = list(generated_batch_tensor[i])
                        try:
                            end_token_index = generated_tgt_tensor.index(end_token) + 1
                        except ValueError:
                            end_token_index = len(generated_tgt_tensor) + 1
                        generated_tgt_tensor = generated_tgt_tensor[:end_token_index]
                        expected_tgt_tensor = output_tensor[i]
                        reference_tokens = [int(x) for x in expected_tgt_tensor]
                        output_tokens = [int(x) for x in generated_tgt_tensor]
                        reference_translation = self.en_tokenizer.decode(reference_tokens, skip_special_tokens=True)
                        output_translation = self.en_tokenizer.decode(output_tokens, skip_special_tokens=True)
                        print(i)
                        print(reference_tokens)
                        print(output_tokens)
                        print(reference_translation)
                        print(output_translation)
                        reference_translations.append(reference_translation)
                        output_translations.append(output_translation)
            bleu_score = BLEU().corpus_score(output_translations, [reference_translations])
            pl_module.log("bleu", bleu_score.score, prog_bar=False, batch_size=batch_size)



if __name__ == "__main__":
    args = parse_args()
    print(args)
    train_model(
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        maximum_length=args.maximum_length,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        number_of_layers=args.number_of_layers,
        dropout=args.dropout,
        random_seed=args.random_seed,
        num_training_samples=args.num_training_samples,
    )
