#"""
import logging
import sys
import io
from datetime import datetime

class StreamToLogger(io.TextIOBase):
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

# Configure the logging module
logging.basicConfig(filename=f'outputs/logs/model_script_output_{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}.log', level=logging.INFO)

# Redirect stdout to the logger
stdout_logger = logging.getLogger('STDOUT')
sys.stdout = StreamToLogger(stdout_logger, logging.INFO)

# Redirect stderr to the logger
stderr_logger = logging.getLogger('STDERR')
sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

print(' '.join(sys.argv))
#"""

import argparse
import logging
import os
import sys
import time
import warnings
from IPython.utils import io
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from machine_translation import MachineTranslationModel
from machine_translation.data import MachineTranslationDataModule
from attention_smithy.numeric_embeddings import SinusoidalPositionEmbedding, LearnedPositionEmbedding, RotaryPositionEmbedding, ALiBiPositionEmbedding, NumericEmbeddingManager, NoAddEmbedding, PassthroughEmbedding
from attention_smithy.components import MultiheadAttention, FeedForwardNetwork
from attention_smithy.attention import StandardAttentionMethod
from attention_smithy.utils import seed_everything
from attention_smithy.generators import GeneratorContext
from transformers import AutoTokenizer
from sacrebleu.metrics import BLEU

def run_training_job(parsed_args):
    seed_everything(parsed_args.random_seed)
    torch.set_float32_matmul_precision('medium')

    data_module = MachineTranslationDataModule(
        en_filepath_suffix='_en.txt',
        de_filepath_suffix='_de.txt',
        maximum_length=parsed_args.maximum_length,
        batch_size=parsed_args.batch_size,
        num_training_samples=parsed_args.num_training_samples,
    )
    data_module.setup()

    logger = TensorBoardLogger(
        "tb_logs",
        name=f"machine_translation_NAS",
    )

    bleu_callback = BleuScoreValidationCallback()

    trainer = pl.Trainer(
        max_epochs=5,
        logger=logger,
        callbacks=[
            bleu_callback,
        ],
        log_every_n_steps=500,
    )

    sinusoidal_position_embedding = SinusoidalPositionEmbedding(parsed_args.embedding_dimension) if parsed_args.sinusoidal_position else NoAddEmbedding()
    learned_position_embedding = LearnedPositionEmbedding(max_sequence_length=3_000, embedding_dimension=parsed_args.embedding_dimension) if parsed_args.learned_position else NoAddEmbedding()
    rotary_position_embedding = RotaryPositionEmbedding(parsed_args.embedding_dimension // parsed_args.number_of_heads) if parsed_args.rotary_position else PassthroughEmbedding()
    alibi_position_embedding = ALiBiPositionEmbedding(parsed_args.number_of_heads) if parsed_args.alibi_position else NoAddEmbedding()

    numeric_embedding_manager = NumericEmbeddingManager(sinusoidal_position=sinusoidal_position_embedding, learned_position=learned_position_embedding, rotary_position=rotary_position_embedding, alibi_position=alibi_position_embedding)
    generic_attention = MultiheadAttention(embedding_dimension= parsed_args.embedding_dimension, number_of_heads= parsed_args.number_of_heads, attention_method= StandardAttentionMethod(parsed_args.dropout))
    decoder_self_attention = MultiheadAttention(embedding_dimension= parsed_args.embedding_dimension, number_of_heads= parsed_args.number_of_heads, attention_method= StandardAttentionMethod(parsed_args.dropout, is_causal_masking=True))
    feedforward_network = FeedForwardNetwork(parsed_args.embedding_dimension, parsed_args.feedforward_dimension, parsed_args.activation, parsed_args.dropout)
    model = MachineTranslationModel(
        src_vocab_size=data_module.de_vocab_size,
        tgt_vocab_size=data_module.en_vocab_size,
        encoder_self_attention=generic_attention,
        decoder_self_attention=decoder_self_attention,
        decoder_cross_attention=generic_attention,
        feedforward_network=feedforward_network,
        numeric_embedding_manager=numeric_embedding_manager,
        tgt_padding_token=data_module.en_pad_token,
        embedding_dimension=parsed_args.embedding_dimension,
        num_encoder_layers=parsed_args.number_of_layers,
        num_decoder_layers=parsed_args.number_of_layers,
        scheduler_warmup_steps = parsed_args.scheduler_warmup_steps,
        loss_type= parsed_args.loss_type,
        label_smoothing = parsed_args.label_smoothing,
    )

    trainer.fit(model, data_module)
    bleu_score = bleu_callback.bleu_score
    return bleu_score

class BleuScoreValidationCallback(pl.Callback):
    def __init__(self):
        self.generator = GeneratorContext(method='beam_batch')
        self.de_tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
        self.en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def on_train_epoch_end(self, trainer, pl_module, **kwargs):
        end_token = self.en_tokenizer.convert_tokens_to_ids(self.en_tokenizer.sep_token)
        start_token = self.en_tokenizer.convert_tokens_to_ids(self.en_tokenizer.cls_token)
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
                    #print(i)
                    #print(reference_tokens)
                    #print(output_tokens)
                    #print(reference_translation)
                    #print(output_translation)
                    reference_translations.append(reference_translation)
                    output_translations.append(output_translation)
        bleu_score = BLEU().corpus_score(output_translations, [reference_translations])
        pl_module.log("bleu_score", bleu_score.score, prog_bar=False, batch_size=batch_size)
        self.bleu_score = bleu_score.score

def parse_args():
    parser = argparse.ArgumentParser(description="generformer-nas")
    parser.add_argument("--log_path", type=str, required=True, help="dir to place tensorboard logs from all trials")
    parser.add_argument('--sinusoidal_position', action='store_true', default=False)
    parser.add_argument('--rotary_position', action='store_true', default=False)
    parser.add_argument('--alibi_position', action='store_true', default=False)
    parser.add_argument('--learned_position', action='store_true', default=False)
    parser.add_argument("--activation", type=str, required=True, help="activation function for all layers but the last one")
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--loss_type', type=str, default='custom', help='Type of loss function')
    parser.add_argument('--label_smoothing', type=float, default=0.9, help='Label smoothing value')
    parser.add_argument('--scheduler_warmup_steps', type=int, default=4000, help='Number of warmup steps for scheduler')
    parser.add_argument('--maximum_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--embedding_dimension', type=int, default=512, help='Embedding dimension. Original model used 512')
    parser.add_argument('--number_of_heads', type=int, default=8, help='Number of attention heads. Original model used 8')
    parser.add_argument('--feedforward_dimension', type=int, default=2048, help='Feedforward dimension. Original model used 2048.')
    parser.add_argument('--number_of_layers', type=int, default=6, help='Number of layers. Original model used 6')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_training_samples', type=int, default=None, help='number of training samples to use. Can reduce to speed up program at cost of accuracy.')
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    bleu_score = run_training_job(parsed_args)
    logger = TensorBoardLogger(parsed_args.log_path)
    logger.log_metrics({"bleu_score": bleu_score})
    logger.save()
    print(f'BLEU SCORE: {bleu_score}')

