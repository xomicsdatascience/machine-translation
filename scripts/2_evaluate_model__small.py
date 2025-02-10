import os
import re
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from machine_translation import MachineTranslationModel
from machine_translation.data import MachineTranslationDataModule
from attention_smithy.numeric_embeddings import SinusoidalPositionEmbedding, NumericEmbeddingManager
from attention_smithy.components import MultiheadAttention, FeedForwardNetwork
from attention_smithy.attention import StandardAttentionMethod
from attention_smithy.utils import seed_everything
from attention_smithy.generators import GeneratorContext
from transformers import AutoTokenizer
from sacrebleu.metrics import BLEU

def evaluate_model(
        checkpoint_filepath: str,
        maximum_length=100,
        batch_size=64,
        embed_dim=128,
        num_heads=8,
        dim_feedforward=512,
        number_of_layers=4,
        dropout=0.1,
        random_seed=0,
):
    seed_everything(random_seed)

    def extract_info_from_filename(filepath):
        # Extract the filename from the filepath
        filename = os.path.basename(filepath)

        # Use regular expression to extract the string, float, and int
        match = re.match(r'([a-zA-Z]+)-([0-9\.]+)-([0-9]+)', filename)

        if match:
            string_part = match.group(1)
            float_part = float(match.group(2))
            int_part = int(match.group(3))

            return string_part, float_part, int_part
        else:
            return None

    data_module = MachineTranslationDataModule(
        en_filepath_suffix='_en.txt',
        de_filepath_suffix='_de.txt',
        maximum_length=maximum_length,
        batch_size=batch_size,
    )
    data_module.setup()

    class ValidateAtCheckpoints(pl.Callback):
        def __init__(self):
            self.generator = GeneratorContext(method='beam_batch', no_repeat_ngram_size=3)
            self.de_tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
            self.en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        def on_train_epoch_end(self, pl_module, val_dataloader, **kwargs):
            end_token = self.en_tokenizer.convert_tokens_to_ids(self.en_tokenizer.sep_token)
            start_token = self.en_tokenizer.convert_tokens_to_ids(self.en_tokenizer.cls_token)
            reference_translations = []
            output_translations = []
            with torch.no_grad():
                count = 0
                max_num_batches = 2
                for batch in val_dataloader:
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
                print(bleu_score)

    sinusoidal_position_embedding = SinusoidalPositionEmbedding(embed_dim)
    numeric_embedding_manager = NumericEmbeddingManager(sinusoidal_position=sinusoidal_position_embedding)
    generic_attention = MultiheadAttention(embedding_dimension = embed_dim, number_of_heads = num_heads, attention_method = StandardAttentionMethod(dropout))
    decoder_self_attention = MultiheadAttention(embedding_dimension = embed_dim, number_of_heads = num_heads, attention_method = StandardAttentionMethod(dropout, is_causal_masking=True))
    feedforward_network = FeedForwardNetwork(embed_dim, dim_feedforward, 'relu', dropout)

    loss_type, label_smoothing, scheduler_warmup_steps = extract_info_from_filename(checkpoint_filepath)

    model = MachineTranslationModel.load_from_checkpoint(
        src_vocab_size=data_module.de_vocab_size,
        tgt_vocab_size=data_module.en_vocab_size,
        encoder_self_attention=generic_attention,
        decoder_self_attention=decoder_self_attention,
        decoder_cross_attention=generic_attention,
        feedforward_network=feedforward_network,
        numeric_embedding_manager=numeric_embedding_manager,
        tgt_padding_token=data_module.en_pad_token,
        embedding_dimension=embed_dim,
        num_encoder_layers=number_of_layers,
        num_decoder_layers=number_of_layers,
        scheduler_warmup_steps = scheduler_warmup_steps,
        loss_type= loss_type,
        label_smoothing = label_smoothing,
        checkpoint_path=checkpoint_filepath,
        strict=True,
    )
    callback = ValidateAtCheckpoints()
    callback.on_train_epoch_end(model, data_module.val_dataloader())

if __name__ == "__main__":
    checkpoint_filepath = sys.argv[1]
    print(checkpoint_filepath)
    evaluate_model(checkpoint_filepath)
