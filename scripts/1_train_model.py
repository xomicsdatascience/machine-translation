import torch
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
        maximum_length=128,
        batch_size=64,
        embed_dim=512,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        random_seed=0,
):
    seed_everything(random_seed)

    data_module = MachineTranslationDataModule(
        en_filepath_suffix='_en.txt',
        de_filepath_suffix='_de.txt',
        maximum_length=maximum_length,
        batch_size=batch_size,
    )
    data_module.setup()
    logger = WandbLogger(project='machine-translation')

    train_loss_checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/",
        every_n_train_steps=50,
        filename="train-loss-{epoch:02d}-{step:08d}",
        save_last=True,
    )

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
                    for batch in trainer.val_dataloaders:
                        batch = tuple(x.to(pl_module.device) for x in batch)
                        src_input_tensor, tgt_input_tensor, expected_output_tensor, src_padding_mask, tgt_padding_mask = batch
                        output_tensor = torch.cat((tgt_input_tensor, expected_output_tensor[:, -1:]), dim=1)
                        batch_size = src_input_tensor.shape[0]
                        tgt_starting_input = torch.full((batch_size, 1), start_token).to(pl_module.device)
                        pl_module.validation_step(tuple([x.to(pl_module.device) for x in batch]), batch_idx)
                        src_encoded = pl_module.forward_encode(src_input_tensor.to(pl_module.device),
                                                               src_padding_mask=None)
                        generated_batch_tensor = self.generator.generate_sequence(pl_module,
                                                                            end_token,
                                                                            tgt_starting_input,
                                                                            src_encoded=src_encoded,
                                                                            src_padding_mask=src_padding_mask,
                                                                            tgt_padding_mask=None,
                                                                            )
                        for i in range(batch_size):
                            generated_tgt_tensor = list(generated_batch_tensor[i])
                            try:
                                end_token_index = generated_tgt_tensor.index(end_token) + 1
                            except ValueError:
                                end_token_index = len(generated_tgt_tensor) + 1
                            generated_tgt_tensor = generated_tgt_tensor[:end_token_index]
                            expected_tgt_tensor = output_tensor[i]
                            reference_translations.append(self.en_tokenizer.decode([int(x) for x in expected_tgt_tensor], skip_special_tokens=True))
                            output_translations.append(self.en_tokenizer.decode([int(x) for x in generated_tgt_tensor], skip_special_tokens=True))
                    bleu_score = BLEU().corpus_score(output_translations, [reference_translations])
                    pl_module.log("bleu", bleu_score.score, prog_bar=False, batch_size=batch_size)

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        callbacks=[
            train_loss_checkpoint_callback,
            ValidateAtCheckpoints(list(range(0, 100_000, 1000))[1:]),
        ],
        log_every_n_steps=50,
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
    )

    trainer.fit(model, data_module)
    torch.save(model, 'model.pth')


if __name__ == "__main__":
    print("start")
    train_model()
