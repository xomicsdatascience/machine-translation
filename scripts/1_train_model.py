import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
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

class TensorBoardLoggingModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        if self.monitor in trainer.callback_metrics:
            metric_value = trainer.callback_metrics[self.monitor]
            trainer.logger.experiment.add_scalar(
                f"checkpoint/{self.monitor}", metric_value, trainer.global_step
            )

def train_model(
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
    )
    data_module.setup()
    logger = TensorBoardLogger(
        "tb_logs",
        name=f"machine_translation",
    )

    train_loss_checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/",
        every_n_train_steps=50,
        filename="train-loss-{epoch:02d}-{step:08d}",
        save_last=True,
    )

    val_loss_checkpoint_callback = TensorBoardLoggingModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/",
        filename="best-val-loss-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    class ValidateAtCheckpoints(pl.Callback):
        def __init__(self, checkpoints):
            self.checkpoints = checkpoints
            self.generator = GeneratorContext(method='beam')
            self.de_tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
            self.en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        def on_train_batch_end(self, trainer, pl_module, outputs, train_batch, batch_idx, **kwargs):
            if batch_idx in self.checkpoints:
                reference_translations = []
                output_translations = []

                for batch in trainer.val_dataloaders:
                    src_input_tensor, tgt_input_tensor, expected_output_tensor, src_padding_mask, tgt_padding_mask = batch
                    batch_size = src_input_tensor.shape[0]
                    for i in range(batch_size):
                        #if i % 5 == 0 and i != 0:
                        #    print(i)
                        #    break
                        src_input_sample = src_input_tensor[i]
                        tgt_input_sample = tgt_input_tensor[i]
                        src_mask = src_input_sample != self.de_tokenizer.convert_tokens_to_ids(self.de_tokenizer.pad_token)
                        tgt_mask = tgt_input_sample != self.en_tokenizer.convert_tokens_to_ids(self.en_tokenizer.pad_token)
                        src_input_sample_without_padding = torch.masked_select(src_input_sample, src_mask)
                        src_input_sample_without_padding = src_input_sample_without_padding.unsqueeze(0)

                        tgt_input_sample_without_padding = torch.masked_select(tgt_input_sample, tgt_mask)
                        tgt_input_sample_without_padding = tgt_input_sample_without_padding.unsqueeze(0)
                        src_encoded = pl_module.forward_encode(src_input_sample_without_padding.to(pl_module.device), src_padding_mask=None)
                        generated_tensor = self.generator.generate_sequence(pl_module,
                                                                            self.en_tokenizer.convert_tokens_to_ids(
                                                                                self.en_tokenizer.sep_token),
                                                                            torch.tensor([[
                                                                                              self.en_tokenizer.convert_tokens_to_ids(
                                                                                                  self.en_tokenizer.cls_token)]]).to(
                                                                                src_encoded.device),
                                                                            src_encoded=src_encoded,
                                                                            src_padding_mask=None,
                                                                            tgt_padding_mask=None,
                                                                            )
                        reference_translations.append(
                            self.en_tokenizer.decode([int(x) for x in tgt_input_sample_without_padding.flatten()]))
                        output_translations.append(self.en_tokenizer.decode([int(x) for x in generated_tensor]))
                    #break
                bleu_score = BLEU().corpus_score(output_translations, [reference_translations])
                #print("BLEU score:", bleu_score.score)
                #print('*' * 50)
                #for output_translation, reference_translation in zip(output_translations, reference_translations):
                #    print(output_translation)
                #    print(reference_translation)
                #    print('*' * 25)
                pl_module.log("bleu", bleu_score.score, prog_bar=False, batch_size=batch_size)

                val_batch = next(iter(trainer.val_dataloaders))
                val_batch = tuple([x.to(pl_module.device) for x in val_batch])
                pl_module.validation_step(val_batch, batch_idx)

    trainer = pl.Trainer(
        max_epochs=30,
        logger=logger,
        callbacks=[
            train_loss_checkpoint_callback,
            val_loss_checkpoint_callback,
            ValidateAtCheckpoints(list(range(0, 28128, 50))[1:]),
        ],
        log_every_n_steps=50,

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
