import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import pytorch_lightning as pl
from copy import deepcopy
import math
from attention_smithy.components import Encoder, Decoder, EncoderLayer, DecoderLayer
from machine_translation.loss import MaskedLoss, LabelSmoothingLoss

class MachineTranslationModel(pl.LightningModule):
    """
    The full transformer model that performs a machine translation task.
    """
    def __init__(self,
             src_vocab_size: int,
             tgt_vocab_size: int,
             encoder_self_attention,
             decoder_self_attention,
             decoder_cross_attention,
             feedforward_network,
             numeric_embedding_facade,
             tgt_padding_token: int,
             scheduler_warmup_steps: int,
             loss_type: str,
             label_smoothing: float,
             embedding_dimension: int=512,
             num_encoder_layers: int=6,
             num_decoder_layers: int=6,
             dropout: float=0.1,
            ):
        """
        Args:
            src_vocab_size (int): The number of total possible tokens in the language being translated
                FROM (in German-to-English, this would be the number of possible German tokens).
            tgt_vocab_size (int): The number of total possible tokens in the language being translated
                TO (in German-to-English, this would be the number of possible English tokens).
            encoder_self_attention (AttentionMethod): The attention method used for the encoder self
                attention block. See AttentionSmithy.attention for available attention methods.
            decoder_self_attention (AttentionMethod): The attention method used for the decoder self
                attention block. See AttentionSmithy.attention for available attention methods.
            decoder_cross_attention (AttentionMethod): The attention method used for the decoder cross
                attention block. See AttentionSmithy.attention for available attention methods.
            feedforward_network (attention_smithy.components.feedforward): The class to be used in
                the feedforward block of both the encoder and the decoder. The class is duplicated
                and the weights are re-randomized for each duplicate.
            numeric_embedding_facade (attention_smithy.numeric_embeddings.NumericEmbeddingFacade):
                The class that contains all numeric (position) embedding strategies to be used in
                the model.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.numeric_embedding_facade = numeric_embedding_facade
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.src_token_embedding = nn.Embedding(src_vocab_size, embedding_dimension)
        self.tgt_token_embedding = nn.Embedding(tgt_vocab_size, embedding_dimension)
        encoder_layer = EncoderLayer(embedding_dimension, encoder_self_attention, feedforward_network, dropout)
        self.encoder = Encoder(encoder_layer, number_of_layers=num_encoder_layers)
        decoder_layer = DecoderLayer(embedding_dimension, decoder_self_attention, decoder_cross_attention, feedforward_network, dropout)
        self.decoder = Decoder(decoder_layer, number_of_layers=num_decoder_layers)
        self.vocab_output_layer = VocabOutputSoftmaxLayer(embedding_dimension, tgt_vocab_size)
        if loss_type == 'custom':
            self.loss_method = LabelSmoothingLoss(tgt_padding_token, confidence_probability_score=label_smoothing)
        elif loss_type == 'simple':
            self.loss_method = MaskedLoss(tgt_padding_token, label_smoothing=label_smoothing)
        else:
            raise RuntimeError("not a valid loss type")

    def forward(self, src_tensor, tgt_tensor, src_padding_mask, tgt_padding_mask):
        src_encoded = self.forward_encode(src_tensor, src_padding_mask)
        decoder_output = self.forward_decode(tgt_tensor, src_encoded, tgt_padding_mask, src_padding_mask)
        vocabulary_logits = self.vocab_output_layer(decoder_output)
        return vocabulary_logits

    def forward_encode(self, src_tensor, src_padding_mask):
        src_embedding = self.src_token_embedding(src_tensor) * math.sqrt(self.embedding_dimension)
        position_embedding = self.numeric_embedding_facade.calculate_sinusoidal_and_learned_tokenizations(src_embedding)
        event_encoded = self.encoder(src=src_embedding + position_embedding, src_padding_mask=src_padding_mask, numeric_embedding_facade=self.numeric_embedding_facade)
        return event_encoded

    def forward_decode(self, tgt_tensor, src_encoded, tgt_padding_mask, src_padding_mask):
        if tgt_tensor.shape[0] != src_encoded.shape[0]:
            beam_width = tgt_tensor.shape[0] // src_encoded.shape[0]
            src_encoded = src_encoded.repeat_interleave(beam_width, dim=0)
            src_padding_mask = src_padding_mask.repeat_interleave(beam_width, dim=0)
        tgt_embedding = self.tgt_token_embedding(tgt_tensor) * math.sqrt(self.embedding_dimension)
        output = self.decoder(
            tgt=tgt_embedding,
            src=src_encoded,
            tgt_padding_mask=tgt_padding_mask,
            src_padding_mask=src_padding_mask,
            numeric_embedding_facade=self.numeric_embedding_facade,
        )
        return output

    def training_step(self, batch, batch_idx):
        src_input_tensor, tgt_input_tensor, expected_output_tensor, src_padding_mask, tgt_padding_mask = batch
        vocabulary_logits = self(src_input_tensor, tgt_input_tensor, src_padding_mask, tgt_padding_mask)
        loss = self.loss_method(vocabulary_logits, expected_output_tensor)
        self.log("train_loss", loss, prog_bar=False, batch_size=vocabulary_logits.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        src_input_tensor, tgt_input_tensor, expected_output_tensor, src_padding_mask, tgt_padding_mask = batch
        vocabulary_logits = self(src_input_tensor, tgt_input_tensor, src_padding_mask, tgt_padding_mask)
        loss = self.loss_method(vocabulary_logits, expected_output_tensor)
        self.log("val_loss", loss, prog_bar=False, batch_size=vocabulary_logits.shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

        def lr_lambda(step):
            step = step + 1
            lr = self.embedding_dimension**(-0.5) * min(step**(-0.5), step * self.scheduler_warmup_steps**(-1.5))
            #print(f'\n\n{lr}\n')
            return lr

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

class VocabOutputSoftmaxLayer(nn.Module):
    def __init__(self,
             embedding_dimension: int,
             tgt_vocab_size: int,
            ):
        super().__init__()
        self.linear = nn.Linear(embedding_dimension, tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, decoder_output_logits):
        return self.softmax(self.linear(decoder_output_logits))
