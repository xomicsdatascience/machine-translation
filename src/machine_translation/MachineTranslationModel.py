import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from copy import deepcopy
import math
from attention_smithy.components import Encoder, Decoder, EncoderLayer, DecoderLayer
from machine_translation.loss import LabelSmoothingLoss

class MachineTranslationModel(pl.LightningModule):
    def __init__(self,
             src_vocab_size: int,
             tgt_vocab_size: int,
             multihead_attention,
             feedforward_network,
             numeric_embedding_facade,
             tgt_padding_token: int,
             embedding_dimension: int=512,
             num_encoder_layers: int=6,
             num_decoder_layers: int=6,
             dropout: float=0.1,
            ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.numeric_embedding_facade = numeric_embedding_facade
        self.src_token_embedding = nn.Embedding(src_vocab_size, embedding_dimension)
        self.tgt_token_embedding = nn.Embedding(tgt_vocab_size, embedding_dimension)
        encoder_layer = EncoderLayer(embedding_dimension, multihead_attention, feedforward_network, dropout)
        self.encoder = Encoder(encoder_layer, number_of_layers=num_encoder_layers)
        decoder_layer = DecoderLayer(embedding_dimension, multihead_attention, multihead_attention, feedforward_network, dropout)
        self.decoder = Decoder(decoder_layer, number_of_layers=num_decoder_layers)
        self.vocab_output_layer = VocabOutputSoftmaxLayer(embedding_dimension, tgt_vocab_size)
        self.loss_method = LabelSmoothingLoss(tgt_padding_token, confidence_probability_score=0.9)

    def forward(self, src_tensor, tgt_tensor, src_padding_mask, tgt_padding_mask):
        src_encoded = self.encode(src_tensor, src_padding_mask)
        decoder_output = self.decode(tgt_tensor, src_encoded, tgt_padding_mask, src_padding_mask)
        vocabulary_logits = self.vocab_output_layer(decoder_output)
        return vocabulary_logits

    def encode(self, src_tensor, src_padding_mask):
        src_embedding = self.src_token_embedding(src_tensor) * math.sqrt(self.embedding_dimension)
        position_embedding = self.numeric_embedding_facade.calculate_sinusoidal_and_learned_tokenizations(src_embedding)
        event_encoded = self.encoder(src=src_embedding + position_embedding, src_padding_mask=src_padding_mask, numeric_embedding_facade=self.numeric_embedding_facade)
        return event_encoded

    def decode(self, tgt_tensor, src_encoded, tgt_padding_mask, src_padding_mask):
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

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
