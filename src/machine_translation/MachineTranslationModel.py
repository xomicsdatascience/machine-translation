import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from copy import deepcopy
import math
from attention_smithy.components import Encoder, Decoder, EncoderLayer, DecoderLayer

class MachineTranslationModel(pl.LightningModule):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 multihead_attention,
                 feedforward_network,
                 numeric_embedding_facade,
                 embedding_dimension=512,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dropout=0.1):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.numeric_embedding_facade = numeric_embedding_facade
        self.src_token_embedding = nn.Embedding(src_vocab_size, embedding_dimension)
        self.tgt_token_embedding = nn.Embedding(tgt_vocab_size, embedding_dimension)
        encoder_layer = EncoderLayer(embedding_dimension, multihead_attention, feedforward_network, dropout)
        self.encoder = Encoder(encoder_layer, number_of_layers=num_encoder_layers)
        decoder_layer = DecoderLayer(embedding_dimension, multihead_attention, multihead_attention, feedforward_network, dropout)
        self.decoder = Decoder(encoder_layer, number_of_layers=num_decoder_layers)
        self.loss_method = None

    def forward(self, src_tensor, tgt_tensor, src_padding_mask, tgt_padding_mask):
        src_encoded = self.encode(src_tensor, src_padding_mask)
        output = self.decode(tgt_tensor, src_encoded, tgt_padding_mask, src_padding_mask)
        return output

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
        logits = self(*batch)
        #self.log("train_loss_categorical", categorical_loss, prog_bar=False, batch_size=logits.shape[0])
        return logits

    def validation_step(self, batch, batch_idx):
        logits = self(*batch)
        #self.log("val_loss_categorical", categorical_loss, prog_bar=False, batch_size=logits.shape[0])
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer



