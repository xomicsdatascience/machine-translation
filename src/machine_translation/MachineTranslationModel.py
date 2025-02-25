import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import pytorch_lightning as pl
from copy import deepcopy
import math
from attention_smithy.components import Encoder, Decoder, EncoderLayer, DecoderLayer
from attention_smithy.numeric_embeddings import (
    SinusoidalPositionEmbedding, LearnedPositionEmbedding,
    RotaryPositionEmbedding, ALiBiPositionEmbedding,
    NumericEmbeddingManager, NoAddEmbedding, PassthroughEmbedding
)
from attention_smithy.components import MultiheadAttention, FeedForwardNetwork
from attention_smithy.attention import StandardAttentionMethod
from machine_translation.loss import MaskedLoss, LabelSmoothingLoss


class MachineTranslationModel(pl.LightningModule):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, tgt_padding_token: int, **kwargs):
        """
        Initialize the model with required parameters and optional kwargs.

        Required Args:
            src_vocab_size (int): Size of source vocabulary
            tgt_vocab_size (int): Size of target vocabulary
            tgt_padding_token (int): Padding token ID for target vocabulary

        Optional Args (kwargs):
            embedding_dimension (int): Dimension of embeddings (default: 512)
            number_of_heads (int): Number of attention heads (default: 8)
            dropout (float): Dropout rate (default: 0.1)
            activation (str): Activation function (default: 'relu')
            feedforward_dimension (int): Dimension of feedforward layer (default: 2048)
            num_encoder_layers (int): Number of encoder layers (default: 6)
            num_decoder_layers (int): Number of decoder layers (default: 6)
            scheduler_warmup_steps (int): Warmup steps for scheduler (default: 4000)
            loss_type (str): Type of loss function (default: 'custom')
            label_smoothing (float): Label smoothing value (default: 0.9)
            use_sinusoidal (bool): Use sinusoidal position embedding (default: False)
            use_learned (bool): Use learned position embedding (default: False)
            use_rotary (bool): Use rotary position embedding (default: False)
            use_alibi (bool): Use ALiBi position embedding (default: False)
        """
        super().__init__()

        # Set default values for kwargs
        self.config = {
            'embedding_dimension': 512,
            'number_of_heads': 8,
            'dropout': 0.1,
            'activation': 'relu',
            'feedforward_dimension': 2048,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'scheduler_warmup_steps': 4000,
            'loss_type': 'custom',
            'label_smoothing': 0.9,
            'use_sinusoidal': False,
            'use_learned': False,
            'use_rotary': False,
            'use_alibi': False,
        }

        self.config.update(kwargs)

        self.save_hyperparameters()

        self.embedding_dimension = self.config['embedding_dimension']
        self.src_token_embedding = nn.Embedding(src_vocab_size, self.embedding_dimension)
        self.tgt_token_embedding = nn.Embedding(tgt_vocab_size, self.embedding_dimension)

        self.numeric_embedding_manager = self._create_embedding_manager()

        generic_attention = MultiheadAttention(
            embedding_dimension=self.embedding_dimension,
            number_of_heads=self.config['number_of_heads'],
            attention_method=StandardAttentionMethod(self.config['dropout'])
        )

        decoder_self_attention = MultiheadAttention(
            embedding_dimension=self.embedding_dimension,
            number_of_heads=self.config['number_of_heads'],
            attention_method=StandardAttentionMethod(self.config['dropout'], is_causal_masking=True)
        )

        feedforward_network = FeedForwardNetwork(
            self.embedding_dimension,
            self.config['feedforward_dimension'],
            self.config['activation'],
            self.config['dropout']
        )

        encoder_layer = EncoderLayer(
            self.embedding_dimension,
            generic_attention,
            feedforward_network,
            self.config['dropout']
        )
        self.encoder = Encoder(encoder_layer, number_of_layers=self.config['num_encoder_layers'])

        decoder_layer = DecoderLayer(
            self.embedding_dimension,
            decoder_self_attention,
            generic_attention,  # Cross attention
            feedforward_network,
            self.config['dropout']
        )
        self.decoder = Decoder(decoder_layer, number_of_layers=self.config['num_decoder_layers'])

        self.vocab_output_layer = VocabOutputSoftmaxLayer(self.embedding_dimension, tgt_vocab_size)

        self.loss_method = (
            LabelSmoothingLoss(tgt_padding_token, confidence_probability_score=self.config['label_smoothing'])
            if self.config['loss_type'] == 'custom'
            else MaskedLoss(tgt_padding_token, label_smoothing=self.config['label_smoothing'])
        )

        self.scheduler_warmup_steps = self.config['scheduler_warmup_steps']

    def _create_embedding_manager(self):
        """Create embedding manager with specified embedding types from config."""
        sinusoidal_position = (
            SinusoidalPositionEmbedding(self.embedding_dimension)
            if self.config['use_sinusoidal'] else NoAddEmbedding()
        )

        learned_position = (
            LearnedPositionEmbedding(max_sequence_length=3_000, embedding_dimension=self.embedding_dimension)
            if self.config['use_learned'] else NoAddEmbedding()
        )

        rotary_position = (
            RotaryPositionEmbedding(self.embedding_dimension // self.config['number_of_heads'])
            if self.config['use_rotary'] else PassthroughEmbedding()
        )

        alibi_position = (
            ALiBiPositionEmbedding(self.config['number_of_heads'])
            if self.config['use_alibi'] else NoAddEmbedding()
        )

        return NumericEmbeddingManager(
            sinusoidal_position=sinusoidal_position,
            learned_position=learned_position,
            rotary_position=rotary_position,
            alibi_position=alibi_position
        )

    def forward(self, src_tensor, tgt_tensor, src_padding_mask, tgt_padding_mask):
        src_encoded = self.forward_encode(src_tensor, src_padding_mask)
        vocabulary_logits = self.forward_decode(tgt_tensor, src_encoded, tgt_padding_mask, src_padding_mask)
        return vocabulary_logits

    def forward_encode(self, src_tensor, src_padding_mask):
        src_embedding = self.src_token_embedding(src_tensor) * math.sqrt(self.embedding_dimension)
        position_embedding = self.numeric_embedding_manager.calculate_sinusoidal_and_learned_tokenizations(src_embedding)
        event_encoded = self.encoder(src=src_embedding + position_embedding, src_padding_mask=src_padding_mask, numeric_embedding_manager=self.numeric_embedding_manager)
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
            numeric_embedding_manager=self.numeric_embedding_manager,
        )
        vocabulary_logits = self.vocab_output_layer(output)
        return vocabulary_logits

    def training_step(self, batch, batch_idx):
        src_input_tensor, tgt_input_tensor, expected_output_tensor, src_padding_mask, tgt_padding_mask = batch
        vocabulary_logits = self(src_input_tensor, tgt_input_tensor, src_padding_mask, tgt_padding_mask)
        loss = self.loss_method(vocabulary_logits, expected_output_tensor, batch_idx)
        self.log("train_loss", loss, prog_bar=False, batch_size=vocabulary_logits.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        src_input_tensor, tgt_input_tensor, expected_output_tensor, src_padding_mask, tgt_padding_mask = batch
        vocabulary_logits = self(src_input_tensor, tgt_input_tensor, src_padding_mask, tgt_padding_mask)
        loss = self.loss_method(vocabulary_logits, expected_output_tensor, batch_idx)
        self.log("val_loss", loss, prog_bar=False, batch_size=vocabulary_logits.shape[0])
        return loss


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        num_gpus = torch.cuda.device_count()
        effective_warmup_steps = self.scheduler_warmup_steps // num_gpus

        def lr_lambda(step):
            step = step + 1
            lr = self.embedding_dimension ** (-0.5) * min(step ** (-0.5), step * effective_warmup_steps ** (-1.5))
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
