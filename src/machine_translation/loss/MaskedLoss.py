import torch
from torch import nn
import torch.nn.functional as F

class MaskedLoss(nn.Module):
    """
    A class that performs masked loss functionality for the machine translation project.
    """

    def __init__(self, padding_token_idx, label_smoothing=0.0):
        """
        Args:
            padding_token_idx (int): The padding token ID. The probability for this specific token
                should always be 0 - it should never be considered in predicting the next token.
            label_smoothing (float, optional): The label smoothing factor. Defaults to 0.0.
        """

        super().__init__()
        self.padding_token_idx = padding_token_idx
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing, reduction='none')

    def forward(self, vocab_logits, expected_output_tokens):
        """
        Note: In training, the tgt input and the expected output are effectively shifted windows of
            each other. So if a single token sentence is represented by `sentence`, tgt input is
            `sentence[:-1]` and the expected output is `sentence[1:]`.

        Args:
            vocab_logits (torch.Tensor): The output of the machine translation model. This should
                be of shape (batch_size, (sequence_length-1), vocab_size). It represents the
                probability calculated by the model for each possible next token of the sequence.
            expected_output_tokens (torch.Tensor): A tensor representing the next tokens to be
                predicted, of shape (batch_size, (sequence_length-1)).
        """
        mask = expected_output_tokens!= self.padding_token_idx
        self._print_predictions(vocab_logits, expected_output_tokens)
        masked_loss = self.criterion(vocab_logits.view(-1, vocab_logits.shape[-1]),
                                     expected_output_tokens.view(-1))
        masked_loss = masked_loss * mask.view(-1)
        masked_loss = masked_loss.sum() / mask.sum()
        return masked_loss

    def _print_predictions(self, vocab_logits, expected_output_tokens):
        first_sample_token_logits = vocab_logits[0]

        first_sample_token_probs = torch.softmax(first_sample_token_logits, dim=-1)

        best_tokens = torch.argmax(first_sample_token_probs, dim=-1)
        printable_expected_output = [int(x) for x in expected_output_tokens[0] if int(x) != self.padding_token_idx]
        printable_generated_output = [int(x) for x in best_tokens][:len(printable_expected_output)]
        print(printable_expected_output)
        print(printable_generated_output)