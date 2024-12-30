import torch
from torch import nn

class LabelSmoothingLoss(nn.Module):
    """
    A class that performs loss functionality for the machine translation project. While a standard
        KLDivLoss could be directly applied, the writers of the Attention Is All You Need paper outlined
        a label smoothing pre-process step that is also performed here.

    In training, we want the model to accurately predict the next token in a sequence. At a high level,
        you could say that the correct next token is 100% right, and all other tokens are 0% right. The
        model will train just fine if you calculate the loss under this assumption. However, the authors
        determined to make the correct next token "probably" right, rather than "absolutely" right. In
        short, the correct next token is 90% (adjustable) right, and the remaining 10% probability is
        evenly distributed across the remaining tokens. This "smooths" the label probabilities, hence
        the name.

    This process hurts the training loss score, but improves the final BLEU score, the primary metric used
        in evaluating translation models from a human's perspective.
    """


    def __init__(self, padding_token_idx, confidence_probability_score):
        """
        Args:
            padding_token_idx (int): The padding token ID. The probability for this specific token
                should always be 0 - it should never be considered in predicting the next token.
            confidence_probability_score (float): The confidence probability assigned to the correct
                token (in place of 1.0, or 100%).
        """

        super().__init__()
        self.padding_token_idx = padding_token_idx
        self.confidence_probability_score = confidence_probability_score
        self.inverse_probability_score = 1.0 - confidence_probability_score
        self.negating_probability_score = 0.0
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, vocab_logits, expected_output_tokens, batch_idx):
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
        self.device = vocab_logits.device
        if batch_idx % 1000 == 0:
            self._print_predictions(vocab_logits, expected_output_tokens)
        smooth_label_expected_distribution = self._create_smooth_label_expected_distribution(expected_output_tokens, *vocab_logits.shape)
        vocab_logits_reshaped, smooth_label_expected_distribution_reshaped = self._reshape_to_remove_padding_token_targets(
            vocab_logits, smooth_label_expected_distribution, expected_output_tokens)
        return self.criterion(vocab_logits_reshaped, smooth_label_expected_distribution_reshaped)

    def _create_smooth_label_expected_distribution(self, expected_output_tokens, batch_size, tgt_sequence_length, tgt_vocab_size):
        smooth_label_expected_distribution = self._initialize_label_distribution_with_low_confidence_values(batch_size, tgt_sequence_length, tgt_vocab_size)
        self._set_target_token_to_high_confidence_value(expected_output_tokens, smooth_label_expected_distribution)
        self._negate_confidence_values_for_padding_tokens(expected_output_tokens, smooth_label_expected_distribution)
        return smooth_label_expected_distribution

    def _initialize_label_distribution_with_low_confidence_values(self, batch_size, tgt_sequence_length, tgt_vocab_size):
        number_of_non_target_non_padding_tokens = tgt_vocab_size - 2
        dispersed_inverse_probability_score = self.inverse_probability_score / number_of_non_target_non_padding_tokens
        smooth_label_expected_distribution = torch.full((batch_size, tgt_sequence_length, tgt_vocab_size),
                                                        dispersed_inverse_probability_score, device=self.device)
        return smooth_label_expected_distribution

    def _set_target_token_to_high_confidence_value(self, expected_output_tokens, smooth_label_expected_distribution):
        smooth_label_expected_distribution.scatter_(-1, expected_output_tokens.unsqueeze(-1), self.confidence_probability_score)

    def _negate_confidence_values_for_padding_tokens(self, expected_output_tokens, smooth_label_expected_distribution):
        smooth_label_expected_distribution[:, :, self.padding_token_idx] = self.negating_probability_score

    def _reshape_to_remove_padding_token_targets(self,
                                                    vocab_logits,
                                                    smooth_label_expected_distribution,
                                                    expected_output_tokens,
                                                ):
        batch_size, tgt_sequence_length, tgt_vocab_size = vocab_logits.shape
        vocab_logits_reshaped = vocab_logits.reshape(batch_size*tgt_sequence_length, tgt_vocab_size)
        smooth_label_expected_distribution_reshaped = smooth_label_expected_distribution.reshape(batch_size*tgt_sequence_length, tgt_vocab_size)
        padding_token_mask = expected_output_tokens.flatten() == self.padding_token_idx
        vocab_logits_reshaped = vocab_logits_reshaped[~padding_token_mask]
        smooth_label_expected_distribution_reshaped = smooth_label_expected_distribution_reshaped[~padding_token_mask]
        return vocab_logits_reshaped, smooth_label_expected_distribution_reshaped

    def _print_predictions(self, vocab_logits, expected_output_tokens):
        first_sample_token_logits = vocab_logits[0]

        first_sample_token_probs = torch.softmax(first_sample_token_logits, dim=-1)

        best_tokens = torch.argmax(first_sample_token_probs, dim=-1)
        printable_expected_output = [int(x) for x in expected_output_tokens[0] if int(x) != self.padding_token_idx]
        printable_generated_output = [int(x) for x in best_tokens][:len(printable_expected_output)]
        print(printable_expected_output)
        print(printable_generated_output)