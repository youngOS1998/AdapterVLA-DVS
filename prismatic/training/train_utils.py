"""Utils for training/fine-tuning scripts."""

import torch

from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids):
    # import pdb; pdb.set_trace()
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def vision_instruction_mask(text_mask, drop_mask, num_patches):
    """Tokens the future-displacement head may read.

    ``text_mask`` and ``drop_mask`` line up with the text token ids. The VLM
    hidden state inserts ``num_patches`` vision tokens after BOS, so the
    returned mask lines up with that hidden state. Dropped text positions are
    the action queries and the stop token. BOS, the vision patches, and the
    instruction stay.
    """
    drop_mask = drop_mask.to(device=text_mask.device)
    keep = text_mask.to(dtype=torch.bool) & ~drop_mask.to(dtype=torch.bool)
    vision = torch.ones(
        text_mask.shape[0],
        int(num_patches),
        dtype=torch.bool,
        device=text_mask.device,
    )
    return torch.cat([keep[:, :1], vision, keep[:, 1:]], dim=1)


def get_next_actions_mask(token_ids):
    # import pdb; pdb.set_trace()
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = cumsum > ACTION_DIM

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss
