import os
import pandas as pd
import numpy as np
import tensorflow as tf

# Import ProteinBERT’s model‐loading function and tokenization components.
from proteinbert import load_pretrained_model
from proteinbert.tokenization import additional_token_to_index, n_tokens

# reference: https://github.com/nadavbra/protein_bert/tree/master

# -------------------------------
# Helper: Masking Function
# -------------------------------
def mask_tokens(token_ids, mlm_probability, vocab_size, special_token_ids):
    """
    Given a list of token IDs, randomly mask tokens (except for special tokens)
    according to a probability mlm_probability.

    For positions chosen for masking, replace the token with a random token
    from the vocabulary and record the original token in mlm_labels.
    Positions not masked receive a label of -100.

    Args:
        token_ids (list of int): The tokenized sequence.
        mlm_probability (float): The probability to mask a token.
        vocab_size (int): Size of the vocabulary.
        special_token_ids (set of int): Token IDs for which masking is not applied.

    Returns:
        masked_input (list of int): The input with some tokens replaced.
        mlm_labels (list of int): The original tokens at masked positions, and -100 elsewhere.
    """
    mlm_labels = [-100] * len(token_ids)
    masked_input = list(token_ids)  # make a copy

    for i, token in enumerate(token_ids):
        if token in special_token_ids:
            continue  # do not mask special tokens

        if np.random.rand() < mlm_probability:
            # Record original token for loss computation
            mlm_labels[i] = token
            masked_input[i] = np.random.randint(0, vocab_size)

    return masked_input, mlm_labels


# -------------------------------
# Main Testing Script
# -------------------------------
def main():
    # --- Load Test Data ---
    test_csv_path = os.path.join("data", "test_data.csv")
    df = pd.read_csv(test_csv_path)

    sequences = df["Sequence"].tolist()
    print(f"Loaded {len(sequences)} sequences from {test_csv_path}")

    # --- Load ProteinBERT Pretrained Model and Input Encoder ---
    # load_pretrained_model returns a tuple: (pretrained_model_generator, input_encoder)
    pretrained_model_generator, input_encoder = load_pretrained_model()
    seq_len = 512
    model = pretrained_model_generator.create_model(seq_len)
    print(f"Loaded ProteinBERT model for sequence length {seq_len}")

    # Use the tokens provided in the vocabulary
    other_token_id = additional_token_to_index['<OTHER>']
    start_token_id = additional_token_to_index['<START>']
    end_token_id = additional_token_to_index['<END>']
    pad_token_id = additional_token_to_index['<PAD>']

    # Define special tokens to avoid masking
    special_token_ids = {start_token_id, end_token_id, pad_token_id, other_token_id}

    # The vocabulary size
    vocab_size = n_tokens

    # --- Set Up Evaluation Variables for Batch-level Averaging ---
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0  # here, each sequence is treated as one batch

    # Keras’s sparse categorical cross-entropy loss with reduction set to SUM.
    # later divide by the number of masked tokens in each sequence to get the average loss per sequence.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM
    )

    # --- Loop Over Test Sequences ---
    for seq in sequences:
        # Tokenize the sequence using ProteinBERT’s input encoder.
        # encode_X expects a list of sequences.
        encoded = input_encoder.encode_X([seq], seq_len)
        # encoded[0] is a numpy array of shape (1, seq_len); extract the first (and only) sequence.
        token_ids = encoded[0][0].tolist()
        token_ids = token_ids[:seq_len]

        # Apply the masking function (using a 5% masking probability).
        masked_input_ids, mlm_labels = mask_tokens(
            token_ids,
            mlm_probability=0.05,
            vocab_size=vocab_size,
            special_token_ids=special_token_ids
        )

        # Build inputs as expected by the model.
        # The model expects keys: "input-seq" and "input-annotations".
        inputs = {
            "input-seq": tf.constant([masked_input_ids], dtype=tf.int32),
            "input-annotations": tf.constant(encoded[1], dtype=tf.int32)
        }

        # Run the model in inference mode.
        outputs = model(inputs, training=False)
        # The model output is expected to be a tuple; use the first element (sequence output logits).
        if isinstance(outputs, (list, tuple)):
            seq_logits = outputs[0]
        else:
            seq_logits = outputs

        # Convert the MLM labels to a tensor and create a mask for positions to consider.
        mlm_labels_tensor = tf.constant([mlm_labels], dtype=tf.int32)
        mask = tf.not_equal(mlm_labels_tensor, -100)

        # Extract logits and labels for masked positions.
        masked_logits = tf.boolean_mask(seq_logits, mask)
        masked_labels = tf.boolean_mask(mlm_labels_tensor, mask)

        # Only update loss/accuracy if at least one token was masked.
        if tf.size(masked_labels) > 0:
            # Calculate the loss as the sum over the masked tokens.
            loss_sum = loss_fn(masked_labels, masked_logits)
            # To get the average loss for this sequence (batch), divide by the number of masked tokens.
            num_masked = tf.cast(tf.size(masked_labels), tf.float32)
            avg_loss = loss_sum / num_masked
            total_loss += avg_loss.numpy()

            # Calculate accuracy for this sequence.
            predictions = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(predictions, masked_labels), tf.int32)).numpy()
            seq_accuracy = correct / float(tf.size(masked_labels))
            total_accuracy += seq_accuracy

            num_batches += 1

    # --- Compute and Report Overall Metrics as Averages Over Batches ---
    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    average_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0

    print(f"Test Loss (average per batch): {average_loss:.4f}")
    print(f"Test Accuracy (average over batches): {average_accuracy:.4f}")

    # Save the results to a text file.
    with open("proteinbert_test_results.txt", "w") as f:
        f.write(f"Test Loss (average per batch): {average_loss:.4f}\n")
        f.write(f"Test Accuracy (average over batches): {average_accuracy:.4f}\n")
    print("Test results saved to proteinbert_test_results.txt")


if __name__ == '__main__':
    main()