import os
import pandas as pd
import numpy as np
import tensorflow as tf

# Import ProteinBERT’s model‐loading function and tokenization components.
from proteinbert import load_pretrained_model
from proteinbert.tokenization import additional_token_to_index, n_tokens


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
# Helper: Logit Analysis Function
# -------------------------------
def analyze_logits(masked_logits, masked_labels, top_k=5):
    """
    Given logits for masked positions and the corresponding true token labels,
    compute the softmax probability distribution and print top-k predictions,
    along with the entropy of the distribution.
    """
    # Convert logits to softmax probabilities.
    probs = tf.nn.softmax(masked_logits, axis=-1).numpy()
    analysis_results = []

    # For each masked position, analyze its probability distribution.
    for i, prob_dist in enumerate(probs):
        # Compute entropy (in nats)
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
        # Get top-k token indices and probabilities.
        top_indices = prob_dist.argsort()[-top_k:][::-1]
        top_probs = prob_dist[top_indices]
        true_token = masked_labels[i].numpy() if isinstance(masked_labels[i], tf.Tensor) else masked_labels[i]
        analysis_results.append({
            "masked_position": i,
            "true_token": true_token,
            "top_predictions": list(zip(top_indices, top_probs)),
            "entropy": entropy
        })
    return analysis_results


# -------------------------------
# Main Testing Script with Debugging, Logit Analysis, and Truncation
# -------------------------------
def main():
    # --- Load Test Data ---
    test_csv_path = os.path.join("data", "test_data.csv")
    df = pd.read_csv(test_csv_path)
    # Assumes that the primary sequence is stored in the "Sequence" column.
    sequences = df["Sequence"].tolist()
    print(f"Loaded {len(sequences)} sequences from {test_csv_path}")

    # --- Load ProteinBERT Pretrained Model and Input Encoder ---
    pretrained_model_generator, input_encoder = load_pretrained_model()
    seq_len = 512
    model = pretrained_model_generator.create_model(seq_len)
    print(f"Loaded ProteinBERT model for sequence length {seq_len}")

    # Print model summary and input names.
    model.summary()
    print("Model input names:", model.input_names)

    # Use the tokens provided in the vocabulary.
    other_token_id = additional_token_to_index['<OTHER>']
    start_token_id = additional_token_to_index['<START>']
    end_token_id = additional_token_to_index['<END>']
    pad_token_id = additional_token_to_index['<PAD>']
    special_token_ids = {start_token_id, end_token_id, pad_token_id, other_token_id}

    # The vocabulary size
    vocab_size = n_tokens

    # --- Set Up Evaluation Variables ---
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    # Use Keras’s sparse categorical cross-entropy loss (from logits).
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

    # For debugging, inspect a few examples.
    debug_examples = 3
    debug_count = 0

    # --- Loop Over Test Sequences ---
    for seq in sequences:
        # Tokenize the sequence using ProteinBERT’s input encoder.
        encoded = input_encoder.encode_X([seq], seq_len)
        token_ids = encoded[0][0].tolist()

        # FIX: Truncate token_ids to seq_len if necessary.
        if len(token_ids) > seq_len:
            token_ids = token_ids[:seq_len]

        if debug_count < debug_examples:
            print("\n=== Debug Example", debug_count + 1, "===")
            print("Original token_ids (length {}):".format(len(token_ids)), token_ids)

        # Apply the masking function (5% masking probability).
        masked_input_ids, mlm_labels = mask_tokens(
            token_ids,
            mlm_probability=0.05,
            vocab_size=vocab_size,
            special_token_ids=special_token_ids
        )

        if debug_count < debug_examples:
            print("Masked input_ids:", masked_input_ids)
            print("MLM labels:", mlm_labels)

        # Build inputs as expected by the model.
        inputs = {
            "input-seq": tf.constant([masked_input_ids], dtype=tf.int32),
            "input-annotations": tf.constant(encoded[1], dtype=tf.int32)
        }

        # Run the model in inference mode.
        outputs = model(inputs, training=False)
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

        if debug_count < debug_examples and tf.size(masked_labels) > 0:
            predictions = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)
            print("Predicted tokens at masked positions:", predictions.numpy())
            print("Actual masked tokens:", masked_labels.numpy())

            # --- Logit Analysis ---
            analysis = analyze_logits(masked_logits, masked_labels)
            for result in analysis:
                print(f"Masked position {result['masked_position']}:")
                print(f"  True token: {result['true_token']}")
                for idx, prob in result['top_predictions']:
                    print(f"  Predicted token {idx} with probability {prob:.4f}")
                print(f"  Entropy: {result['entropy']:.4f}")

        # Update loss/accuracy if at least one token was masked.
        if tf.size(masked_labels) > 0:
            loss = loss_fn(masked_labels, masked_logits)
            total_loss += loss.numpy()

            predictions = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(predictions, masked_labels), tf.int32)).numpy()
            total_correct += correct
            total_masked += int(masked_labels.shape[0])

        debug_count += 1

    # --- Compute and Report Overall Metrics ---
    average_loss = total_loss / (total_masked if total_masked > 0 else 1)
    accuracy = total_correct / total_masked if total_masked > 0 else 0.0

    print(f"\nTest Loss (per masked token): {average_loss:.4f}")
    print(f"Test Accuracy (masked tokens): {accuracy:.4f}")

    # Save the results to a text file.
    with open("proteinbert_test_results.txt", "w") as f:
        f.write(f"Test Loss (per masked token): {average_loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
    print("Test results saved to proteinbert_test_results.txt")


if __name__ == '__main__':
    main()