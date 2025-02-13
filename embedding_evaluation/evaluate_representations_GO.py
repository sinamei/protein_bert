#!/usr/bin/env python3
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import plotly.express as px  # For interactive plots
import distinctipy  # Make sure to install via: pip install distinctipy

# Import ProteinBERT functions and tokenizer constants.
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from proteinbert.tokenization import additional_token_to_index, n_tokens


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_proteinbert_model(seq_len=512):
    """
    Load the ProteinBERT pretrained model and input encoder,
    then wrap the model so that it outputs hidden representations.
    """
    pretrained_model_generator, input_encoder = load_pretrained_model()
    base_model = pretrained_model_generator.create_model(seq_len)
    model = get_model_with_hidden_layers_as_outputs(base_model)
    return model, input_encoder


def force_fixed_length(token_list, seq_len, pad_token):
    """
    Given a list of token IDs, truncate to seq_len if too long,
    or pad with pad_token if too short.
    """
    tokens = list(token_list)
    if len(tokens) > seq_len:
        return tokens[:seq_len]
    else:
        return tokens + [pad_token] * (seq_len - len(tokens))


def encode_sequences_fixed(input_encoder, sequences, seq_len):
    """
    Encode a list of protein sequences individually using ProteinBERT's input encoder,
    forcing each tokenized sequence to be exactly seq_len tokens long.

    Returns:
      encoded_x_fixed: numpy array of shape (N, seq_len) with token IDs.
      encoded_annotations_fixed: numpy array of shape (N, seq_len) with annotations.
    """
    pad_token = additional_token_to_index['<PAD>']
    encoded_list = []
    annotations_list = []

    for seq in sequences:
        # Encode the sequence (input_encoder.encode_X expects a list of sequences).
        encoded_x, encoded_annotations = input_encoder.encode_X([seq], seq_len)
        tokens = encoded_x[0].tolist()  # Convert token list to Python list.
        tokens_fixed = force_fixed_length(tokens, seq_len, pad_token)
        encoded_list.append(tokens_fixed)
        annotations_list.append(encoded_annotations[0])

    encoded_x_fixed = np.array(encoded_list, dtype=np.int32)
    encoded_annotations_fixed = np.array(annotations_list, dtype=np.int32)
    return encoded_x_fixed, encoded_annotations_fixed


def get_global_embeddings(model, input_encoder, sequences, seq_len=512, batch_size=32):
    """
    Given a list of protein sequences, pre-truncate each sequence to seq_len characters,
    then encode them using our custom fixed-length encoder, and finally extract
    global representations from the model.

    Returns:
      global_embeddings: numpy array of shape (N, embedding_dim)
    """
    sequences_truncated = [seq[:seq_len] for seq in sequences]
    encoded_x_fixed, encoded_annotations_fixed = encode_sequences_fixed(input_encoder, sequences_truncated, seq_len)
    # The model's predict method returns a tuple: (local_reps, global_reps)
    _, global_representations = model.predict((encoded_x_fixed, encoded_annotations_fixed), batch_size=batch_size)
    return global_representations


def generate_distinct_color_palette(num_colors):
    """
    Generate a list of distinct hex color codes using distinctipy.

    Parameters:
      num_colors (int): Number of colors to generate.

    Returns:
      list: List of hex color codes.
    """
    colors = distinctipy.get_colors(num_colors)
    colors_hex = [distinctipy.get_hex(color) for color in colors]
    return colors_hex


def plot_2d_interactive(embeddings_2d, labels=None, title="2D Projection", save_path=None):
    """
    Create and (optionally) save an interactive 2D scatter plot using Plotly.
    If labels are provided, use them for coloring.

    Parameters:
      embeddings_2d: (N,2) numpy array of coordinates.
      labels: (Optional) list/array of label values.
      title: Plot title.
      save_path: (Optional) file path to save the plot as HTML.
    """
    if labels is not None:
        df_plot = pd.DataFrame({
            'Component 1': embeddings_2d[:, 0],
            'Component 2': embeddings_2d[:, 1],
            'Label': labels
        })
        unique_groups = sorted(df_plot['Label'].unique())
        color_palette = generate_distinct_color_palette(len(unique_groups))
        color_map = {group: color for group, color in zip(unique_groups, color_palette)}
        fig = px.scatter(df_plot, x='Component 1', y='Component 2', title=title,
                         color='Label', color_discrete_map=color_map)
    else:
        df_plot = pd.DataFrame({
            'Component 1': embeddings_2d[:, 0],
            'Component 2': embeddings_2d[:, 1]
        })
        fig = px.scatter(df_plot, x='Component 1', y='Component 2', title=title,
                         color_discrete_sequence=px.colors.qualitative.Alphabet)

    fig.show()
    if save_path:
        fig.write_html(save_path)
        print(f"2D plot saved to {save_path}")


def main():
    set_seed(42)
    seq_len = 512
    batch_size = 32

    # Load the test CSV file (expects a "Sequence" column and a "go_ids" column).
    test_csv_path = '../output.csv'
    df = pd.read_csv(test_csv_path)
    if "Sequence" not in df.columns:
        print("Column 'Sequence' not found in the CSV.")
        return
    sequences = df["Sequence"].tolist()
    print(f"Loaded {len(sequences)} sequences from {test_csv_path}")

    # For evaluation by GO IDs, if the CSV contains a "go_ids" column,
    # extract a label for each protein. If multiple GO IDs are present (comma-separated),
    # here we take the first one.
    if "go_ids" in df.columns:
        labels = df["go_ids"].apply(
            lambda x: x.split(",")[0].strip() if isinstance(x, str) and x.strip() != "" else "NA"
        ).tolist()
    else:
        labels = None

    # Load ProteinBERT model and input encoder.
    model, input_encoder = load_proteinbert_model(seq_len=seq_len)
    print(f"Loaded ProteinBERT model for sequence length {seq_len}")

    # Extract global embeddings.
    global_embeddings = get_global_embeddings(model, input_encoder, sequences, seq_len=seq_len, batch_size=batch_size)
    print("Extracted global embeddings with shape:", global_embeddings.shape)

    # --- Dimensionality Reduction and Plotting using TSNE (2D) ---
    tsne_2d = TSNE(n_components=2, random_state=42)
    embeddings_2d_tsne = tsne_2d.fit_transform(global_embeddings)
    plot_2d_interactive(embeddings_2d_tsne, labels=labels,
                        title="2D TSNE of ProteinBERT Global Embeddings (by GO IDs)",
                        save_path="proteinbert_go_tsne_2d.html")

    # --- Compute Silhouette Score ---
    try:
        score = silhouette_score(embeddings_2d_tsne, labels)
        print(f"Silhouette score (GO IDs, TSNE): {score:.3f}")
    except Exception as e:
        print(f"Could not compute silhouette score for GO IDs (TSNE): {e}")


if __name__ == '__main__':
    main()