#!/usr/bin/env python3
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE
import umap.umap_ as umap
import plotly.express as px  # For interactive plots

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
        # Use the encoder on a single-sequence list.
        encoded_x, encoded_annotations = input_encoder.encode_X([seq], seq_len)
        # encoded_x[0] should be the token list; convert it to a Python list.
        tokens = encoded_x[0].tolist()  # Removed extra indexing
        tokens_fixed = force_fixed_length(tokens, seq_len, pad_token)
        encoded_list.append(tokens_fixed)
        # For annotations, we assume they are already of length seq_len.
        annotations_list.append(encoded_annotations[0])

    encoded_x_fixed = np.array(encoded_list, dtype=np.int32)
    encoded_annotations_fixed = np.array(annotations_list, dtype=np.int32)
    return encoded_x_fixed, encoded_annotations_fixed


def get_global_embeddings(model, input_encoder, sequences, seq_len=512, batch_size=32):
    """
    Given a list of protein sequences, first pre-truncate each sequence to seq_len characters,
    then encode them using our custom fixed-length encoder, and finally extract
    global representations from the model.

    Returns:
        global_embeddings: numpy array of shape (N, embedding_dim)
    """
    # Pre-truncate each sequence to at most seq_len characters.
    sequences_truncated = [seq[:seq_len] for seq in sequences]
    # Use our custom encoder to force fixed-length tokenization.
    encoded_x_fixed, encoded_annotations_fixed = encode_sequences_fixed(input_encoder, sequences_truncated, seq_len)
    # Run the model's predict method.
    # According to ProteinBERT documentation, predict returns a tuple: (local_reps, global_reps)
    _, global_representations = model.predict((encoded_x_fixed, encoded_annotations_fixed), batch_size=batch_size)
    return global_representations


def plot_2d_interactive(embeddings_2d, title, save_path=None):
    """
    Create and (optionally) save an interactive 2D scatter plot using Plotly.
    """
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


def plot_3d_interactive(embeddings_3d, title, save_path=None):
    """
    Create and (optionally) save an interactive 3D scatter plot using Plotly.
    """
    df_plot = pd.DataFrame({
        'Component 1': embeddings_3d[:, 0],
        'Component 2': embeddings_3d[:, 1],
        'Component 3': embeddings_3d[:, 2]
    })
    fig = px.scatter_3d(df_plot, x='Component 1', y='Component 2', z='Component 3',
                        title=title,
                        color_discrete_sequence=px.colors.qualitative.Alphabet)
    fig.show()
    if save_path:
        fig.write_html(save_path)
        print(f"3D plot saved to {save_path}")


def main():
    set_seed(42)
    seq_len = 512
    batch_size = 32

    # Load the test CSV file (expects a "Sequence" column)
    test_csv_path = '../output.csv'
    df = pd.read_csv(test_csv_path)
    if "Sequence" not in df.columns:
        print("Column 'Sequence' not found in the CSV.")
        return
    sequences = df["Sequence"].tolist()
    print(f"Loaded {len(sequences)} sequences from {test_csv_path}")

    # Load ProteinBERT model and input encoder.
    model, input_encoder = load_proteinbert_model(seq_len=seq_len)
    print(f"Loaded ProteinBERT model for sequence length {seq_len}")

    # Extract global embeddings for all proteins.
    global_embeddings = get_global_embeddings(model, input_encoder, sequences, seq_len=seq_len, batch_size=batch_size)
    print("Extracted global embeddings with shape:", global_embeddings.shape)

    # --- Dimensionality Reduction and Plotting ---
    # TSNE: 2D
    tsne_2d = TSNE(n_components=2, random_state=42)
    embeddings_2d_tsne = tsne_2d.fit_transform(global_embeddings)
    plot_2d_interactive(embeddings_2d_tsne, title="2D TSNE of ProteinBERT Global Embeddings",
                        save_path="../plots/proteinbert_tsne_2d.html")

    # TSNE: 3D
    tsne_3d = TSNE(n_components=3, random_state=42)
    embeddings_3d_tsne = tsne_3d.fit_transform(global_embeddings)
    plot_3d_interactive(embeddings_3d_tsne, title="3D TSNE of ProteinBERT Global Embeddings",
                        save_path="../plots/proteinbert_tsne_3d.html")

    # UMAP: 2D
    umap_2d = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d_umap = umap_2d.fit_transform(global_embeddings)
    plot_2d_interactive(embeddings_2d_umap, title="2D UMAP of ProteinBERT Global Embeddings",
                        save_path="../plots/proteinbert_umap_2d.html")

    # UMAP: 3D
    umap_3d = umap.UMAP(n_components=3, random_state=42)
    embeddings_3d_umap = umap_3d.fit_transform(global_embeddings)
    plot_3d_interactive(embeddings_3d_umap, title="3D UMAP of ProteinBERT Global Embeddings",
                        save_path="../plots/proteinbert_umap_3d.html")


if __name__ == '__main__':
    main()