#!/usr/bin/env python3
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import plotly.express as px  # For interactive plots
import distinctipy          # For generating distinct color palettes
import umap.umap_ as umap   # For UMAP dimensionality reduction (optional)

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
    Given a list of protein sequences, truncate each sequence to seq_len tokens,
    then encode them using ProteinBERT's fixed-length encoder and extract
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
    """
    colors = distinctipy.get_colors(num_colors)
    colors_hex = [distinctipy.get_hex(color) for color in colors]
    return colors_hex


def plot_2d_interactive(embeddings_2d, labels, title, unique_label_name, save_path):
    """
    Create an interactive 2D scatter plot using Plotly Express.

    Parameters:
      embeddings_2d: (N,2) numpy array of 2D coordinates.
      labels: list or array of label values.
      title: Title for the plot.
      unique_label_name: Name of the label (e.g., "Phylum").
      save_path: File path to save the interactive plot as HTML.
    """
    df_plot = pd.DataFrame({
        'Component 1': embeddings_2d[:, 0],
        'Component 2': embeddings_2d[:, 1],
        unique_label_name: labels
    })
    unique_groups = sorted(df_plot[unique_label_name].unique())
    color_palette = generate_distinct_color_palette(len(unique_groups))
    color_map = {group: color for group, color in zip(unique_groups, color_palette)}

    fig = px.scatter(df_plot, x='Component 1', y='Component 2', title=title,
                     color=unique_label_name, color_discrete_map=color_map)
    fig.show()
    fig.write_html(save_path)
    print(f"Plot saved to {save_path}")


def main():
    set_seed(42)
    seq_len = 512
    batch_size = 32

    # Load the test CSV file.
    # The CSV is expected to have a "Sequence" column and a "supfam_ids" column.
    test_csv_path = 'output.csv'
    df = pd.read_csv(test_csv_path)
    if "Sequence" not in df.columns:
        print("Column 'Sequence' not found in the CSV.")
        return

    # Filter to entries with non-empty supfam_ids.
    if "supfam_ids" in df.columns:
        df_supfam = df[df["supfam_ids"].notnull() & (df["supfam_ids"].str.strip() != "")]
        # Split comma-separated supfam IDs and explode the list.
        df_supfam = df_supfam.copy()
        df_supfam["supfam_ids"] = df_supfam["supfam_ids"].apply(lambda x: [s.strip() for s in x.split(",")])
        df_supfam = df_supfam.explode("supfam_ids").reset_index(drop=True)
        # Find the most frequent SUPFAM.
        most_common_supfam = df_supfam["supfam_ids"].value_counts().idxmax()
        print(f"Most frequent SUPFAM: {most_common_supfam}")
        # Filter the dataset for entries that contain the most common SUPFAM.
        df_filtered = df_supfam[df_supfam["supfam_ids"] == most_common_supfam].copy()
        if df_filtered.empty:
            print("No entries found for the most frequent SUPFAM.")
            return
        # Use the "Sequence" and "phylum" columns from the filtered dataset.
        sequences = df_filtered["Sequence"].tolist()
        if "phylum" in df_filtered.columns:
            labels = df_filtered["phylum"].tolist()
            label_name = "Phylum"
        else:
            labels = ["NA"] * len(sequences)
            label_name = "Label"
    else:
        print("Column 'supfam_ids' not found in the CSV. Using entire dataset.")
        sequences = df["Sequence"].tolist()
        labels = ["NA"] * len(sequences)
        label_name = "Label"

    print(f"Using {len(sequences)} sequences for analysis after filtering by SUPFAM.")

    # Load the ProteinBERT model and input encoder.
    model, input_encoder = load_proteinbert_model(seq_len=seq_len)
    print("Loaded ProteinBERT model for sequence length", seq_len)

    # Extract global embeddings for the filtered sequences.
    global_embeddings = get_global_embeddings(model, input_encoder, sequences, seq_len=seq_len, batch_size=batch_size)
    print("Extracted global embeddings with shape:", global_embeddings.shape)

    # --- TSNE Visualization ---
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d_tsne = tsne.fit_transform(global_embeddings)
    plot_2d_interactive(
        embeddings_2d_tsne,
        labels,
        title=f"2D t-SNE of ProteinBERT Global Embeddings\n(Most frequent SUPFAM: {most_common_supfam}, colored by {label_name})",
        unique_label_name=label_name,
        save_path="proteinbert_tsne_supfam_by_phylum.html"
    )
    try:
        score_tsne = silhouette_score(embeddings_2d_tsne, labels)
        print(f"Silhouette score (TSNE, {label_name}): {score_tsne:.3f}")
    except Exception as e:
        print(f"Could not compute silhouette score for TSNE: {e}")

    # Uncomment below to also run UMAP visualization.
    # umap_model = umap.UMAP(n_components=2, random_state=42)
    # embeddings_2d_umap = umap_model.fit_transform(global_embeddings)
    # plot_2d_interactive(
    #     embeddings_2d_umap,
    #     labels,
    #     title=f"2D UMAP of ProteinBERT Global Embeddings\n(Most frequent SUPFAM: {most_common_supfam}, colored by {label_name})",
    #     unique_label_name=label_name,
    #     save_path="proteinbert_umap_supfam_by_phylum.html"
    # )
    # try:
    #     score_umap = silhouette_score(embeddings_2d_umap, labels)
    #     print(f"Silhouette score (UMAP, {label_name}): {score_umap:.3f}")
    # except Exception as e:
    #     print(f"Could not compute silhouette score for UMAP: {e}")


if __name__ == '__main__':
    main()