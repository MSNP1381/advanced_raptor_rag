import glob
import json
import tiktoken
from typing import Dict, List, Optional, Tuple
import logging
import vertexai
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
import numpy as np
import pandas as pd
import umap
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
import os

RANDOM_SEED = 224  # Fixed seed for reproducibility
summary_dict = {}
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("la_post_rag.log"), logging.StreamHandler()],
)
logger = logging.getLogger("la_post_RAG")

### --- Code from citations referenced above (added comments and docstrings) --- ###


PROJECT_ID = "research-and-development-wlc"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}


vertexai.init(project=PROJECT_ID, location=LOCATION)

print("Vertex AI initialized")

embd = VertexAIEmbeddings(model_name="text-embedding-005")

model = ChatVertexAI(temperature=0, model="gemini-2.0-flash-lite")
print("Model loaded")


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Perform global dimensionality reduction on the embeddings using UMAP.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - n_neighbors: Optional; the number of neighbors to consider for each point.
                   If not provided, it defaults to the square root of the number of embeddings.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    logger.info(f"Starting global dimensionality reduction with target dim={dim}")
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
        logger.debug(f"Using calculated n_neighbors={n_neighbors}")

    try:
        result = umap.UMAP(
            n_neighbors=n_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)
        logger.info(f"Global dimensionality reduction complete: {result.shape}")
        return result
    except Exception as e:
        logger.error(f"Error in global_cluster_embeddings: {str(e)}")
        raise


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - num_neighbors: The number of neighbors to consider for each point.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    logger.info(
        f"Starting local dimensionality reduction with dim={dim}, num_neighbors={num_neighbors}"
    )
    try:
        result = umap.UMAP(
            n_neighbors=num_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)
        logger.info(f"Local dimensionality reduction complete: {result.shape}")
        return result
    except Exception as e:
        logger.error(f"Error in local_cluster_embeddings: {str(e)}")
        raise


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - max_clusters: The maximum number of clusters to consider.
    - random_state: Seed for reproducibility.

    Returns:
    - An integer representing the optimal number of clusters found.
    """
    logger.info(f"Finding optimal number of clusters (max={max_clusters})")
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []

    for n in n_clusters:
        logger.debug(f"Testing cluster count: {n}")
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))

    optimal_clusters = n_clusters[np.argmin(bics)]
    logger.info(f"Optimal number of clusters determined: {optimal_clusters}")
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - threshold: The probability threshold for assigning an embedding to a cluster.
    - random_state: Seed for reproducibility.

    Returns:
    - A tuple containing the cluster labels and the number of clusters determined.
    """
    logger.info(f"Starting GMM clustering with threshold={threshold}")
    n_clusters = get_optimal_clusters(embeddings)
    logger.info(f"Using {n_clusters} clusters for GMM")

    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]

    # Log distribution of assignments
    cluster_counts = [len(label) for label in labels]
    avg_clusters_per_item = sum(cluster_counts) / len(labels)
    logger.info(
        f"GMM clustering complete. Avg clusters per item: {avg_clusters_per_item:.2f}"
    )
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """
    Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
    using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for UMAP reduction.
    - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

    Returns:
    - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
    """
    logger.info(
        f"Starting hierarchical clustering process with {len(embeddings)} embeddings"
    )

    if len(embeddings) <= dim + 1:
        logger.warning(
            f"Insufficient data for clustering ({len(embeddings)} <= {dim + 1}). Assigning all to cluster 0."
        )
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)

    # Global clustering
    logger.info("Starting global clustering")
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )
    logger.info(f"Found {n_global_clusters} global clusters")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        logger.info(f"Processing global cluster {i + 1}/{n_global_clusters}")
        # Extract embeddings belonging to the current global cluster
        global_cluster_indices = np.array([i in gc for gc in global_clusters])
        global_cluster_embeddings_ = embeddings[global_cluster_indices]

        logger.debug(
            f"Global cluster {i} contains {len(global_cluster_embeddings_)} embeddings"
        )

        if len(global_cluster_embeddings_) == 0:
            logger.warning(f"Global cluster {i} is empty, skipping")
            continue

        if len(global_cluster_embeddings_) <= dim + 1:
            # Handle small clusters with direct assignment
            logger.debug(
                f"Global cluster {i} too small, assigning all to local cluster 0"
            )
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Local dimensionality reduction and clustering
            logger.debug(f"Performing local clustering within global cluster {i}")
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )
            logger.debug(
                f"Found {n_local_clusters} local clusters in global cluster {i}"
            )

        # Assign local cluster IDs, adjusting for total clusters already processed
        for j in range(n_local_clusters):
            local_cluster_indices = np.array([j in lc for lc in local_clusters])
            local_cluster_embeddings_ = global_cluster_embeddings_[
                local_cluster_indices
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    logger.info(f"Clustering complete. Total clusters: {total_clusters}")
    return all_local_clusters


### --- Our code below --- ###


def load_and_consolidate_embeddings(directory="embd_out"):
    """
    Loads all embedding batch files from the specified directory and
    consolidates them into a single numpy array.

    Parameters:
    - directory: The directory where embedding batch files are stored

    Returns:
    - A single numpy array containing all embeddings
    """
    logger.info(f"Loading and consolidating embeddings from {directory}")

    try:
        # Find all .npy files in the directory
        batch_files = glob.glob(os.path.join(directory, "batch_*.npy"))

        # Check if there's a consolidated file already
        all_embeddings_path = os.path.join(directory, "all_embeddings.npy")
        if os.path.exists(all_embeddings_path):
            logger.info(f"Found consolidated embeddings file: {all_embeddings_path}")
            batch_files = [f for f in batch_files if f != all_embeddings_path]

            # Load the existing consolidated embeddings
            all_embeddings = np.load(all_embeddings_path)
            logger.info(
                f"Loaded existing consolidated embeddings with shape {all_embeddings.shape}"
            )

            # Load and append batch files if any
            if batch_files:
                logger.info(
                    f"Found {len(batch_files)} additional batch files to append"
                )
                for batch_file in sorted(
                    batch_files,
                    key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
                ):
                    logger.info(f"Loading batch file: {batch_file}")
                    batch_embeddings = np.load(batch_file)
                    all_embeddings = np.vstack((all_embeddings, batch_embeddings))

                # Save the updated consolidated embeddings
                np.save(all_embeddings_path, all_embeddings)
                logger.info(
                    f"Updated consolidated embeddings saved to {all_embeddings_path}"
                )
        else:
            if not batch_files:
                logger.warning(f"No embedding files found in {directory}")
                return None

            # Sort the batch files by their batch number
            batch_files.sort(
                key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
            )

            # Load the first batch to get dimensions
            logger.info(f"Loading first batch: {batch_files[0]}")
            all_embeddings = np.load(batch_files[0])

            # Load and append the rest of the batches
            for batch_file in batch_files[1:]:
                logger.info(f"Loading batch file: {batch_file}")
                batch_embeddings = np.load(batch_file)
                all_embeddings = np.vstack((all_embeddings, batch_embeddings))

            # Save the consolidated embeddings
            np.save(all_embeddings_path, all_embeddings)
            logger.info(f"Consolidated embeddings saved to {all_embeddings_path}")

        logger.info(f"Consolidation complete. Final shape: {all_embeddings.shape}")
        return all_embeddings

    except Exception as e:
        logger.error(f"Error consolidating embeddings: {str(e)}")
        raise


async def embed(texts, max_tokens_per_request=1_000_000, batch_size=512):
    """
    Generate embeddings for a list of text documents.

    This function assumes the existence of an `embd` object with a method `embed_documents`
    that takes a list of texts and returns their embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be embedded.
    - max_tokens_per_request: Maximum number of tokens to process in a single request
    - batch_size: Number of documents to process in each batch

    Returns:
    - numpy.ndarray: An array of embeddings for the given text documents.
    """
    logger.info(f"Generating embeddings for {len(texts)} documents")

    if os.path.exists(f"embd_{len(texts)}_out/all_embeddings.npy"):
        logger.info("Found existing embeddings file, loading...")
        loaded_data = np.load(f"embd_{len(texts)}_out/all_embeddings.npy")
        if len(loaded_data) == len(texts):
            logger.info("Embeddings already exist for all documents, loading...")
            return loaded_data
        else:
            logger.info("Embeddings exist but not for all documents, regenerating...")

    try:
        # Create output directory if it doesn't exist
        os.makedirs(f"embd_{len(texts)}_out", exist_ok=True)

        # Process in batches to handle large volumes of text
        logger.info(f"Processing {len(texts)} documents for embedding")

        # Initialize list to store all embeddings
        all_embeddings = []

        # Process in batches of specified size
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts) + 1, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_filename = f"embd_{len(texts)}_out/batch_{i // batch_size}.npy"

            # Check if this batch already exists

            logger.info(
                f"Embedding batch {i // batch_size + 1}/{total_batches} ({len(batch_texts)} documents)"
            )

            # Rough estimate: 1 token ≈ 4 characters
            estimated_tokens = sum(
                [num_tokens_from_string(i, "cl100k_base") for i in batch_texts]
            )

            if estimated_tokens > max_tokens_per_request:
                logger.warning(
                    f"Batch size may exceed token limit. Estimated tokens: {estimated_tokens}"
                )
                # Process in smaller chunks if needed
                sub_batch_size = max(
                    1, int(batch_size * max_tokens_per_request / estimated_tokens)
                )
                logger.info(
                    f"Reducing sub-batch size to approximately {sub_batch_size} documents"
                )

                batch_embeddings = []
                for j in range(0, len(batch_texts), sub_batch_size):
                    sub_batch = batch_texts[j : j + sub_batch_size]
                    logger.info(
                        f"Processing sub-batch {j // sub_batch_size + 1}/{(len(batch_texts) + sub_batch_size - 1) // sub_batch_size}"
                    )
                    sub_embeddings = await embd.aembed_documents(sub_batch)
                    batch_embeddings.extend(sub_embeddings)

                batch_embeddings = np.array(batch_embeddings)
            else:
                # Process the whole batch at once
                batch_embeddings = await embd.aembed_documents(batch_texts)

            # Save this batch's embeddings
            np.save(batch_filename, batch_embeddings)
            logger.info(f"Saved batch embeddings to {batch_filename}")

            all_embeddings.extend(batch_embeddings)

        # Convert to numpy array
        embeddings_np = np.array(all_embeddings)

        # Save the consolidated embeddings
        np.save(f"embd_{len(texts)}_out/all_embeddings.npy", embeddings_np)
        logger.info(
            f"All embeddings consolidated and saved. Shape: {embeddings_np.shape}"
        )

        return embeddings_np
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise


async def embed_cluster_texts(texts):
    """
    Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

    This function combines embedding generation and clustering into a single step. It assumes the existence
    of a previously defined `perform_clustering` function that performs clustering on the embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be processed.

    Returns:
    - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
    """
    logger.info(f"Starting embed_cluster_texts for {len(texts)} documents")

    # Generate embeddings
    text_embeddings_np = await embed(texts)
    logger.info("Embeddings generated, proceeding to clustering")

    # Perform clustering on the embeddings
    cluster_labels = perform_clustering(text_embeddings_np, 10, 0.1)

    # Create and populate DataFrame
    logger.info("Creating results DataFrame")
    df = pd.DataFrame()
    df["text"] = texts
    df["embd"] = list(text_embeddings_np)
    df["cluster"] = cluster_labels

    # Log some statistics about the clustering
    cluster_counts = [len(labels) for labels in cluster_labels]
    logger.info(
        f"Clustering stats: Avg clusters per document: {sum(cluster_counts) / len(cluster_counts):.2f}"
    )
    logger.info(
        f"Documents with no clusters: {sum(1 for c in cluster_counts if c == 0)}"
    )

    return df


def fmt_txt(df: pd.DataFrame) -> str:
    """
    Formats the text documents in a DataFrame into a single string.

    Parameters:
    - df: DataFrame containing the 'text' column with text documents to format.

    Returns:
    - A single string where all text documents are joined by a specific delimiter.
    """
    logger.debug(f"Formatting {len(df)} text documents")
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)


async def embed_cluster_summarize_texts(
    texts: List[str], level: int, batch_size=80
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
    clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
    the content within each cluster.

    Parameters:
    - texts: A list of text documents to be processed.
    - level: An integer parameter that could define the depth or detail of processing.

    Returns:
    - Tuple containing two DataFrames:
      1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
      2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
         and the cluster identifiers.
    """
    logger.info(
        f"Starting embed_cluster_summarize_texts with {len(texts)} documents at level {level}"
    )

    # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
    df_clusters = await embed_cluster_texts(texts)

    # Prepare to expand the DataFrame for easier manipulation of clusters
    logger.info("Expanding cluster assignments")
    expanded_list = []

    # Expand DataFrame entries to document-cluster pairings for straightforward processing
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    # Create a new DataFrame from the expanded list
    expanded_df = pd.DataFrame(expanded_list)

    # Retrieve unique cluster identifiers for processing
    all_clusters = expanded_df["cluster"].unique()

    logger.info(f"--Generated {len(all_clusters)} clusters--")

    # Summarization
    template = """You are a world class summarization expert. Please provide a concise and informative summary of the following text, extracting the key points and main ideas.
Text:
{context}
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()

    # Format text within each cluster for summarization
    logger.info("Starting summarization of clusters")
    summaries = []
    # Configurable batch size
    formatted_texts = []
    cluster_ids = []

    # Prepare data for batch processing
    for i, cluster_id in enumerate(all_clusters):
        logger.info(f"Preparing cluster {i + 1}/{len(all_clusters)} (ID: {cluster_id})")
        df_cluster = expanded_df[expanded_df["cluster"] == cluster_id]
        logger.debug(f"Cluster {cluster_id} contains {len(df_cluster)} documents")

        formatted_txt = fmt_txt(df_cluster)
        formatted_texts.append(formatted_txt)
        cluster_ids.append(cluster_id)

    # Process in batches
    for batch_start in range(0, len(formatted_texts), batch_size):
        batch_end = min(batch_start + batch_size, len(formatted_texts))
        batch_texts = formatted_texts[batch_start:batch_end]
        batch_cluster_ids = cluster_ids[batch_start:batch_end]

        logger.info(
            f"Processing batch {batch_start // batch_size + 1}/{(len(formatted_texts) + batch_size - 1) // batch_size}"
        )

        # Filter out texts that are already in the summary_dict
        new_batch_texts = []
        new_batch_indices = []

        for i, text in enumerate(batch_texts):
            if text not in summary_dict:
                new_batch_texts.append(text)
                new_batch_indices.append(i)

        if new_batch_texts:
            try:
                # Use batch invocation
                batch_inputs = [{"context": text} for text in new_batch_texts]
                batch_results = chain.batch(batch_inputs)

                # Store results
                for i, result in enumerate(batch_results):
                    original_index = new_batch_indices[i]
                    original_text = batch_texts[original_index]
                    summary_dict[original_text] = result
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                # Handle batch failure by processing individually
                for i in new_batch_indices:
                    text = batch_texts[i]
                    cluster_id = batch_cluster_ids[i]
                    try:
                        result = await chain.ainvoke({"context": text})
                        summary_dict[text] = result
                    except Exception as inner_e:
                        logger.error(
                            f"Error generating summary for cluster {cluster_id}: {str(inner_e)}"
                        )
                        summary_dict[text] = f"Error generating summary: {str(inner_e)}"

        # Collect summaries for this batch
        for i in range(batch_start, batch_end):
            text = formatted_texts[i]
            summary = summary_dict[text]
            summaries.append(summary)
            logger.debug(
                f"Summary for cluster {cluster_ids[i]} has length {len(summary)}"
            )
        # Create a DataFrame to store summaries with their corresponding cluster and level
    logger.info("Creating summary DataFrame")
    with open("summaries_dict.json", "w") as f:
        json.dump(
            {
                "summaries": summaries,
                "level": [level] * len(summaries),
                "cluster": list(all_clusters),
            },
            f,
        )
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary


async def recursive_embed_cluster_summarize(
    texts: List[str], level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Recursively embeds, clusters, and summarizes texts up to a specified level or until
    the number of unique clusters becomes 1, storing the results at each level.

    Parameters:
    - texts: List[str], texts to be processed.
    - level: int, current recursion level (starts at 1).
    - n_levels: int, maximum depth of recursion.

    Returns:
    - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
      levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
    """
    logger.info(
        f"Starting recursive processing at level {level}/{n_levels} with {len(texts)} documents"
    )
    results = {}  # Dictionary to store results at each level

    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary = await embed_cluster_summarize_texts(texts, level)

    # Store the results of the current level
    results[level] = (df_clusters, df_summary)

    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary["cluster"].nunique()
    logger.info(f"Level {level} produced {unique_clusters} unique clusters")

    if level < n_levels and unique_clusters > 1:
        logger.info(
            f"Proceeding to level {level + 1} with {len(df_summary['summaries'])} summaries"
        )
        # Use summaries as the input texts for the next level of recursion
        new_texts = df_summary["summaries"].tolist()
        next_level_results = await recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )

        # Merge the results from the next level into the current results dictionary
        results.update(next_level_results)
    else:
        if unique_clusters <= 1:
            logger.info(
                f"Stopping recursion at level {level}: Only {unique_clusters} clusters found"
            )
        else:
            logger.info(f"Stopping recursion at level {level}: Reached maximum depth")

    return results


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
