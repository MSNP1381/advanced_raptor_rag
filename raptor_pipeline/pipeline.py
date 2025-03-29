#%%
import matplotlib.pyplot as plt
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import DirectoryLoader
from raptor import *
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
import vertexai
import asyncio
from dotenv import load_dotenv,find_dotenv
from langchain_google_vertexai import VertexAIEmbeddings,ChatVertexAI
load_dotenv(find_dotenv(),override=True)

#%%
#######################
#
#   load Documents
#
######################

#you can add title and url to document metadata for usage in rag app
loader = DirectoryLoader(
    "./markdown_output/",
    glob="**/*.md",
    show_progress=True,
    max_concurrency=-1,
    silent_errors=True,
    loader_cls=UnstructuredMarkdownLoader,
)
docs = loader.load()
print(len(docs))
with open("loaded_docs.pickle", "wb") as f:
    pickle.dump(docs, f)
# with open("loaded_docs.pickle", "rb") as f:
#     docs=pickle.load(f)
# # Doc texts
docs_texts = [d.page_content for d in docs]
print("Number of documents:", len(docs_texts))

#%%

#######################
#
#   visualize token counts
#
######################

counts = [num_tokens_from_string(d, "cl100k_base") for d in docs_texts]

plt.figure(figsize=(10, 6))
plt.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)
plt.title("Histogram of Token Counts")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)

plt.savefig("token_counts_histogram.png")
#%%


concatenated_content = "\n\n\n --- \n\n\n".join([doc.page_content for doc in docs])

chunk_size_tok = 3600 # choose based on visualization

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=chunk_size_tok, chunk_overlap=0
)
texts_split = text_splitter.split_text(concatenated_content)

with open("splited_txt.pickle", "wb") as f:
  texts_split  =pickle.dump( texts_split,f)

# with open("splited_txt.pickle", "rb") as f:
#   texts_split  =pickle.load( f)



print("split done")
#%%
#######################
#
#   run main process
#
######################


leaf_texts = texts_split
# For large document sets, we should:
# 1. Use higher dimensionality reduction for better cluster separation
# 2. Adjust threshold for more precise clusters
# 3. Set appropriate recursion levels for hierarchical summarization

# Note: these are in raptor module
# in embed(texts, max_tokens_per_request=1_000_000, batch_size=512) i have implemented a max_tokens_per_request parameter to avoid the error of exceeding the token limit.
# You can set it to a value that is less than the max token limit of your model.

# in embed_cluster_summarize_texts( texts: List[str], level: int, batch_size=80) i have added a batching to avoid the error of exceeding the token limit.


# Configure and run the recursive embedding and clustering
loop = asyncio.get_event_loop()
results = loop.run_until_complete(recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3))



def save_results(results, filename):
    """Saves the results to a pickle file.

    Args:
        results: The results to save.
        filename: The name of the file to save to.
    """
    with open(filename, "wb") as f:
        pickle.dump(results, f)


def load_results(filename):
    """Loads the results from a pickle file.

    Args:
        filename: The name of the file to load from.

    Returns:
        The loaded results.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)
save_results(results, "results.pickle")
