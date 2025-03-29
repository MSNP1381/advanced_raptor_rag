import os
import pickle
from dotenv import load_dotenv, find_dotenv
import vertexai

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_postgres import PGVector
from tqdm import tqdm

load_dotenv(find_dotenv(), override=True)

# %%
PROJECT_ID = "<my-prj>"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}


vertexai.init(project=PROJECT_ID, location=LOCATION)

# %%


embd = VertexAIEmbeddings(model_name="text-embedding-005")


# %%


# See docker command above to launch a postgres instance with pgvector enabled.
connection = os.environ["PG_CONN"]
collection_name = "my_rag"


vectorstore = PGVector(
    embeddings=embd,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
# Now, use all_texts to build the vectorstore with Chroma
retriever = vectorstore.as_retriever()


# %%


def reset_db(vectorstore):
    vectorstore.delete_collection()

    vectorstore.drop_tables()

    vectorstore.create_tables_if_not_exists()

    vectorstore.create_collection()


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


# %%

# reset_db()

# %%
all_texts = load_results("all_texts.pickle")

# %%

batch_size = 1000  # Adjust based on your vector dimensions
for i in tqdm(range(0, len(all_texts), batch_size)):
    batch = all_texts[i : i + batch_size]
    vectorstore.add_documents(batch)
