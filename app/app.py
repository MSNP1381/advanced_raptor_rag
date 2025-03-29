from copy import deepcopy
import os
from typing import List
from langchain.prompts import ChatPromptTemplate
from utils import expand_query, rerank, retrieve_expanded_queries, contextualize_docs
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_postgres import PGVector
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
import vertexai
from langchain_core.documents import Document


load_dotenv(override=True)


st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("advanced Information Assistant")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.conversation = []

PROJECT_ID = "<my-prj>"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}


vertexai.init(project=PROJECT_ID, location=LOCATION)
embd = VertexAIEmbeddings(model_name="text-embedding-005")

# from langchain_openai import ChatOpenAI

# model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

model = ChatVertexAI(temperature=0, model="gemini-2.0-flash")

# Load data
# @st.cache_resource


# See docker command above to launch a postgres instance with pgvector enabled.
connection = os.environ["PG_CONN"]  # Uses psycopg3!
collection_name = "my_rag"


vectorstore = PGVector(
    embeddings=embd,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
# Now, use all_texts to build the vectorstore with Chroma
retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.5, "fetch_k": 50}
)
# Prompt
human_message = """\
<Question>
{question}
</Question>

<Context>
{context}
</Context>"""
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a friendly and knowledgeable virtual assistant. Your primary role is to answer questions accurately and concisely, *exclusively* using the information found in the provided documents.

Here's how you will operate:

*   **Data Source:** Your answers *must* be based solely on the retrieved documents wich is provided in <Context> tag. Do not use any external knowledge.
*   **Conciseness:** Keep your responses short and to the point, using a maximum of five sentences.
*   **Language:** Respond *exclusively* in French.
*   **Tone:** Maintain a polite, professional, and helpful tone, as if you were a customer service representative.
*   **Handling Unknowns:** If the answer is not found within the provided documents, respond with: "Je suis désolé, mais je ne trouve pas la réponse à votre question dans les documents fournis."
*   **Conversation History:** Use the previous turns in the conversation to ensure your responses are relevant and contextualized.
* **Easy Language**: use easy and lucid language so any user can get it without any doubt.
*   **Greeting (Conditional):** If it's the start of a new conversation, begin your response with a brief, friendly French greeting (e.g., "Bonjour !", "Bien sûr !", "Avec plaisir !").

Your overall goal is to provide clear, accurate, and helpful information from the documentation to its customers, in French.
""",
        ),
        # Add previous conversation turns
        ("placeholder", "{conversation_history}"),
        ("human", human_message),
    ]
)


# Post-processing
def format_docs(docs):
    formatted_docs = ""
    for doc in docs:
        if "url" in doc.metadata:
            appeded_txt = f"<title>{doc.metadata['title']}</title>\n<url>{doc.metadata['url']}</url>"
        else:
            appeded_txt = "<title>provided from summary not any source</title>"
        formatted_docs += f"""

<Context>
{appeded_txt}
<Document>
{doc.page_content}
</Document>
</Context>

"""
    return formatted_docs


def format_docs_ref(docs: List[Document]):
    output_txt = "\n\n\n"
    for index, doc in enumerate(docs):
        doc_metadata = doc.metadata
        if "url" in doc_metadata:
            # add #source url as reference in markdown fromat to list of sources
            output_txt += f"[source{index + 1}]({doc_metadata['url']}) | "
        else:
            output_txt += f"[source{index + 1}](#summary) | "
    return output_txt


# Chain
rag_chain = (
    # {"context":  | format_docs, "question": RunnablePassthrough()}
    # |
    prompt | model | StrOutputParser()
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            print(message["content"])

        st.markdown(message["content"])

# Get user input
if user_query_init := st.chat_input("Ask about la post..."):
    # Add user message to chat history
    if st.session_state.messages:
        user_query = contextualize_docs(
            llm=model,
            retriever=retriever,
            query=user_query_init,
            conversation=(st.session_state.conversation),
        )
    else:
        user_query = "" + user_query_init
        print(user_query)
    st.session_state.messages.append(
        {"role": "user", "content": deepcopy(user_query_init)}
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query_init)
    with st.chat_message("assistant"):
        with st.spinner("Expanding query..."):
            expanded_queries = expand_query(user_query, model)
        with st.spinner("Retrieving..."):
            retrieved = retrieve_expanded_queries(expanded_queries, retriever)
        with st.spinner("Reranking..."):
            reordered_docs = rerank(retrieved, user_query)
        with st.spinner("Thinking..."):
            llm_input = {
                "context": format_docs(reordered_docs),
                "question": user_query,
                "conversation_history": st.session_state.conversation,
            }

            response = rag_chain.invoke(llm_input)
            st.session_state.conversation.append(
                (
                    "human",
                    human_message.format(
                        question=user_query, context=format_docs(reordered_docs)
                    ),
                )
            )
        st.session_state.conversation.append(("ai", response))
    st.markdown(response + format_docs_ref(reordered_docs))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
