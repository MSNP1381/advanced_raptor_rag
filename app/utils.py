from langchain.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_transformers import LongContextReorder
from langchain.chains.history_aware_retriever import (
    create_history_aware_retriever,
)  # Import the create_history_aware_retriever function
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)  # Import the ChatPromptTemplate and MessagesPlaceholder classes
from langchain_core.output_parsers import (
    StrOutputParser,
)  # Import the StrOutputParser class
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class LineList(BaseModel):
    lines: list[str] = Field(description="expanded queries")


def expand_query(
    query: str,
    model: ChatVertexAI,
) -> LineList:

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question", "institute_name"],
        template=(
            """You are a query optimization assistant for information retrieval. Your task is to improve the chances of finding relevant documents by generating *three distinct variations* of a user's original question. These variations should aim to:

*   **Rephrase:** Express the same basic question using different words and sentence structures.
*   **Expand:** Add related terms, synonyms, or closely associated concepts that the user *might* have meant, even if they didn't explicitly mention them. *Be careful* not to add irrelevant information.
*   **Consider different aspects:** Think about the question from different angles, such as customer needs, product features, or common issues.


Output format: Provide each alternative query on a *new line*. Do not include *any* numbering or labels. Do *not* include the original question in your output.

Original question: {question}"""
        ),
    )
    model.temperature = 0.1

    llm_chain = QUERY_PROMPT | model.with_structured_output(LineList)
    queries: LineList = llm_chain.invoke(query)
    return queries.lines


def retrieve_expanded_queries(queries, retriever: VectorStoreRetriever):
    print("\n\n\n--------\n\n\n")
    docs = [retriever.invoke(query) for query in queries]
    docs_dict = {}
    unique_contents = set()
    unique_docs = []
    for sublist in docs:
        for doc in sublist:
            if doc.page_content not in unique_contents:
                unique_docs.append(doc)
                unique_contents.add(doc.page_content)
    unique_contents = list(unique_contents)
    return unique_docs


def rerank(unique_contents: list[Document], query):

    pairs = []
    for doc in unique_contents:
        pairs.append([query, doc.page_content])
    scores = cross_encoder.predict(pairs)

    scored_docs = zip(scores, unique_contents)
    sorted_docs = sorted(scored_docs, reverse=True)
    reranked_docs = [doc for _, doc in sorted_docs][0:8]
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(reranked_docs)
    return reordered_docs


def contextualize_docs(
    llm: ChatVertexAI, retriever: VectorStoreRetriever, query, conversation
):

    # Define the system prompt for contextualizing the question
    contextualize_q_system_prompt = """You are a contextualization assistant for La Banque Postale's question-answering system. Your task is to take a follow-up question from a user and the history of the previous conversation, and rephrase the follow-up question into a single, standalone question that includes all the necessary context.

Here's what you need to do:

*   **Understand the conversation:** Carefully analyze the conversation history to grasp the topic and all important details.
*   **Incorporate context:** Integrate relevant information from the conversation history *directly into* the rephrased question. The new question should be understandable *without* needing to see the previous conversation.
*   **Maintain clarity:** The rephrased question should be clear, concise, and easy to understand. Use language appropriate for La Banque Postale customers.
*   **Focus on information needs:** Ensure that the rephrased question accurately reflects the user's *underlying information need*, even if their original wording was ambiguous.
*   **Be autonomous:** the autonomous question should not contain words such as "it", "that", "this", "they", etc., and must contain all the entities for an autonomous question.

"""

    # Create a ChatPromptTemplate for contextualizing the question
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),  # Set the system prompt
            MessagesPlaceholder("chat_history"),  # Placeholder for the chat history
            (
                "human",
                "Entrée de suivi : {input}\nQuestion autonome reformulée :",
            ),  # Placeholder for the user's input question
        ]
    )
    chain = contextualize_q_prompt | llm | StrOutputParser()
    result = chain.invoke({"chat_history": conversation, "input": query})
    return result
