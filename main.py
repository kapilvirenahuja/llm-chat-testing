import pandas as pd

import os
from dotenv import load_dotenv

from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

import streamlit as st


def load_dataset(dataset_name: str = "dataset.csv"):
    """ 
    helper function to load dataset from the filesystem. this must be a CSV file
      Args:
        dataset_name: name of the dataset to load
        Returns:
            pandas dataframe
    """

    data_dir = "./data/"
    file_path = os.path.join(data_dir, dataset_name)
    df = pd.read_csv(file_path)
    return df


def create_chunks(dataset: pd.DataFrame, chunk_size: int, chunk_overlap: int) -> list:
    """ 
    helper function to create chunks from a dataset
      Args:
        dataset: dataset to chunk
        chunk_size: size of each chunk
        chunk_overlap: overlap between chunks
        Returns:
            list of chunks
    """

    text_chunks = DataFrameLoader(
        dataset, page_content_column = "body"
        ).load_and_split(
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = chunk_size,
                chunk_overlap = chunk_overlap,
                length_function = len
            )
        )

    # we add metadata to the chunks themselves to facilitate retreival
    for chunk in text_chunks:
        title = chunk.metadata["title"]
        description = chunk.metadata["description"]
        content = chunk.page_content
        url = chunk.metadata["url"]
        final_text = f"TITLE: {title}\DESCRIPTION: {description}\BODY: {content}\nURL: {url}"
        chunk.page_content = final_text

    return text_chunks


def get_vector_store(chunks: list) -> FAISS:
    """ 
    helper function to create a vector store from a list of chunks
      Args:
        chunks: list of chunks
        Returns:
            list of vector stores
    """

    embeddings = OpenAIEmbeddings()

    if not os.path.exists("./vector_store"):
        print ("Creating vector store")
        vector_store = FAISS.from_documents(
            chunks, embeddings
        )
        vector_store.save_local("./vector_store")
    else:
        print ("Loading vector store")
        vector_store = FAISS.load_local("./vector_store", embeddings)
        
    return vector_store


def get_conversational_chain(vector_store: FAISS, human_message: str, system_message: str) -> None:
    """ 
    helper function to get a conversational chain from a vector store
      Args:
        vector_store: vector store to search
        human_message: human message to search for
        system_message: system message to search for
        Returns:
            None
    """

    llm = ChatOpenAI(model = "gpt-4")
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vector_store.as_retriever(),
        memory = memory,
        combine_docs_chain_kwargs = {
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message
                ]
            ),
        },
    )
    return conversational_chain


def handle_style_and_responses(user_question: str) -> None:
    """
    Handle user input to create the chatbot conversation in Streamlit

    Args:
        user_question (str): User question
    """
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    human_style = "background-color: #e6f7ff; border-radius: 10px; padding: 10px; color: black;"
    chatbot_style = "background-color: #f9f9f9; border-radius: 10px; padding: 10px; color: black;"

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(
                f"<p style='text-align: right;'><b>User</b></p> <p style='text-align: right;{human_style}'> <i>{message.content}</i> </p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='text-align: left;'><b>Chatbot</b></p> <p style='text-align: left;{chatbot_style}'> <i>{message.content}</i> </p>",
                unsafe_allow_html=True,
            )


def main():
    load_dotenv()
    df = load_dataset()
    chunks = create_chunks(df, 1000, 0)

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """
        You are a chatbot tasked with responding to questions about the documentation of the LangChain library and project.

        Do not answer questions that are not about the LangChain library or project.

        Given a question, you should respond with the most relevant documentation page by following the relevant context below:\n
        {context}
        """
    )
    
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        """
        {question}
        """
    )

    if "vector_sector" not in st.session_state:
        st.session_state.vector_store = get_vector_store(chunks)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    st.set_page_config(
        page_title = "Innovation AI",
        page_icon = ":books:",
    )
    st.title = "Innovation AI"
    st.subheader = "A conversational AI to help you innovate"
    st.markdown(
        """
        This is a conversational AI that can help you innovate. It is powered by the LangChain library and project.
        """
    )
    st.image("https://images.unsplash.com/photo-1485827404703-89b55fcc595e")

    user_question = st.text_input("What do you want to innovate on?")
    with st.spinner("Processing..."):
        if user_question:
            handle_style_and_responses(user_question)

    st.session_state.conversation = get_conversational_chain(
        st.session_state.vector_store, system_message_prompt, human_message_prompt
    )



if __name__ == "__main__":
    main()


