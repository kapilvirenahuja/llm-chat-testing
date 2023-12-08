import pandas as pd
import os

from langchain.vectorstores import FAISS

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


from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



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
    pass

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
    pass

def main():
    dataset = load_dataset()
    chunks = create_chunks(dataset, 1000, 0)
    pass

if __name__ == "__main__":
    main()


