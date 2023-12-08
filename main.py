import pandas as pd

from langchain.vectorstores import FAISS

def load_dataset(dataset_name: str = "dataset.cvs"):
    """ 
    helper function to load dataset from the filesystem. this must be a CSV file
      Args:
        dataset_name: name of the dataset to load
        Returns:
            pandas dataframe
    """
    pass


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
    pass

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
    pass

if __name__ == "__main__":
    main()


