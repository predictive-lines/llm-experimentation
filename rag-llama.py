import os
from pathlib import Path
import pandas as pd  # Import pandas for reading Excel files
from pinecone import Pinecone, PodSpec
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def process_directory(directory_path):
    documents = []
    for file_path in directory_path.glob('*'):
        if file_path.suffix == '.xlsx':
            documents.extend(process_excel_file(file_path))
        elif file_path.suffix in ['.pdf', '.docx', '.pptx']:
            documents.extend(process_other_files(file_path))
    return documents

def process_excel_file(file_path):
    df = pd.read_excel(file_path)
    records = df.to_dict(orient='records')
    documents = [{'text': str(record)} for record in records]
    return documents

def process_other_files(file_path):
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    return documents

def main():
    documents_folder = Path('C:/Users/JustinMiller/OneDrive - Predictive Lines/Internal/2023/resources')
    documents = process_directory(documents_folder)
    # This can be done by exporting the token in your terminal or command prompt using commands like export HUGGING_FACE_TOKEN=your_actual_token_here for Linux/Mac or set HUGGING_FACE_TOKEN=your_actual_token_here for Windows.
    # HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN", '<HUGGINGFACE_TOKEN>')
    HF_TOKEN = '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'
    remotely_run = HuggingFaceInferenceAPI(
        model_name='HuggingFaceH4/zephyr-7b-alpha', token=HF_TOKEN
    )
    embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')
    Settings.embed_model = embed_model
    Settings.llm = remotely_run

    api_key = os.environ.get('PINECONE_API_KEY', '43644636-3158-4b02-8741-5f6f2913d493')
    pc = Pinecone(api_key=api_key)
    index_name = "quickstart"
    # check if index_name exists on pinecone before creating
    if index_name not in pc.list_indexes().names():
        print("Index not found, creating.")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='euclidean',
            spec=PodSpec(environment='gcp-starter', pod_type='s1.x1'),
        )
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    query_engine = index.as_query_engine()
    response = query_engine.query("How to implement a process change at a service-based company that does not currently have a change management process?")
    print(response)

if __name__ == "__main__":
    main()