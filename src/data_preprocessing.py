import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

def markdown_splitter(data):
    md_splits =  splitter.split_text(data.page_content)
    splits = r_splitter.split_documents(md_splits)
    for split in splits:
        split.metadata['source'] = data.metadata['source']
    return splits

def load_lecture_notes(directory):
    splitted_notes = []
    for filename in os.listdir(directory):
        loader = TextLoader(os.path.join(directory, filename))
        data = loader.load()
        for split in markdown_splitter(data[0]):
            splitted_notes.append(split)
    return filter_complex_metadata(splitted_notes)

def load_architecture_table(path):
    loader = CSVLoader(path)
    return loader.load()
