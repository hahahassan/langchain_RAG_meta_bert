# Langchain RAG

This repository is a modified version of the [Langchain RAG Tutorial](https://github.com/pixegami/langchain-rag-tutorial). It utilizes Hugging Face's `meta-llama/Meta-Llama-3-8B-Instruct` model and `HuggingFaceEmbeddings` with `bert-base-uncased` for document and query embeddings.

## Install Dependencies

1. **MacOS Users:** Due to current challenges installing `onnxruntime` through `pip install onnxruntime`, use the following workaround:

    ```sh
    conda install onnxruntime -c conda-forge
    ```
    Refer to this [thread](https://github.com/microsoft/onnxruntime/issues/11037) for additional help if needed.

2. **Windows Users:** Follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Ensure you follow through to the last step to set the environment variable path.

3. Install dependencies from the `requirements.txt` file:

    ```sh
    pip install -r requirements.txt
    ```

4. Install markdown dependencies with:

    ```sh
    pip install "unstructured[md]"
    ```

## Create and Query Database

1. **Create the Chroma DB:**

    ```sh
    python create_database.py
    ```

2. **Query the Chroma DB:**

    ```sh
    python query_data.py "How does Alice meet the Mad Hatter?"
    ```

> Ensure you have set up a Hugging Face account and configured your Hugging Face API key in your environment variables for this to work.

For a detailed step-by-step tutorial, watch this video: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).
