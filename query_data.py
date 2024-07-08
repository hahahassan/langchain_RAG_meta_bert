import argparse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient
import numpy as np

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def normalize_scores(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    normalized_scores = (scores - min_score) / (max_score - min_score)
    return normalized_scores

def query_huggingface(client, question, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{context}\n\n{question}"}
    ]
    
    response_text = ""
    for message in client.chat_completion(
        messages=messages,
        max_tokens=500,
        stream=True,
    ):
        if "choices" in message and message["choices"]:
            response_text += message["choices"][0]["delta"].get("content", "")
    
    return response_text

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = HuggingFaceEmbeddings(model_name='bert-base-uncased')
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    scores = [score for _, score in results]
    normalized_scores = normalize_scores(np.array(scores))
    normalized_results = [(doc, score) for (doc, _), score in zip(results, normalized_scores)]
    print(normalized_results)
    if len(normalized_results) == 0 or all(score < 0.1 for _, score in normalized_results):
        print("Unable to find matching results.")
        

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in normalized_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print('-----------------------------------------------------------------')
    print('prompt :', prompt)

    # Use HuggingFace InferenceClient for question answering
    client = InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        token="hf_************************************",  # Replace with your Hugging Face API token
    )

    response_text = query_huggingface(client, question=query_text, context=context_text)
    sources = [doc.metadata.get("source", None) for doc, _score in normalized_results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()

