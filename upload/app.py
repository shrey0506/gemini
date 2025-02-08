import os
import langchain
import langgraph
from flask import Flask, request, jsonify
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize Flask app (Controller in MVC)
app = Flask(__name__)

# Load environment variables for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_KEY = os.getenv("HUGGINGFACEHUB_API_KEY")

# Define available models (Model in MVC)
MODELS = {
    "openai": OpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY),
    "huggingface": HuggingFaceHub(repo_id="facebook/bart-large-cnn", huggingfacehub_api_token=HUGGINGFACEHUB_API_KEY)
}

def create_summary_chain(model_name):
    """Creates an LLM chain for text summarization."""
    if model_name not in MODELS:
        raise ValueError("Invalid model selected")
    
    prompt = PromptTemplate(template="Summarize the following text:\n{text}", input_variables=["text"])
    chain = LLMChain(llm=MODELS[model_name], prompt=prompt)
    return chain

@app.route("/summarize", methods=["POST"])
def summarize():
    """Handles summarization requests."""
    data = request.json
    text = data.get("text", "")
    model_name = data.get("model", "openai")
    
    try:
        summarizer_chain = create_summary_chain(model_name)
        summary = summarizer_chain.run(text)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)