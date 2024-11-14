import ollama

# Initialize the Ollama client
client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")


# Placeholder for a RAG retrieval function (this would be an external API or a database query)
def retrieve_relevant_context(prompt):
    # Perform retrieval here
    return "In this context, sick doesn't have a bad meaning. It means person feels excited."


# Example function to request a response from Ollama model
def get_model_response_with_context(prompt, context):
    # Combine context and prompt for a RAG-style input
    full_input = f"Context: {context}\nQuestion: {prompt}"
    response = client.chat(model="llama3.1", messages=[{"role": "user", "content": full_input}])
    return response["message"]["content"]


# Calculate metrics as before
def calculate_context_metrics(context, response):
    context_tokens = set(context.lower().split())
    response_tokens = set(response.lower().split())
    relevant_tokens = context_tokens & response_tokens

    precision = len(relevant_tokens) / len(response_tokens) if response_tokens else 0.0
    recall = len(relevant_tokens) / len(context_tokens) if context_tokens else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1_score


if __name__ == "__main__":
    # Define prompt and retrieve relevant context (RAG-style)
    prompt = "I am about to cry, this is so sick!!!! - what emotion is expressed here??"
    context = retrieve_relevant_context(prompt)  # Get relevant information using retrieval
    print("Retrieved context:", context)

    # Get model response with added context
    response = get_model_response_with_context(prompt, context)
    print("Model response:", response)

    # Calculate Precision, Recall, and F1 Score
    precision, recall, f1_score = calculate_context_metrics(context, response)
    print(f"Context Precision: {precision:.2f}")
    print(f"Context Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
