import os
import gradio as gr
from groq import Groq

# The API key is read from the environment variable you set on Render
# (Environment Variables -> GROQ_API_KEY). Never hardcode it here.
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"  # change to any model your Groq account supports


def chat(message, history):
    # Build the message list Groq expects: a list of {"role": ..., "content": ...}
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    return response.choices[0].message.content


demo = gr.ChatInterface(
    fn=chat,
    title="Groq Chatbot",
    description="A simple chatbot powered by the Groq API.",
)

if __name__ == "__main__":
    # host="0.0.0.0" and the PORT env var are required for Render to detect the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
