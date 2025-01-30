import cohere
from rich import print
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
CohereAPIKey = os.getenv("CohereApiKey")

if not CohereAPIKey:
    raise ValueError("CohereApiKey is missing from .env file or not loaded properly.")

co = cohere.Client(api_key=CohereAPIKey)

# Define functions and variables
funcs = [
    "exit", "general", "realtime", "open", "close", "play",
    "generate image", "system", "content", "google search",
    "youtube search", "reminder"
]


# Keywords that determine the query type
realtime_keywords = ["who", "what", "when", "where", "tell me about"]
messages = []

preamble = """
You are a very accurate Decision-Making Model, which decides what kind of a query is given to you.
You will decide whether a query is a 'general' query, a 'realtime' query, or is asking to perform any task or automation like 'open facebook, instagram', 'can you write an application and open it in notepad.'
*** Do not answer any query, just decide what kind of query is given to you. ***
-> Respond with 'general ( query )' if a query can be answered by a llm model (conversational AI chatbot) and doesn't require any up-to-date information.
-> Respond with 'realtime ( query )' if a query cannot be answered by a llm model and requires up-to-date information or contains keywords like 'who,' 'what,' 'when,' 'where,' or 'tell me about.'
-> Respond with 'open (application name or website name)' if a query is asking to open any application or website.
-> Respond with 'close (application name)' if a query is asking to close any application or website.
-> Respond with 'play (song name)' if a query is asking to play any song.
-> Respond with 'generate image (image prompt)' if a query is requesting to generate an image with a given prompt.
-> Respond with 'reminder (datetime with message)' if a query is requesting to set a reminder.
-> Respond with 'system (task name)' if a query is asking to perform system-level tasks like mute, unmute, volume up, etc.
-> Respond with 'content (topic)' if a query is asking to write any type of content like application, codes, emails, or anything else.
-> Respond with 'google search (topic)' if a query is asking to search a specific topic on Google.
-> Respond with 'youtube search (topic)' if a query is asking to search a specific topic on YouTube.
-> If the query is asking to perform multiple tasks, respond with each task separately (e.g., 'open facebook, open telegram, close whatsapp').
-> If the user is saying goodbye or wants to end the conversation, respond with 'exit'.
-> Respond with 'realtime (query)' if you can't decide the kind of query or if a query is asking to perform a task which is not mentioned above.
"""

ChatHistory = [
    {"role": "User", "message": "how are you"},
    {"role": "Chatbot", "message": "general how are you"},
    {"role": "User", "message": "do you like pizza?"},
    {"role": "Chatbot", "message": "general do you like pizza?"},
    {"role": "User", "message": "open chrome and tell me about mahatma gandhi."},
    {"role": "Chatbot", "message": "open chrome, realtime tell me about mahatma gandhi."},
    {"role": "User", "message": "open chrome and firefox"},
    {"role": "Chatbot", "message": "open chrome, open firefox"},
    {"role": "User", "message": "what is today's date and by the way remind me that i have a dancing performance on 5th aug at 11 pm"},
    {"role": "Chatbot", "message": "realtime what is today's date, reminder 11:00pm 5th aug dancing performance"},
    {"role": "User", "message": "chat with me."},
    {"role": "Chatbot", "message": "general chat with me."}
]

def FirstLayerDMM(prompt: str = "test"):
    messages.append({"role": "user", "content": f"{prompt}"})

    # Check for realtime keywords
    if any(keyword in prompt.lower() for keyword in realtime_keywords):
        return [f"realtime ({prompt})"]

    # Using Cohere's chat_stream
    stream = co.chat_stream(
        model='command-r-plus',
        message=prompt,
        temperature=0.7,
        chat_history=ChatHistory,
        prompt_truncation='OFF',
        connectors=[],
        preamble=preamble
    )

    response = ""

    for event in stream:
        if event.event_type == "text-generation":
            response += event.text

    response = response.replace("\n", "")
    response = response.split(",")

    response = [i.strip() for i in response]

    temp = []

    for task in response:
        for func in funcs:
            if task.startswith(func):
                temp.append(task)
    response = temp

    if "(query)" in response:
        newresponse = FirstLayerDMM(prompt=prompt)
        return newresponse
    else:
        return response


if __name__ == "__main__":
    while True:
        user_input = input(">>> ")
        print(FirstLayerDMM(user_input))
