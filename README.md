# LLM-based-FAQ-bot
A chatbot that answers based on context provided in the form of documents.
# Hwo to run
This app is based on Flask APIs and a RAG system built with Llama-index.
Before everything, install the requirements. (requirements.txt)

The next step is to run the main.py file which hosts the Flask app.

Once the FLask app is up and running, we can send requests via Thunder Client/Postman, or create our own client scripts to send our requests.

Below given are the endpoints you should use:

1. [POST] URL:  '/query':

   You have to send a POST request with a JSON file which contains your query in the form:
   {"query":your_query}

    ![image](https://github.com/user-attachments/assets/88205301-0ab1-4219-8679-1db7955dfc94)


