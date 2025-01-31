# LLM-based-FAQ-bot
A chatbot that answers based on context provided in the form of documents.
# How to run
This app is based on Flask APIs and a RAG system built with Llama-index.
Before everything, install the requirements. (requirements.txt)

The next step is to run the main.py file which hosts the Flask app.

Once the FLask app is up and running, we can send requests via Thunder Client/Postman, or create our own client scripts to send our requests.

Below given are the endpoints you should use:

1. [POST] URL:  '/query':

   You have to send a POST request with a JSON file which contains your query in the form:
   {"query":your_query}

    ![image](https://github.com/user-attachments/assets/88205301-0ab1-4219-8679-1db7955dfc94)

2. [POST] URL: '/admin/upload':

   You have to send a POST request with your document to upload included in the 'Files'. You also need to add your X-API key to the request as this is an admin-only service.
   Below is the image from the script you can use for that:
   ![image](https://github.com/user-attachments/assets/9e9918e9-52fc-4151-ad77-1d4a998e0875)

3. [GET] URL: '/admin/log':

   This endpoint retrieves the logging information. You can simply hit the URL to get the logs. If you want to filter with the start and end dates, you can add them as parameters. Remember you will need your X-API-Key here as well.

   ![image](https://github.com/user-attachments/assets/6b9c1638-5078-4a41-aa20-a18e1e0a7509)

#  How it works

This follows a RAG based system:

1. The documents are read from the Docstore (DATA_DIR)
2. We perform semantic chunking using Llama-index modules on the documents.
3. We create node objects from the documents.
4. The Embedding model creates embeddings of the node text.
5. We add metadata and text to the node objects. (Metadata is achieved from the Document objects, this will be used to show sources of our answer.)
6. The FAISS index of the embeddings is created.
7. We embed the user query.
8. We retrieve top-k embeddings from FAISS based on the query embedding.
9. We retrieve the text data and sources from the Docstore.
10. We pass the context and the query to the LLM.
11. Returns the answer to the user keeping in mind the context and the query. 
