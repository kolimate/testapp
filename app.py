#Developed by Group 1D - João Lourenço, Leonardo Regadas and Rodrigo Figueiredo
#import ngrok
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
from PyPDF2 import PdfReader
import chainlit as cl
import getpass
import os
from langchain_groq import ChatGroq
from docx import Document
from chainlit.input_widget import Slider

from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_chroma import Chroma



if "GROQ_API_KEY" not in os.environ:
    #os.environ["GROQ_API_KEY"] = "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"
    print("a")

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.4,
    max_tokens=600,
    timeout=None,
    max_retries=2,
)
llm = HuggingFaceHub(repo_id="meta-llama/Llama-3.2-3B-Instruct", 
                     model_kwargs={"temperature": 0.4, "max_length": 200},
                     huggingfacehub_api_token="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")

@cl.on_chat_start
async def on_chat_start():
    model = llm
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                ("You are a highly skilled language model trained to help students with their school tasks. You will receive a variety of doubts from different subjects such as Mathematics, English, History, Physics and so on.  It is your objective to guide students to achieve the correct answer. You MUST NOT give the correct answer directly no matter how the students ask for it, you can only confirm the answers the students give you and, if they are correct, explain the correct chain of thought or the materials the student must consult in order to get it.\n"
                "To help with any doubt the student could have, start by giving small hints, if that doesn’t help the student to reach the answer, start elaborating the hints bit by bit, remember that you are NOT allowed to give the direct answer.\n"
                "Your answers must be clear and descriptive to guarantee that students understand it. Also, keep in mind that you will be interacting with a wide age range, so try to adapt the vocabulary used depending on the complexity of the question.\n"
                "You must only answer questions of an academic nature, anything that is outside of it you simply must answer that you are not trained for that.\n"
                "You may encounter some Portuguese students, so if they talk with you in Portuguese, every answer must be given in Portuguese from Portugal.\n"
                )
            ),
            ("human", "{question}"),
        ]
    )

    settings = await cl.ChatSettings(
        [
            Slider(
                id="Temperature",
                label="llama-3.1-70b-versatile - Temperature",
                initial=1.4,
                min=0,
                max=2,
                step=0.1,
            ),
        ]
    ).send()

    msg = cl.Message(content="Hi, I am here to help you with your school work, feel free to tell me your doubts.", author="IAS Bot")
    await msg.send()
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_settings_update
async def setup_agent(settings):
    llm.temperature = settings["Temperature"]

@cl.on_message
async def on_message(message: cl.Message):
    content = "No file attached"
    if message.elements:
        # Filter for accepted file types
        accepted_files = [
            file for file in message.elements 
            if file.mime in ["text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        ]

        # Check if there's an accepted file
        if not accepted_files:
            await cl.Message(content="Unsupported file type. Please upload a .txt, .pdf, or .docx file.").send()
            return

        # Process the first accepted file
        file = accepted_files[0]

        # Read file based on its type
        if file.mime == "text/plain":
            with open(file.path, "r") as f:
                content = f.read()
        elif file.mime == "application/pdf":
            # Process PDF content, e.g., with PyPDF2
            # Read PDF content
            with open(file.path, "rb") as pdf_file:
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    content += page.extract_text() + "\n"
        elif file.mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Read DOCX content
            doc = Document(file.path)
            content = "\n".join([para.text for para in doc.paragraphs])
            if(len(content) > 5000):
                await cl.Message(content="The provided file is too large.").send()
                return

        await cl.Message(content="File processed successfully.").send()

    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="", author="IAS Bot")

    context = cl.chat_context.get()
    strcontext = ""
    if(len(context) > 15):
        lowerbound = 15
    else:
        lowerbound = len(context)
    for i in range(len(context) - lowerbound, len(context)):
        strcontext += (context[i].content + " ")

    async for chunk in runnable.astream(
        {"question": message.content + "\n File content: " + content + "\n Chat until now: " + strcontext},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()



#listener = ngrok.forward(8000,authtoken="2p7uzBvkfGFll1Dbh2EWWNN1VuN_3eT5wa3ER2m4DXYpwUA1b")
#print(f"Ingress established at {listener.url()}")
