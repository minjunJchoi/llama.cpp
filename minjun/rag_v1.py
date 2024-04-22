
# #### ====================================================================================
# #### =========================== Run llama using ctransformer ===========================
# #### ====================================================================================

# from ctransformers import AutoModelForCausalLM

# # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
# model_path = "/Users/mjchoi/Work/codes/llama.cpp/models/llama-2-70b.Q5_K_M.gguf"
# model_path = "/Users/mjchoi/Work/codes/llama.cpp/models/falcon-180b-chat.Q5_K_S.gguf"
# llm = AutoModelForCausalLM.from_pretrained(model_path, model_type="falcon", gpu_layers=50)

# print(llm("What NFL team won the Super Bowl in the year Justin Bieber was born? Think step by step and answer. "))



#### ====================================================================================
#### =========================== Run llama using langchain ==============================
#### ====================================================================================

#### direct invoke

# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_community.llms import LlamaCpp

# n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
# n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
# # model_path = "/Users/mjchoi/Work/codes/llama.cpp/models/llama-2-70b.Q5_K_M.gguf"
# model_path = "/Users/mjchoi/Work/codes/llama.cpp/models/llama-2-70b-chat.Q5_K_M.gguf"

# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path=model_path,
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     verbose=True,  # Verbose is required to pass to the callback manager
# )

# # question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
# # llm_chain.invoke(question)

# # llm.invoke("What NFL team won the Super Bowl in the year Justin Bieber was born?")


# #### prompt chain invoke template 

# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.chains import LLMChain
# from langchain.chains.prompt_selector import ConditionalPromptSelector
# from langchain.prompts import PromptTemplate
# from langchain_community.llms import LlamaCpp
# from langchain import hub

# n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
# n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
# n_ctx = 2048 # length of input to your model
# # model_path = "/Users/mjchoi/Work/codes/llama.cpp/models/llama-2-70b.Q5_K_M.gguf"
# # model_path = "/Users/mjchoi/Work/codes/llama.cpp/models/llama-2-70b-chat.Q5_K_M.gguf"
# model_path = "/Users/mjchoi/Work/codes/llama.cpp/models/falcon-180b-chat.Q5_K_S.gguf"

# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path=model_path,
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     n_ctx=n_ctx,
#     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     verbose=True,  # Verbose is required to pass to the callback manager
#     max_tokens=50,
# )

# # prompt 1
# prompt = PromptTemplate(
#     input_variables=['question'], 
#     template="You are an assistant for question-answering task. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \nQuestion: {question} \nAnswer:"
# )

# # # prompt 2
# # DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
# #     input_variables=["question"],
# #     template="""<<SYS>> \n You are an assistant tasked with improving Google search \
# # results. \n <</SYS>> \n\n [INST] Generate THREE Google search queries that \
# # are similar to this question. The output should be a numbered list of questions \
# # and each should have a question mark at the end: \n\n {question} [/INST]""",
# # )
# # DEFAULT_SEARCH_PROMPT = PromptTemplate(
# #     input_variables=["question"],
# #     template="""You are an assistant tasked with improving Google search \
# # results. Generate THREE Google search queries that are similar to \
# # this question. The output should be a numbered list of questions and each \
# # should have a question mark at the end: {question}""",
# # )
# # QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
# #     default_prompt=DEFAULT_SEARCH_PROMPT,
# #     conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
# # )
# # prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)

# llm_chain = LLMChain(prompt=prompt, llm=llm)
# question = "In which Major League Baseball team is Tyler Glasnow playing for now?"
# llm_chain.invoke({"question": question})


# #### ====================================================================================
# #### =========================== RAG using langchain v0 =================================
# #### ====================================================================================

#### <<<<<<<<<<<<<<< Tracing using langsmith >>>>>>>>>>>>>>>>>>

import os
from langchain.callbacks.tracers import LangChainTracer

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__4bbb05cc61374749b29f8e64ba75fac0"
os.environ["LANGCHAIN_PROJECT"] = "rag with papers"

os.environ["TOKENIZERS_PARALLELISM"] = "false" # huggingface tokenizers problem https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning

# specify tracer  
# tracer = LangChainTracer(project_name="rag with papers")
# rag_chain.invoke(question, config={"callbacks": [tracer]})


#### <<<<<<<<<<<<<<< 0. indexing: loading >>>>>>>>>>>>>>>>>>

import requests
import bs4
from langchain_community.document_loaders import WebBaseLoader
import urllib3
urllib3.disable_warnings()

web_path = "https://www.espn.com/mlb/story/_/id/39821532/sources-dodgers-smith-close-10-year-140m-deal"
class_to_find = "article-body"

# Only keep contents of class_to_find
bs4_strainer = bs4.SoupStrainer(class_=(class_to_find))
loader = WebBaseLoader(
    web_paths=(web_path,),
    bs_kwargs={"parse_only": bs4_strainer},
)
loader.requests_kwargs = {'verify':False}
docs = loader.load()

# print(len(docs))
# print(len(docs[0].page_content))
# print(docs[0].page_content[:1000])


#### <<<<<<<<<<<<<<< 1. indexing: split >>>>>>>>>>>>>>>>>>

from langchain_text_splitters import RecursiveCharacterTextSplitter

chunk_size = 1000
chunk_overlap = 200

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# print(len(all_splits))
# print(len(all_splits[0].page_content))
# print(all_splits[10].metadata)


#### <<<<<<<<<<<<<<< 2. indexing: embedding and store >>>>>>>>>>>>>>>>>>

# from langchain_openai import OpenAIEmbeddings # not free
# from langchain_community.embeddings import GPT4AllEmbeddings # SSL error
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings

from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings()
# text = "This is a test document."
# query_result = embeddings.embed_query(text)
# print(query_result[:3])

# HuggingFace default model 
embedding = HuggingFaceEmbeddings()
 
# # specified model via HuggingFace
# model_name = "BAAI/bge-m3"
# # model_kwargs = {'device': 'mps'} # Metal Performance Shaders(for Apple Silicon) or 'cpu'
# model_kwargs = {'device': 'cpu'} # Metal Performance Shaders(for Apple Silicon) or 'cpu'

# embedding = HuggingFaceEmbeddings(
#     model_name=model_name, 
#     model_kwargs=model_kwargs, 
#     encode_kwargs=model_kwargs,
#     show_progress=True
# )

# # instruct model

# from InstructorEmbedding import INSTRUCTOR

# model = INSTRUCTOR('hkunlp/instructor-xl')
# # model_kwargs = {'device': 'mps'} # Metal Performance Shaders(for Apple Silicon) or 'cpu'
# # model_kwargs = {'device': 'cpu'} # Metal Performance Shaders(for Apple Silicon) or 'cpu'

# sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
# instruction = "Represent the Reporter title:"
# embedding = model.encode([[instruction,sentence]])
# print(embedding)


# using only current input
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding) 

# # use previously vectorized documents saved in db
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory='db') 
# # save current document for persist-use 
# vectorstore.persist() 


#### 3. retrieval: retrieve

# search_type = "similarity"
# search_k = 5 # number of close chunks to find
# retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": search_k}) # number of close chunks to find

retriever = vectorstore.as_retriever()

#### 4. generate: RAG chain with prompt, docs, llm
#### <<<<<<<<<<<<<<< 4.1. llm >>>>>>>>>>>>>>>>>>

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
n_ctx = 2048 # Length of input tokens
# model_path = "/Users/mjchoi/Work/codes/llama.cpp/models/llama-2-70b.Q5_K_M.gguf"
model_path = "/Users/mjchoi/Work/codes/llama.cpp/models/llama-2-70b-chat.Q5_K_M.gguf"
# model_path = "/Users/mjchoi/Work/codes/llama.cpp/models/falcon-180b-chat.Q5_K_S.gguf"

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=n_ctx,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,  # Verbose is required to pass to the callback manager
    max_tokens=2000,
)

# top_p=0.75,
# top_k=40,
# repetition_penalty=1.1,


#### <<<<<<<<<<<<<<< 4.2. prompt >>>>>>>>>>>>>>>>>>

# # rag_prompt 
# from langchain.prompts import PromptTemplate

# # rag_prompt = PromptTemplate(
# #     input_variables=['context', 'question'], 
# #     template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \nQuestion: {question} \nContext: {context} \nAnswer:"
# # )

# rag_prompt with citation 

from langchain_core.prompts import ChatPromptTemplate

# Return a citation for every quote across all sources that is used to make the answer. \

system = """I'm a helpful AI assistant. Given contents of the following sources, \
I'll answer the question asked by Human ONLY based on the provided sources. \
If none of the sources answer the question, I'll just say I don't know.
Remember, I return BOTH an answer and the relevant citations. 
A citation consists of a VERBATIM quote of cited contents and the EXACT source ID of the cited content. \

Here are the sources followed by the question by Human in the end:{context}"""

# system = """Falcon is a helpful AI assistant. Given contents of the following sources, \
# Falcon will answer the question asked by Human ONLY based on the provided sources. \
# If none of the sources answer the question, Falcon will just say Falcon does not know.
# Remember, Falcon returns BOTH an answer and the relevant citations. 
# A citation consists of a VERBATIM quote of cited contents and the EXACT source ID for the cited content. \

# Here are the sources followed by the question by Human in the end:{context}"""

rag_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{question}")]
)

#### <<<<<<<<<<<<<<< 4.3. run via chain >>>>>>>>>>>>>>>>>>

question = "Which Major League Baseball team is Tyler Glasnow playing for? System:"

# from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnablePick, RunnableLambda

output_parser = StrOutputParser()


# ## chain form
# setup_and_retriveval = RunnableParallel(
#                         {"context": retriever, "question": RunnablePassthrough()}
# )

# rag_chain = setup_and_retriveval | rag_prompt | llm | output_parser
# rag_chain.invoke(question)

## chain with source
from operator import itemgetter
from typing import List
from langchain_core.documents import Document

# source (context) format
def format_docs(docs: List[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        doc_str = f"\nContent of source ID={i}: {doc.page_content}"
        formatted.append(doc_str)
    return "\n".join(formatted)

format = itemgetter("docs") | RunnableLambda(format_docs)

answer = rag_prompt | llm | output_parser 
# answer = rag_prompt | llm | output_parser | itemgetter("cited_answer")

rag_chain = (
    RunnableParallel(question=RunnablePassthrough(), docs=retriever)
    .assign(context=format)
    .assign(answer=answer)
    # .assign(cited_answer=answer)
    # .pick(["cited_answer", "docs"])
)
rag_chain.invoke(question)

# invoke retriever to return retrieved docs
retrieved_docs = retriever.invoke(question)
for i, docs in enumerate(retrieved_docs):
    print()
    print(f'Source ID {i}: {docs.page_content}')

    


# #### Import the RetrievalQA class from the langchain module 

# from langchain.chains import RetrievalQA 

# # Create a RetrievalQA instance with specified components 
# rag_chain = RetrievalQA.from_chain_type( 
#     llm=llm,                       # Provide the language model 
#     chain_type="stuff",            # Specify the type of the language model chain 
#     retriever=retriever,           # Provide the document retriever 
#     return_source_documents=True   # Returning source-documents with answers 
# ) 

# # Define a query for the RetrievalQA chain 
# query = "In which Major League Baseball team is Tyler Glasnow playing for now?"

# # Execute the query using the RetrievalQA chain and store the result 
# result = rag_chain(query) 

# # Print or use the formatted result text 
# print(result['result'])  

# #Extracting content from the retrieved document chunks used for specific query 
# documents=result['source_documents'] 
# for document in documents: 
#     page_content = document.page_content 
#     metadata = document.metadata 

#     print("Page Content:", page_content) 
#     print("Source:", metadata['source']) 
#     print("Start Index:", metadata['start_index']) 

















# #### ====================================================================================
# #### =========================== OpenAI chat using API ==================================
# #### ====================================================================================

# from openai import OpenAI

# client = OpenAI()

# completion = client.chat.completions.create(
#     model="gpt-3.5-turbo-0125",
#     messages=[
#     {"role":"system", "content":"너는 프로그래머야. 파이썬을 활용해서 사용자의 요구를 충족시키는 프로그램을 만들고 설명해줘"},
#     {"role":"user", "content":"1부터 100까지 소수만 출력하는 프로그램을 만들어봐"}
#     ]
# )

# print(completion.choices[0].message.content)



#article-feed > article.article.article-espn-plus.ungated > div > div.article-body

# import requests
# from bs4 import BeautifulSoup

# # webpage load
# web_path = 'https://iopscience.iop.org/article/10.1088/1361-6587/ad290e'
# # web_path = 'https://www.google.com'
# # web_path = 'https://blog.naver.com/prologue/PrologueList.naver?blogId=hj_bw&skinType=&skinId=&from=menu%EF%BB%BF'
# response = requests.get(web_path, verify=False)

# # webpage parsing
# soup = BeautifulSoup(response.text, "html.parser")
# elements = soup.select_one("#prologue > dl > dd:nth-child(1) > ul > li.p_title > a")
# elements = soup.select_one("#page-content > div:nth-child(3) > div.article-content > div.article-text.wd-jnl-art-abstract.cf > p")




#

# elements = soup.find_all(attrs={'itemprop':'articleBody'})
# elements = soup.select('.inline-eqn')

# print("=================================")
# for element in elements:
#     print(element.text)


# for sub_heading in soup.find_all('title'):
#     print(sub_heading.text)



# ## asking after learning document 
# from langchain.document_loaders import PyPDFLoader
# from langchain.vectorstores import Chroma
# from langchain.embeddings import LlamaCppEmbeddings

# # loading document 
# pdf_path = "/Users/mjchoi/Downloads/Fujisawa_review of plasma turbulence.pdf"
# loader = PyPDFLoader(pdf_path)
# pages = loader.load_and_split()

# # indexing document
# embeddings = LlamaCppEmbeddings(model_path=model_path)
# vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")

# vectordb.persist()

# from langchain.chains import RetrievalQA

# MIN_DOCS = 1 # only one result from the database

# qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(search_kwargs={"k": MIN_DOCS}))

# query = "In which tokamak device is the zonal flow first demonstrated experimentally?"
# qa.run(query)