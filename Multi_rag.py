import os
import fitz  # PyMuPDF
from PIL import Image
import io
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

def extract_images_from_pdf(pdf_path):
    images = {}
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images[f"page_{page_num + 1}_image_{img_index + 1}"] = image
    return images

def create_vector_store(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(texts, embeddings)
    return vector_store

def get_relevant_image(images, content):
    # This is a simple implementation. You may need to improve this based on your specific use case.
    for image_name, image in images.items():
        page_num = int(image_name.split('_')[1])
        if f"page {page_num}" in content.lower():
            return image
    return None

def answer_question(question, vector_store, images):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    prompt_template = """
    You are an AI assistant that answers questions based on a user manual. 
    Use the following context to answer the question. If you can't answer the question based on the context, say "I don't have enough information to answer that question."
    Also, mention if there's an image related to the answer.

    Context: {context}

    Question: {question}

    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    
    result = qa_chain({"query": question})
    answer = result['result']
    
    relevant_image = get_relevant_image(images, answer)
    
    return answer, relevant_image

# Main execution
pdf_path = "path_to_your_user_manual.pdf"
images = extract_images_from_pdf(pdf_path)
vector_store = create_vector_store(pdf_path)

while True:
    question = input("Ask a question about the user manual (or type 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    
    answer, image = answer_question(question, vector_store, images)
    print(f"Answer: {answer}")
    
    if image:
        print("Related image found. Displaying...")
        image.show()
    else:
        print("No related image found.")
