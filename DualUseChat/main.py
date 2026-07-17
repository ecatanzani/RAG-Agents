import streamlit as st
from pathlib import Path

from langchain_aws import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.output_parsers import StrOutputParser

import params

prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "Sei un assistente virtuale legale e tecnico professionista..\n"
        "Il tuo obiettivo è rispondere alle domande basandoti rigorosamente sulle clausole e sui documenti forniti.\n\n"
        "REGOLA CRITICA\n"
        "Rispondi alla domanda dell'utente utilizzando ESCLUSIVAMENTE il contesto del documento fornito qui sotto. "
        "Mantieni un'assoluta precisione legale. "
        "Se la risposta o un dettaglio specifico non possono essere trovati nel contesto, dichiara esplicitamente: 'Non sono in grado di verificarlo sulla base della documentazione fornita."
        "Non estrapolare o speculare oltre il testo.\n\n"
        "--- PROVIDED DOCUMENT CONTEXT ---\n"
        "{context_str}"
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{user_query}")
])

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def main():
    st.set_page_config(page_title="Dual-Use Legal Chat Assistant")
    st.title("Dual-Use Legal Chat Assistant")
    
    if not Path(params.DOCUMENTS_DIR).is_dir():
        st.error(f"[ERROR] Documents directory {params.DOCUMENTS_DIR} does not exist")
        raise Exception(f"[ERROR] Documents directory {params.DOCUMENTS_DIR} does not exist")

    @st.cache_resource(show_spinner="Initializing embeddings model")
    def load_embeddings_model():
        return BedrockEmbeddings(
            model_id=params.AWS_EMBEDDINGS_MODEL_ID, 
            region_name=params.AWS_EMBEDDINGS_REGION
        )

    embeddings_model = load_embeddings_model()

    @st.cache_resource(show_spinner="Initializing legal knowledge base...")
    def initialize_vector_store():
        db_exists = False
        chroma_dir = Path(params.VECTOR_STORE_DIR)
        if chroma_dir.is_dir():
            files = list(chroma_dir.iterdir())
            if any(f.suffix == '.sqlite3' or f == 'chroma.sqlite3' for f in files) or len(files) > 0:
                db_exists = True

        if db_exists:
            return Chroma(
                collection_name=params.VECTOR_STORE_COLLECTION_NAME,
                persist_directory=params.VECTOR_STORE_DIR,
                embedding_function=embeddings_model
            )

        data_dir = Path(params.DOCUMENTS_DIR)
        pdf_files = [f for f in list(data_dir.iterdir()) if f.suffix.lower() == '.pdf']
        
        if not pdf_files:
            st.error(f"No PDFs found in {params.DOCUMENTS_DIR} folder! Drop some PDFs there and refresh.")
            raise Exception(f"[ERROR] Documents directory {params.DOCUMENTS_DIR} is empty")

        all_docs = []
        for file_path in pdf_files:
            loader = PyPDFLoader(str(file_path))
            all_docs.extend(loader.load())

        text_splitter = SemanticChunker(
            embeddings_model, 
            breakpoint_threshold_type="percentile"
        )
        split_docs = text_splitter.split_documents(all_docs)

        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings_model,
            persist_directory=params.VECTOR_STORE_DIR,
            collection_name=params.VECTOR_STORE_COLLECTION_NAME
        )
        return vectorstore

    vector_store = initialize_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4}) if vector_store else None


    @st.cache_resource(show_spinner="Initializing LLM")
    def load_model():
        return ChatBedrockConverse(
            model_id=params.AWS_LLM_MODEL_ID,
            temperature=params.AWS_LLM_TEMPERATURE,
            region_name=params.AWS_LLM_REGION
        )
    
    llm = load_model()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)

    if user_query := st.chat_input("Ask a question about the stored legal/technical documents..."):
        
        with st.chat_message("user"):
            st.write(user_query)
        
        context_str = "No reference documents indexed."

        if retriever:
            retrieved_docs = retriever.invoke(user_query)
            context_str = format_docs(retrieved_docs)
            
            with st.expander("🔍 Review retrieved document context blocks"):
                for doc in retrieved_docs:
                    source_name = Path(doc.metadata.get('source', 'unknown')).name
                    page_num = doc.metadata.get('page', 0) + 1
                    st.caption(f"Source: {source_name} (Page {page_num})")
                    st.write(doc.page_content)
                    st.markdown("---")

        rag_chain = (prompt_template | llm | StrOutputParser())

        # 4. Invoke LLM and Render Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context and drafting response..."):
                try:
                    payload = {
                        "context_str": context_str,
                        "chat_history": st.session_state.messages,
                        "user_query": user_query
                    }
                    response_text = rag_chain.invoke(payload)
                    
                    st.write(response_text)
                    
                    st.session_state.messages.append(HumanMessage(content=user_query))
                    st.session_state.messages.append(AIMessage(content=response_text))
                    
                except Exception as e:
                    st.error(f"Error communicating with AWS Bedrock: {e}")

if __name__ == '__main__':
    main()