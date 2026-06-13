import streamlit as st
from app import KnowledgeGraphBackend

st.set_page_config(page_title="Knowledge Graph System", layout="centered")
st.title("Knowledge Graph Ingestor & Chatbot")

backend = KnowledgeGraphBackend()


st.header("1. Ingest Unstructured Data")
tab1, tab2 = st.tabs(["Paste Raw Text", "Upload PDF Document"])

raw_text = ""
with tab1:
    raw_text = st.text_area(
        "Paste text block here:", 
        height=150,
        placeholder="e.g., Steve Jobs co-founded Apple with Steve Wozniak..."
    )

with tab2:
    uploaded_file = st.file_uploader("Choose a PDF file wrapper", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Extracting content segments from PDF..."):
            try:
                raw_text = backend.extract_text_from_pdf(uploaded_file)
                st.success("PDF extraction completed!")
            except Exception as e:
                st.error(f"Failed to parse PDF: {str(e)}")

if st.button("Process Data into Graph", type="primary"):
    if not raw_text.strip():
        st.error("Please provide valid textual input data first.")
    else:
        try:
            status_element = st.empty()
            progress_bar = st.progress(10)
            
            status_element.text("Extracting entities and loading triples into database...")
            progress_bar.progress(50)
            
            backend.ingest_text_to_graph(raw_text)
            
            progress_bar.progress(100)
            status_element.empty()
            st.success("Knowledge graph compilation completed successfully!")
        except Exception as e:
            st.error(f"Ingestion lifecycle failed: {str(e)}")

st.markdown("---")

st.header("2. Ask Questions From Your Graph")

user_query = st.text_input(
    "Query connections inside your knowledge matrix:", 
    placeholder="e.g., Explain the relationship between Steve Jobs and Apple."
)

if st.button("🔍 Query Knowledge Graph"):
    if not user_query.strip():
        st.warning("Please type a valid prompt query.")
    else:
        with st.spinner("Traversing knowledge structures..."):
            try:
                answer = backend.query_graph_rag(user_query)
                st.markdown("### Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Graph retrieval failed: {str(e)}")