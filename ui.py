"""Streamlit front end. Run: streamlit run ui.py"""
import requests
import streamlit as st

API = "http://localhost:8000"
st.set_page_config(page_title="Chat with your documents", page_icon="📄")
st.title("📄 Chat with your documents")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text, sources)

# ---- Sidebar: upload + choose document ----
with st.sidebar:
    st.header("Documents")
    f = st.file_uploader("Upload a PDF", type="pdf")
    if f and st.button("Process"):
        with st.spinner("Reading, chunking and embedding..."):
            r = requests.post(f"{API}/upload", files={"file": (f.name, f.getvalue())})
        if r.ok:
            st.success(f"Indexed {r.json()['chunks']} chunks from {r.json()['pages']} pages")
        else:
            st.error(r.json().get("detail", "Upload failed"))

    try:
        docs = requests.get(f"{API}/documents").json()
    except requests.exceptions.ConnectionError:
        docs = []
        st.error("Backend not running. Start it with: uvicorn app.main:app --reload")

    options = {"All documents": None} | {d["filename"]: d["doc_id"] for d in docs}
    choice = st.selectbox("Search in", list(options))
    doc_id = options[choice]
    k = st.slider("Chunks retrieved (k)", 1, 10, 5)

# ---- Chat ----
for role, text, sources in st.session_state.history:
    with st.chat_message(role):
        st.write(text)
        if sources:
            with st.expander("Sources"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**[{i}] {s['filename']}, page {s['page']}**")
                    st.caption(s["text"][:400] + "...")

if q := st.chat_input("Ask a question about your document"):
    st.chat_message("user").write(q)
    with st.spinner("Thinking..."):
        r = requests.post(f"{API}/ask", json={"question": q, "doc_id": doc_id, "k": k})
    if r.ok:
        data = r.json()
        with st.chat_message("assistant"):
            st.write(data["answer"])
            with st.expander("Sources"):
                for i, s in enumerate(data["sources"], 1):
                    st.markdown(f"**[{i}] {s['filename']}, page {s['page']}**")
                    st.caption(s["text"][:400] + "...")
        st.session_state.history += [("user", q, None), ("assistant", data["answer"], data["sources"])]
    else:
        st.error(r.text)
