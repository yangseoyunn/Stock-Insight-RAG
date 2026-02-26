import os
import json
import streamlit as st
from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# 1. API Key 설정
os.environ["GROQ_API_KEY"] = "API Key를 입력해 주세요."

# 2. 웹 페이지 UI 기본 설정
st.set_page_config(page_title="Stock RAG Chatbot", page_icon="📈", layout="centered")
st.title("📈 Stock-Insight-RAG")
st.caption("RLHF 피드백 수집 파이프라인 구축")


# 3. RAG 지식 베이스 구축 (한 번만 로드하도록 캐싱)
@st.cache_resource
def init_rag():
    loader = TextLoader("data/stock_kb.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    vectorstore = FAISS.from_documents(texts, embeddings)

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=vectorstore.as_retriever()
    )
    return qa_chain


qa_chain = init_rag()

# 4. 상태 저장 (질문과 답변을 화면에 유지하기 위함)
if "current_qa" not in st.session_state:
    st.session_state.current_qa = None

# 5. 사용자 채팅 입력
prompt = st.chat_input(
    "주식 투자에 대해 질문해 보세요. (ex. 금리가 오르면 기술주는 어떻게 돼?)"
)

if prompt:
    with st.spinner("지식 베이스를 검색하여 답변을 생성 중입니다..."):
        answer = qa_chain.run(prompt)
        st.session_state.current_qa = {"question": prompt, "answer": answer}

# 6. 결과 출력 및 RLHF 피드백 폼 (답변이 생성되었을 때만 표시)
if st.session_state.current_qa:
    st.chat_message("user").write(st.session_state.current_qa["question"])
    st.chat_message("assistant").write(st.session_state.current_qa["answer"])

    st.divider()
    st.subheader("📝 RLHF 보상 데이터 수집 (Human Feedback)")

    with st.form(key="feedback-form", clear_on_submit=True):
        st.write(
            "생성된 답변에 대한 피드백을 남겨 주세요. (모델 개선에 큰 도움이 됩니다!)"
        )
        score = st.slider("Reworad Score(1 : 매우 나쁨 ~ 5: 매우 좋음)", 1, 5, 3)
        reason = st.text_input("피드백 이유 (선택 사항)")
        submit_btn = st.form_submit_button("데이터셋에 저장")

        if submit_btn:
            # 평가 데이터를 JSONL 형식으로 저장
            log_data = {
                "timestamp": str(datetime.now()),
                "prompt": st.session_state.current_qa["question"],
                "response": st.session_state.current_qa["answer"],
                "reward_score": score,
                "feedback_reason": reason,
            }
            with open("dataset/feedback.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data) + "\n")

            st.success(
                "피드백이 저장되었습니다. 도움을 주셔서 감사합니다 😊 (`dataset/feedback.jsonl`)"
            )
            st.session_state.current_qa = None  # 완료 후 상태 초기화


