# 📈 Stock-Insight-RAG: RLHF 기반 주식 Q&A 및 피드백 수집 시스템

### 🎯 Project Overview
본 프로젝트는 금융 도메인의 특수성을 고려하여, 외부 지식 베이스를 활용한 **RAG(Retrieval-Augmented Generation)** 시스템을 구축하고, 모델의 답변 품질 개선을 위한 **RLHF(Reinforce Learning from Human Feedback)** 리워드 데이터셋을 수집하는 파이프라인 실험입니다.

--- 
### 🛠 Key Features
- **Knowledge-Based Retrieval**: `data/stock_kb.txt`에 저장된 전문적인 주식 투자 지표 및 거시 경제 데이터를 바탕으로 hallucination 현상을 최소화한 답변을 생성합니다.

- **High-Performance Inference**: Groq 엔진과 Llama-3.3-70b 모델을 결합하여 실시간에 가까운 빠른 추론 속도를 구현했습니다.

- **Efficient Vector Search**: FAISS와 HuggingFace Embeddings를 활용하여 CPU 환경에서도 최적화된 시맨틱 검색 기능을 제공합니다.

- **RLHF Feedback Loop**: 사용자가 답변에 대한 점수를 1~5점으로 평가하고 피드백 이유를 남길 수 있는 인터페이스를 통해, <ins>향후 모델 정렬 학습에 활용 가능한</ins>(예정) JSONL 데이터셋을 자동으로 구축합니다.


### ⚙️ Tech Stack
- **Framework**: LangChain, Streamlit

- **LLM & Embedding**: Groq (Llama-3.3-70b), HuggingFace (all-MiniLM-L6-v2)

- **Vector DB**: FAISS

- **Language**: Python 3.12+


### 📁 Structure

```text
.  
├── app.py                # Streamlit 기반 메인 애플리케이션 코드  
├── data/  
│   └── stock_kb.txt      # 주식 도메인 지식 베이스  
├── dataset/  
│   └── feedback.jsonl    # 수집된 RLHF 피드백 데이터셋
└── README.md            
