from __future__ import annotations

import asyncio
import uuid

import streamlit as st

from src.agents.language_agent import LanguageAgent
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.embedding_service import EmbeddingService
from src.rag_news import NewsRAG
from src.sentiment_service import SentimentService
from src.session_store import SessionStore


@st.cache_resource
def init_dependencies() -> dict:
    """Khởi tạo và cache toàn bộ services / agents."""
    embed_service = EmbeddingService()
    rag = NewsRAG(embed_service)
    session_store = SessionStore()
    sentiment_service = SentimentService()
    language_agent = LanguageAgent()
    retrieval_agent = RetrievalAgent(rag)
    summarizer_agent = SummarizerAgent()
    orchestrator = OrchestratorAgent(
        session_store=session_store,
        language_agent=language_agent,
        retrieval_agent=retrieval_agent,
        summarizer_agent=summarizer_agent,
        sentiment_service=sentiment_service,
    )

    return {
        "orchestrator": orchestrator,
        "rag": rag,
        "retrieval": retrieval_agent,
        "sentiment": sentiment_service,
    }


deps = init_dependencies()
orchestrator: OrchestratorAgent = deps["orchestrator"]

st.set_page_config(page_title="Multi-Agent Financial Assistant", layout="wide")
st.title("📊 Multi-Agent Financial Research Assistant (Phase 1 MVP)")
st.caption("Trợ lý đa agent: RAG tin tức + FinBERT sentiment + LLM tóm tắt.")


if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state["history"] = []


with st.sidebar:
    st.header("Bộ lọc nâng cao")
    manual_ticker = st.text_input("Ticker (optional)", "")
    preview_count = st.slider("Số tin hiển thị", 1, 10, 3)
    st.markdown("---")
    st.subheader("Trạng thái hệ thống")
    st.write(f"- Tin tức: {len(deps['rag'].news)} bản ghi")


user_input = st.text_area("Nhập câu hỏi về tài chính (Vi/En)", height=100)
submit = st.button("Phân tích", type="primary")


async def run_orchestrator(prompt: str) -> str:
    return await orchestrator.handle(st.session_state["session_id"], prompt)


if submit and user_input.strip():
    with st.spinner("Đang phân tích..."):
        answer = asyncio.run(run_orchestrator(user_input))

    st.session_state["history"].append(
        {"question": user_input, "answer": answer}
    )

    st.success("Hoàn tất phân tích 🎯")
    st.markdown("### Trả lời")
    st.write(answer)

    preview_query = manual_ticker or user_input
    rag_preview = deps["retrieval"].get_relevant_news(
        preview_query, ticker=manual_ticker or None, top_k=preview_count
    )
    if rag_preview:
        st.markdown("### Tin tức đã sử dụng")
        for item in rag_preview:
            with st.expander(f"{item.date} · {item.title}"):
                st.write(item.content)
                st.caption(f"Ticker: {item.ticker} · ID: {item.id}")
else:
    st.info("Nhập câu hỏi rồi bấm Phân tích để bắt đầu.")


if st.session_state["history"]:
    st.markdown("---")
    st.subheader("Lịch sử trao đổi")
    for idx, entry in enumerate(reversed(st.session_state["history"]), start=1):
        st.markdown(f"**Lần {idx}:** {entry['question']}")
        st.write(entry["answer"])
        st.markdown("---")

