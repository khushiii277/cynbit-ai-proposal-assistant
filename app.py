from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


from pypdf import PdfReader

from src.generator import generate_proposal
from src.retriever import ProjectRetriever



BASE_DIR = Path(__file__).resolve().parent
SAMPLE_RFP = BASE_DIR / "data" / "sample_rfp.txt"


st.set_page_config(
    page_title="Cynbit AI Proposal Assistant",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Cynbit AI Proposal & SOW Assistant")
st.caption(
    "RFP-to-proposal proof of concept • RAG retrieval • Evidence-backed drafting • Human approval required"
)

with st.sidebar:
    st.header("Configuration")
    st.write("Reference corpus: `data/past_projects.json`")
    api_key = st.text_input(
        "OpenAI API key (optional)",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Leave blank to run the transparent template demo without an LLM call.",
    )
    model = st.text_input(
        "Model",
        value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Must be available to your API account.",
    )
    base_url = st.text_input(
        "OpenAI-compatible base URL (optional)",
        value=os.getenv("OPENAI_BASE_URL", ""),
        help="For a compatible provider; leave blank for OpenAI's default endpoint.",
    )
    top_k = st.slider("Reference projects to retrieve", min_value=1, max_value=5, value=3)
    st.divider()
    st.markdown("**Important**")
    st.caption(
        "The bundled reference projects are synthetic demo data. Do not upload confidential "
        "client documents unless your organization's policies authorize the chosen model/provider."
    )

@st.cache_resource
def get_retriever() -> ProjectRetriever:
    return ProjectRetriever()

try:
    retriever = get_retriever()
except Exception as exc:
    st.error(f"Could not load reference projects: {exc}")
    st.stop()

tab_build, tab_library, tab_about = st.tabs(
    ["📝 Generate proposal", "📚 Reference library", "ℹ️ How it works"]
)

with tab_build:
    st.subheader("1. Add a client RFP or brief")
    uploaded = st.file_uploader(
        "Upload a TXT or PDF file",
        type=["txt", "pdf"],
        help="The prototype extracts text only. Scanned/image-only PDFs need OCR before use.",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        use_sample = st.button("Load sample RFP", use_container_width=True)
    with col_b:
        clear_text = st.button("Clear input", use_container_width=True)

    if "rfp_text" not in st.session_state:
        st.session_state.rfp_text = SAMPLE_RFP.read_text(encoding="utf-8")

    if use_sample:
        st.session_state.rfp_text = SAMPLE_RFP.read_text(encoding="utf-8")
    if clear_text:
        st.session_state.rfp_text = ""

    if uploaded is not None:
        try:
            if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
                reader = PdfReader(uploaded)
                extracted = "\n\n".join((page.extract_text() or "") for page in reader.pages)
                if not extracted.strip():
                    st.warning("No selectable text was found. This may be a scanned PDF; use OCR or upload TXT.")
                else:
                    st.session_state.rfp_text = extracted
            else:
                st.session_state.rfp_text = uploaded.getvalue().decode("utf-8", errors="replace")
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")

    rfp_text = st.text_area(
        "RFP / project brief",
        key="rfp_text",
        height=280,
        placeholder="Paste the client RFP or upload a file above...",
    )

    generate = st.button(
        "✨ Retrieve references and draft proposal",
        type="primary",
        use_container_width=True,
        disabled=not rfp_text.strip(),
    )

    if generate:
        with st.spinner("Retrieving relevant projects..."):
            try:
                references = retriever.search(rfp_text, top_k=top_k)
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")
                references = []

        with st.spinner("Preparing the proposal draft..."):
            proposal, mode = generate_proposal(
                rfp_text=rfp_text,
                references=references,
                model=model.strip() or None,
                api_key=api_key.strip() or None,
                base_url=base_url.strip() or None,
            )

        st.session_state.last_proposal = proposal
        st.session_state.last_references = references
        st.session_state.last_mode = mode
        st.session_state.last_generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if st.session_state.get("last_proposal"):
        st.divider()
        st.subheader("2. Review generated draft")
        st.caption(
            f"Mode: **{st.session_state.get('last_mode', 'Unknown')}** · "
            f"Generated at {st.session_state.get('last_generated_at', 'unknown time')}"
        )
        st.warning(
            "Review every claim, estimate, technology recommendation and contractual detail. "
            "This prototype does not send proposals to clients."
        )

        out_col, ref_col = st.columns([1.6, 1])
        with out_col:
            edited_proposal = st.text_area(
                "Editable proposal (Markdown)",
                value=st.session_state.last_proposal,
                height=650,
                key="edited_proposal",
            )
            st.download_button(
                "⬇️ Download proposal (.md)",
                data=edited_proposal,
                file_name="cynbit_proposal_draft.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with ref_col:
            st.markdown("#### Retrieved evidence")
            refs = st.session_state.get("last_references", [])
            if not refs:
                st.info("No reference projects were matched. Consider adding more approved examples.")
            for ref in refs:
                with st.expander(
                    f"{ref.get('project_id')} · {ref.get('title')} "
                    f"(score {ref.get('retrieval_score', 0):.3f})",
                    expanded=True,
                ):
                    st.write(ref.get("summary", ""))
                    st.caption(f"Domain: {ref.get('domain', 'Unknown')}")
                    st.write("**Technologies**")
                    st.write(", ".join(ref.get("technologies", [])))
                    st.write("**Deliverables**")
                    for item in ref.get("deliverables", []):
                        st.write(f"- {item}")
                    st.caption(f"Retrieval method: {ref.get('retrieval_method', 'unknown')}")
                    st.caption(ref.get("note", ""))

        with st.expander("Preview rendered Markdown"):
            st.markdown(st.session_state.get("edited_proposal", st.session_state.last_proposal))

with tab_library:
    st.subheader("Reference project library")
    st.info(
        "All bundled records are synthetic examples created for this assessment prototype. "
        "They must not be represented as verified Cynbit engagements."
    )
    for project in retriever.projects:
        with st.expander(f"{project['project_id']} · {project['title']}"):
            st.markdown(f"**Domain:** {project.get('domain', 'N/A')}")
            st.write(project.get("summary", ""))
            st.markdown("**Solution**")
            st.write(project.get("solution", ""))
            st.markdown("**Technologies**")
            st.write(", ".join(project.get("technologies", [])))
            st.markdown("**Effort (synthetic reference only)**")
            st.json(project.get("effort_person_days", {}))
            st.caption(project.get("note", ""))

with tab_about:
    st.subheader("System architecture")
    st.markdown(
        """
        1. **Input:** user uploads a text-based PDF/TXT or pastes a brief.
        2. **Retrieval:** Sentence Transformers creates embeddings; FAISS returns similar reference projects.
        3. **Fallback:** if model weights cannot be downloaded, keyword retrieval is used.
        4. **Generation:** an OpenAI-compatible chat-completions API drafts the proposal when a key is configured.
        5. **Validation:** the interface makes the human-review requirement visible and supports Markdown export.
        """
    )
    st.subheader("Current limitations")
    st.markdown(
        """
        - Sample project records are synthetic, not actual Cynbit history.
        - PDF support extracts selectable text only; OCR is not included.
        - The prototype does not implement authentication, multi-user permissions, persistent audit logs, or production document storage.
        - Effort estimates are not calibrated against actual delivery history.
        - No proposal is emailed or submitted automatically.
        """
    )
    st.subheader("Suggested production next steps")
    st.markdown(
        """
        - Ingest only approved historical proposals and verified delivery metrics.
        - Add authentication, tenant isolation, access control and retention policies.
        - Evaluate retrieval and groundedness using a human-reviewed test set.
        - Add structured JSON schemas for requirements, risks, milestones and estimates.
        - Add DOCX export, version history, auditability and reviewer approval.
        """
    )
