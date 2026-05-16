"""
MnemoSync Streamlit demo.
Three tabs: Drift Timeline | Intent Classifier | Memory Query
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="MnemoSync",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def get_classifier():
    from classifier.intent_classifier import IntentClassifier
    clf = IntentClassifier()
    clf._ensure_loaded()
    return clf


@st.cache_resource
def get_memory_store():
    from rag.memory_store import MemoryStore
    from rag.retrieval_engine import load_persona_into_store
    store = MemoryStore()
    load_persona_into_store(store)
    store.rebuild_index()
    return store


@st.cache_data
def get_timeline():
    from drift.drift_engine import compute_drift_timeline
    return compute_drift_timeline()


st.title("🧠 MnemoSync")
st.caption("Local-first memory integrity engine · persona drift · intent classification · conflict resolution")
st.divider()

tab1, tab2, tab3 = st.tabs(["📈 Drift Timeline", "🏷️ Intent Classifier", "🔍 Memory Query"])


# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Persona Drift Timeline")
    st.markdown("Drift score = 1 - cosine similarity between consecutive day embeddings.")

    with st.spinner("Computing drift timeline..."):
        timeline = get_timeline()

    df = pd.DataFrame(timeline)

    fig_drift = go.Figure()
    fig_drift.add_trace(go.Scatter(
        x=df["day"], y=df["drift_score"],
        mode="lines+markers",
        name="Drift Score",
        line=dict(color="#ef4444", width=2),
        marker=dict(size=8),
        hovertemplate="Day %{x}<br>Drift: %{y:.3f}<extra></extra>"
    ))
    fig_drift.add_hrect(y0=0, y1=0.15, fillcolor="green", opacity=0.05, line_width=0, annotation_text="stable")
    fig_drift.add_hrect(y0=0.15, y1=0.30, fillcolor="yellow", opacity=0.05, line_width=0, annotation_text="mild")
    fig_drift.add_hrect(y0=0.30, y1=0.50, fillcolor="orange", opacity=0.05, line_width=0, annotation_text="moderate")
    fig_drift.add_hrect(y0=0.50, y1=1.1, fillcolor="red", opacity=0.05, line_width=0, annotation_text="significant")
    fig_drift.update_layout(
        title="Semantic Drift Score by Day",
        xaxis_title="Day", yaxis_title="Drift Score (0-1)",
        height=320, showlegend=False,
    )
    st.plotly_chart(fig_drift, use_container_width=True)

    fig_sentiment = go.Figure()
    fig_sentiment.add_trace(go.Bar(
        x=df["day"], y=df["sentiment_compound"],
        marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in df["sentiment_compound"]],
        hovertemplate="Day %{x}<br>Compound: %{y:.3f}<extra></extra>"
    ))
    fig_sentiment.update_layout(
        title="Sentiment Compound Score by Day",
        xaxis_title="Day", yaxis_title="VADER Compound",
        height=280,
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

    st.markdown("### Day-by-Day Summary")
    for entry in timeline:
        severity_color = {"stable": "🟢", "mild": "🟡", "moderate": "🟠", "significant": "🔴"}.get(entry["drift_severity"], "⚪")
        with st.expander(f"Day {entry['day']} — {entry['mood']} & {entry['tone']} {severity_color}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Mood", entry["mood"])
            col2.metric("Tone", entry["tone"])
            col3.metric("Drift Score", f"{entry['drift_score']:.3f}")
            if entry["trigger"]:
                st.info(f"**Trigger:** {entry['trigger']}")
            st.caption(f"Topics: {', '.join(entry['topics'])}")


# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Offline Intent Classifier")
    st.markdown("TF-IDF char n-grams + calibrated LinearSVC. Model: **~504 KB**. Inference: **~2ms** on CPU.")

    clf = get_classifier()

    user_input = st.text_input(
        "Enter a message:",
        placeholder="e.g. remind me to call mum tomorrow",
        key="clf_input"
    )

    st.caption("Try an example:")
    cols = st.columns(5)
    examples = [
        ("reminder", "remind me to call mum"),
        ("emotional-support", "i'm feeling really overwhelmed"),
        ("action-item", "fix the failing tests before the standup"),
        ("small-talk", "lol nothing much, just chilling"),
        ("unknown", "asdf jkl;"),
    ]
    for i, (label, text) in enumerate(examples):
        if cols[i].button(label, key=f"ex_{i}"):
            user_input = text

    if user_input:
        result = clf.predict(user_input)
        label_icons = {
            "reminder": "🔔", "emotional-support": "💙",
            "action-item": "✅", "small-talk": "💬", "unknown": "❓",
        }
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Predicted Intent", f"{label_icons.get(result['label'], '❓')} {result['label']}")
            st.metric("Confidence", f"{result['confidence']:.1%}")
        with col2:
            if "all_scores" in result:
                scores_df = pd.DataFrame([
                    {"label": k, "probability": v}
                    for k, v in result["all_scores"].items()
                ]).sort_values("probability", ascending=True)
                fig = px.bar(scores_df, x="probability", y="label", orientation="h",
                             color="probability", color_continuous_scale="blues", range_x=[0, 1])
                fig.update_layout(height=220, showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Known limitation:** CV F1 ~0.65 on 125 training examples. Short/ambiguous messages like `I'm fine` may misclassify.")


# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Memory Query with Conflict Resolution")
    st.markdown("Ranking: `0.5 x semantic + 0.3 x recency + 0.2 x emotion`")

    store = get_memory_store()

    query_input = st.text_input(
        "Ask about the user's memory:",
        value="Did I mention anything about my sister?",
        key="query_input"
    )

    if st.button("Query Memory", type="primary"):
        from rag.retrieval_engine import query as retrieve
        from rag.conflict_resolver import detect_contradictions, resolve_conflicts

        with st.spinner("Retrieving and resolving..."):
            chunks = retrieve(store, query_input, top_k=6)
            contradictions = detect_contradictions(chunks)
            resolution = resolve_conflicts(query_input, chunks, contradictions)

        if resolution["has_contradiction"]:
            st.warning(f"Contradiction detected: {resolution['contradiction_note']}")
        else:
            st.success("No contradictions detected.")

        st.markdown("### Merged Narrative")
        st.info(resolution["merged_narrative"])

        st.markdown("### Retrieved Chunks")
        for chunk in resolution["chunks"]:
            status_icon = "🔴" if chunk["status"] == "contradicted" else "🟢"
            with st.expander(
                f"{status_icon} Day {chunk['day']} — {chunk['polarity']} [{chunk['status']}] "
                f"(score: {chunk['scores']['final']:.3f})"
            ):
                st.write(chunk["text"])
                col1, col2, col3 = st.columns(3)
                col1.metric("Recency", f"{chunk['scores']['recency']:.3f}")
                col2.metric("Emotion", f"{chunk['scores']['emotion']:.3f}")
                col3.metric("Final", f"{chunk['scores']['final']:.3f}")

        if resolution["contradiction_pairs"]:
            st.markdown("### Contradiction Details")
            for pair in resolution["contradiction_pairs"]:
                with st.expander(f"Day {pair['day_a']} vs Day {pair['day_b']}"):
                    col1, col2 = st.columns(2)
                    col1.markdown(f"**Day {pair['day_a']}** ({pair['polarity_a']})")
                    col1.write(pair["text_a"])
                    col2.markdown(f"**Day {pair['day_b']}** ({pair['polarity_b']})")
                    col2.write(pair["text_b"])
                    st.caption(pair["explanation"])

        with st.expander("Source Trace"):
            for s in resolution["source_trace"]:
                st.caption(f"id={s['id']} | Day {s['day']} | [{s['status']}]")