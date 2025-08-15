"""
Streamlit App for Microsoft Financial Q&A System Comparison
RAG vs MoE Fine-tuning Systems

Group 78 - Conversational AI Assignment
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our custom modules
try:
    from rag_system import RAGSystem
    from moe_system import FineTuningSystem
    from data_processor import get_sample_data
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please make sure rag_system.py, moe_system.py, and data_processor.py are in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="RAG vs MoE Fine-tuning Comparison",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .system-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'systems_loaded' not in st.session_state:
    st.session_state.systems_loaded = False
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'moe_system' not in st.session_state:
    st.session_state.moe_system = None
if 'test_results' not in st.session_state:
    st.session_state.test_results = []

@st.cache_data
def load_data():
    """Load sample data with caching."""
    return get_sample_data()

@st.cache_resource
def initialize_systems():
    """Initialize both RAG and MoE systems with caching."""
    try:
        chunks_data, qa_pairs = load_data()
        
        # Initialize RAG system
        rag_system = RAGSystem(chunks_data)
        rag_system.create_embeddings()
        rag_system.build_faiss_index()
        
        # Initialize MoE system
        moe_system = FineTuningSystem()
        
        return rag_system, moe_system, chunks_data, qa_pairs
    except Exception as e:
        st.error(f"Error initializing systems: {e}")
        return None, None, None, None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 RAG vs MoE Fine-tuning Comparison</h1>', unsafe_allow_html=True)
    st.markdown("**Group 78 - Conversational AI Assignment**")
    st.markdown("**Advanced Techniques: Cross-Encoder Re-ranking (RAG) vs Mixture-of-Experts (Fine-tuning)**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "🔍 RAG System", "🧠 MoE System", "⚖️ Side-by-Side Comparison", "📊 Performance Analysis", "📋 Assignment Requirements"]
    )
    
    # Load systems
    if not st.session_state.systems_loaded:
        with st.spinner("🚀 Loading AI systems..."):
            rag_system, moe_system, chunks_data, qa_pairs = initialize_systems()
            if rag_system and moe_system:
                st.session_state.rag_system = rag_system
                st.session_state.moe_system = moe_system
                st.session_state.chunks_data = chunks_data
                st.session_state.qa_pairs = qa_pairs
                st.session_state.systems_loaded = True
                st.success("✅ Both systems loaded successfully!")
            else:
                st.error("❌ Failed to load systems")
                return
    
    # Route to selected page
    if page == "🏠 Overview":
        show_overview()
    elif page == "🔍 RAG System":
        show_rag_system()
    elif page == "🧠 MoE System":
        show_moe_system()
    elif page == "⚖️ Side-by-Side Comparison":
        show_comparison()
    elif page == "📊 Performance Analysis":
        show_performance_analysis()
    elif page == "📋 Assignment Requirements":
        show_assignment_requirements()

def show_overview():
    """Show system overview page."""
    st.header("📋 System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 RAG System")
        st.write("""
        **Retrieval Augmented Generation with Cross-Encoder Re-ranking**
        
        **Architecture:**
        - Bi-Encoder: Fast semantic search using SentenceTransformers
        - Cross-Encoder: Accurate re-ranking for improved relevance
        - FAISS: Efficient vector similarity search
        
        **Advanced Technique:** Cross-Encoder Re-ranking improves retrieval quality by jointly encoding query and document pairs.
        """)
        
        # RAG metrics
        if st.session_state.systems_loaded:
            chunks_data = st.session_state.chunks_data
            total_chunks = sum(len(chunks) for chunks in chunks_data.values())
            st.metric("Total Chunks", total_chunks)
            st.metric("Embedding Dimension", "384")
            st.metric("Cross-Encoder Model", "ms-marco-MiniLM-L-6-v2")
    
    with col2:
        st.subheader("🧠 MoE System")
        st.write("""
        **Mixture-of-Experts Fine-tuning**
        
        **Architecture:**
        - Base Model: GPT-2 with MoE layers
        - Expert Networks: 8 specialized experts per layer
        - Gating Network: Routes inputs to top-2 experts
        - Load Balancing: Ensures equal expert utilization
        
        **Advanced Technique:** MoE architecture increases model capacity while maintaining efficiency through sparse activation.
        """)
        
        # MoE metrics
        if st.session_state.systems_loaded:
            moe_stats = st.session_state.moe_system.get_model_stats()
            st.metric("Total Parameters", f"{moe_stats['total_parameters']:,}")
            st.metric("MoE Layers", moe_stats['moe_layers'])
            st.metric("Experts per Layer", moe_stats['experts_per_layer'])
    
    # Data overview
    st.subheader("📊 Dataset Overview")
    if st.session_state.systems_loaded:
        qa_pairs = st.session_state.qa_pairs
        df = pd.DataFrame(qa_pairs)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Q&A Pairs", len(qa_pairs))
        with col2:
            st.metric("Categories", len(df['category'].unique()) if 'category' in df.columns else "N/A")
        with col3:
            avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else 0
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")

def show_rag_system():
    """Show RAG system testing page."""
    st.header("🔍 RAG System with Cross-Encoder Re-ranking")
    
    st.write("""
    **Two-Stage Retrieval Process:**
    1. **Bi-Encoder**: Retrieves candidate chunks using semantic similarity
    2. **Cross-Encoder**: Re-ranks candidates for improved relevance
    """)
    
    # System status
    if st.session_state.systems_loaded:
        st.success("✅ RAG System Ready")
        
        # Question input
        st.subheader("Ask a Question")
        question = st.text_input(
            "Enter your question about Microsoft's financials:",
            placeholder="What was Microsoft's total revenue in fiscal year 2024?"
        )
        
        # Predefined questions
        st.subheader("Or choose a predefined question:")
        predefined_questions = [
            "What was Microsoft's total revenue in fiscal year 2024?",
            "What are Microsoft's three main business segments?",
            "How did Microsoft's operating income change from 2023 to 2024?",
            "What was Microsoft's operating margin in 2024?",
            "How much did Microsoft spend on research and development in 2024?"
        ]
        
        selected_question = st.selectbox("Select a question:", [""] + predefined_questions)
        if selected_question:
            question = selected_question
        
        # Process question
        if question and st.button("🔍 Search with RAG"):
            with st.spinner("Processing with RAG system..."):
                result = st.session_state.rag_system.search(question)
                
                # Display results
                st.subheader("📋 RAG Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Inference Time", f"{result['inference_time']:.2f}s")
                with col2:
                    st.metric("Confidence Score", f"{result['confidence_score']:.3f}")
                with col3:
                    st.metric("Chunks Retrieved", f"{result['num_candidates']} → {result['num_final']}")
                
                # Answer
                st.markdown(f"""
                <div class="answer-box">
                    <strong>Question:</strong> {result['query']}<br><br>
                    <strong>Answer:</strong> {result['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Retrieval details
                st.subheader("🔍 Retrieval Details")
                if result['retrieved_chunks']:
                    for i, chunk in enumerate(result['retrieved_chunks'][:3]):
                        with st.expander(f"Chunk {i+1} (Score: {chunk['cross_encoder_score']:.3f})"):
                            st.write(f"**Year:** {chunk['year']}")
                            st.write(f"**Section:** {chunk['section']}")
                            st.write(f"**Text:** {chunk['text']}")

def show_moe_system():
    """Show MoE system testing page."""
    st.header("🧠 MoE Fine-tuning System")
    
    st.write("""
    **Mixture-of-Experts Architecture:**
    - **8 Expert Networks** specialized for different financial domains
    - **Gating Network** routes inputs to top-2 relevant experts
    - **Load Balancing** ensures equal expert utilization
    """)
    
    # System status
    if st.session_state.systems_loaded:
        st.success("✅ MoE System Ready")
        
        # Model stats
        st.subheader("📊 Model Architecture")
        moe_stats = st.session_state.moe_system.get_model_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Parameters", f"{moe_stats['total_parameters']:,}")
        with col2:
            st.metric("MoE Layers", moe_stats['moe_layers'])
        with col3:
            st.metric("Experts per Layer", moe_stats['experts_per_layer'])
        with col4:
            st.metric("Top-K Experts", moe_stats['top_k_experts'])
        
        # Question input
        st.subheader("Ask a Question")
        question = st.text_input(
            "Enter your question about Microsoft's financials:",
            placeholder="What was Microsoft's total revenue in fiscal year 2024?",
            key="moe_question"
        )
        
        # Predefined questions
        st.subheader("Or choose a predefined question:")
        predefined_questions = [
            "What was Microsoft's total revenue in fiscal year 2024?",
            "What are Microsoft's three main business segments?",
            "How did Microsoft's operating income change from 2023 to 2024?",
            "What was Microsoft's operating margin in 2024?",
            "How much did Microsoft spend on research and development in 2024?"
        ]
        
        selected_question = st.selectbox("Select a question:", [""] + predefined_questions, key="moe_select")
        if selected_question:
            question = selected_question
        
        # Process question
        if question and st.button("🧠 Generate with MoE"):
            with st.spinner("Generating with MoE system..."):
                result = st.session_state.moe_system.generate_answer(question)
                
                # Display results
                st.subheader("📋 MoE Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Inference Time", f"{result['inference_time']:.2f}s")
                with col2:
                    st.metric("Confidence Score", f"{result['confidence_score']:.3f}")
                
                # Answer
                st.markdown(f"""
                <div class="answer-box">
                    <strong>Question:</strong> {result['question']}<br><br>
                    <strong>Answer:</strong> {result['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Model info
                st.subheader("🔧 Model Information")
                model_info = result['model_info']
                st.write(f"**Base Model:** {model_info['base_model']}")
                st.write(f"**Experts:** {model_info['experts']}")
                st.write(f"**MoE Layers:** {model_info['moe_layers']}")
                st.write(f"**Parameters:** {model_info['parameters']:,}")

def show_comparison():
    """Show side-by-side comparison page."""
    st.header("⚖️ Side-by-Side Comparison")
    
    st.write("Compare both systems on the same question to see their different approaches and results.")
    
    if st.session_state.systems_loaded:
        # Question input
        question = st.text_input(
            "Enter your question for comparison:",
            placeholder="What was Microsoft's total revenue in fiscal year 2024?"
        )
        
        # Predefined questions
        predefined_questions = [
            "What was Microsoft's total revenue in fiscal year 2024?",
            "What are Microsoft's three main business segments?",
            "How did Microsoft's operating income change from 2023 to 2024?",
            "What was Microsoft's operating margin in 2024?",
            "How much did Microsoft spend on research and development in 2024?"
        ]
        
        selected_question = st.selectbox("Or select a predefined question:", [""] + predefined_questions)
        if selected_question:
            question = selected_question
        
        # Compare button
        if question and st.button("⚖️ Compare Both Systems"):
            with st.spinner("Running comparison..."):
                try:
                    # Get results from both systems
                    rag_result = st.session_state.rag_system.search(question)
                    moe_result = st.session_state.moe_system.generate_answer(question)
                except Exception as e:
                    st.error(f"Error during comparison: {str(e)}")
                    st.error("Please try again or use a different question.")
                    return
                
                # Store results for analysis
                comparison_result = {
                    'question': question,
                    'timestamp': datetime.now().isoformat(),
                    'rag_result': rag_result,
                    'moe_result': moe_result
                }
                st.session_state.test_results.append(comparison_result)
                
                # Display comparison
                st.subheader("📊 Comparison Results")
                
                # Metrics comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<h3 class="system-header">🔍 RAG System</h3>', unsafe_allow_html=True)
                    
                    # Metrics
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        st.metric("Inference Time", f"{rag_result['inference_time']:.2f}s")
                    with subcol2:
                        st.metric("Confidence", f"{rag_result['confidence_score']:.3f}")
                    
                    # Answer
                    st.markdown(f"""
                    <div class="answer-box">
                        <strong>Answer:</strong><br>
                        {rag_result['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Method details
                    st.write(f"**Method:** {rag_result['method']}")
                    st.write(f"**Chunks Retrieved:** {rag_result['num_candidates']} → {rag_result['num_final']}")
                
                with col2:
                    st.markdown('<h3 class="system-header">🧠 MoE System</h3>', unsafe_allow_html=True)
                    
                    # Metrics
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        st.metric("Inference Time", f"{moe_result['inference_time']:.2f}s")
                    with subcol2:
                        st.metric("Confidence", f"{moe_result['confidence_score']:.3f}")
                    
                    # Answer
                    st.markdown(f"""
                    <div class="answer-box">
                        <strong>Answer:</strong><br>
                        {moe_result['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Method details
                    st.write(f"**Method:** {moe_result['method']}")
                    st.write(f"**Model:** {moe_result['model_info']['base_model']} with {moe_result['model_info']['experts']} experts")
                
                # Performance comparison
                st.subheader("⚡ Performance Comparison")
                
                comparison_data = {
                    'Metric': ['Inference Time (s)', 'Confidence Score', 'Method'],
                    'RAG System': [
                        f"{rag_result['inference_time']:.2f}",
                        f"{rag_result['confidence_score']:.3f}",
                        rag_result['method']
                    ],
                    'MoE System': [
                        f"{moe_result['inference_time']:.2f}",
                        f"{moe_result['confidence_score']:.3f}",
                        moe_result['method']
                    ]
                }
                
                df_comparison = pd.DataFrame(comparison_data)
                st.table(df_comparison)

def show_performance_analysis():
    """Show performance analysis page."""
    st.header("📊 Performance Analysis")
    
    if not st.session_state.test_results:
        st.info("No test results available. Please run some comparisons first.")
        return
    
    # Analysis of test results
    results = st.session_state.test_results
    
    # Extract metrics
    rag_times = [r['rag_result']['inference_time'] for r in results]
    moe_times = [r['moe_result']['inference_time'] for r in results]
    rag_confidence = [r['rag_result']['confidence_score'] for r in results]
    moe_confidence = [r['moe_result']['confidence_score'] for r in results]
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 RAG System**")
        st.metric("Avg Inference Time", f"{np.mean(rag_times):.2f}s")
        st.metric("Avg Confidence", f"{np.mean(rag_confidence):.3f}")
        st.metric("Total Tests", len(results))
    
    with col2:
        st.markdown("**🧠 MoE System**")
        st.metric("Avg Inference Time", f"{np.mean(moe_times):.2f}s") 
        st.metric("Avg Confidence", f"{np.mean(moe_confidence):.3f}")
        st.metric("Total Tests", len(results))
    
    # Performance charts
    if len(results) > 1:
        st.subheader("📊 Performance Charts")
        
        # Inference time comparison
        fig_time = go.Figure()
        fig_time.add_trace(go.Box(y=rag_times, name="RAG System", marker_color="lightblue"))
        fig_time.add_trace(go.Box(y=moe_times, name="MoE System", marker_color="lightgreen"))
        fig_time.update_layout(title="Inference Time Comparison", yaxis_title="Time (seconds)")
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Confidence score comparison
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Box(y=rag_confidence, name="RAG System", marker_color="lightblue"))
        fig_conf.add_trace(go.Box(y=moe_confidence, name="MoE System", marker_color="lightgreen"))
        fig_conf.update_layout(title="Confidence Score Comparison", yaxis_title="Confidence Score")
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    detailed_data = []
    for i, result in enumerate(results):
        detailed_data.append({
            'Test #': i + 1,
            'Question': result['question'][:50] + "..." if len(result['question']) > 50 else result['question'],
            'RAG Time (s)': f"{result['rag_result']['inference_time']:.2f}",
            'RAG Confidence': f"{result['rag_result']['confidence_score']:.3f}",
            'MoE Time (s)': f"{result['moe_result']['inference_time']:.2f}",
            'MoE Confidence': f"{result['moe_result']['confidence_score']:.3f}",
            'Timestamp': result['timestamp']
        })
    
    df_detailed = pd.DataFrame(detailed_data)
    st.dataframe(df_detailed, use_container_width=True)
    
    # Export results
    if st.button("📥 Export Results as JSON"):
        json_str = json.dumps(st.session_state.test_results, indent=2)
        st.download_button(
            label="Download Results",
            data=json_str,
            file_name=f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def show_assignment_requirements():
    """Show assignment requirements and submission checklist."""
    st.header("📋 Assignment Requirements Checklist")
    
    st.write("**Group 78 - Advanced Techniques: Cross-Encoder Re-ranking (RAG) vs Mixture-of-Experts (Fine-tuning)**")
    
    # Submission requirements
    st.subheader("📦 Submission Requirements")
    
    requirements = [
        "Python Notebook (.ipynb) with data processing steps ✅",
        "RAG implementation with Cross-Encoder re-ranking ✅", 
        "Fine-tuning implementation with Mixture-of-Experts ✅",
        "Advanced technique sections with markdown explanations ✅",
        "Testing and comparison tables ✅",
        "PDF report with 3 screenshots ⏳",
        "Summary comparison table ✅",
        "Hosted app link (this Streamlit app) ✅",
        "Open-source models and software only ✅",
        "Clear code documentation and comments ✅"
    ]
    
    for req in requirements:
        if "✅" in req:
            st.success(req)
        elif "⏳" in req:
            st.warning(req)
        else:
            st.info(req)
    
    # Screenshots guidance
    st.subheader("📸 Screenshot Requirements")
    st.write("""
    For your PDF report, capture these 3 screenshots from this app:
    
    1. **RAG System Test**: Show test query, answer, confidence score, and inference time
    2. **MoE System Test**: Show test query, answer, confidence score, and inference time  
    3. **Side-by-Side Comparison**: Show both systems comparing answers to the same question
    
    Each screenshot should clearly show:
    - The question asked
    - The system's answer
    - Confidence score
    - Inference time
    - Method used (RAG vs MoE)
    """)
    
    # Architecture comparison table
    st.subheader("📊 Architecture Comparison Table")
    
    comparison_data = {
        'Aspect': [
            'Approach',
            'Architecture', 
            'Knowledge Source',
            'Inference Speed',
            'Memory Usage',
            'Customization',
            'Training Required',
            'Factual Accuracy',
            'Scalability'
        ],
        'RAG System': [
            'Retrieval + Generation',
            'Bi-Encoder + Cross-Encoder',
            'External Knowledge Base',
            'Slower (retrieval + re-ranking)',
            'Lower (no model fine-tuning)',
            'Moderate (retrieval tuning)',
            'No (uses pre-trained models)',
            'High (grounded in documents)',
            'High (add more documents)'
        ],
        'MoE Fine-tuning': [
            'Specialized Model Training',
            'GPT-2 + Mixture of Experts',
            'Internal Knowledge (weights)',
            'Faster (direct inference)',
            'Higher (stores knowledge)',
            'High (expert specialization)',
            'Yes (domain-specific training)',
            'Variable (depends on training)',
            'Moderate (requires retraining)'
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.table(df_comparison)
    
    # Deployment information
    st.subheader("🌐 Deployment Information")
    st.write("""
    **Current Status:** Running locally
    
    **For Hosted App Link:**
    1. Push this code to GitHub repository
    2. Connect to Streamlit Cloud (share.streamlit.io)
    3. Deploy and get public URL
    4. Include URL in your PDF report
    
    **Files needed for deployment:**
    - streamlit_app.py (this file)
    - rag_system.py
    - moe_system.py  
    - data_processor.py
    - requirements.txt
    """)
    
    # Requirements.txt content
    with st.expander("📄 requirements.txt content"):
        st.code("""
streamlit
pandas
numpy
plotly
torch
transformers
sentence-transformers
faiss-cpu
datasets
scikit-learn
        """)

if __name__ == "__main__":
    main()