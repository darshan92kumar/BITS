# Microsoft Financial Q&A System - Streamlit App

**Group 78 - Conversational AI Assignment**

## Overview

This Streamlit app demonstrates the comparison between **RAG with Cross-Encoder Re-ranking** and **MoE Fine-tuning** systems for Microsoft financial Q&A.

## Quick Start

### Local Testing

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the app:**
   ```bash
   streamlit run streamlit_app.py
   ```
   
   Or use the launch script:
   ```bash
   python launch_app.py
   ```

3. **Access the app:**
   - Open browser to: http://localhost:8501
   - Navigate through different pages using the sidebar

## App Features

### Overview Page
- System architecture comparison
- Dataset statistics
- Model specifications

### RAG System Page  
- Test RAG with Cross-Encoder re-ranking
- View retrieval details and confidence scores
- See chunk-level information

### MoE System Page
- Test Mixture-of-Experts fine-tuning
- View model architecture details
- Generate answers with specialized experts

### Side-by-Side Comparison
- Compare both systems on same questions
- Performance metrics comparison
- Method comparison table

### Performance Analysis
- Statistical analysis of test results
- Interactive charts and graphs
- Export functionality

### Assignment Requirements
- Submission checklist
- Screenshot guidance  
- Architecture comparison table
- Deployment instructions

## Deployment for Hosted Link

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Streamlit app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Connect your GitHub repository
   - Select `streamlit_app.py` as main file
   - Deploy and get hosted URL

### Option 2: Hugging Face Spaces

1. Create new Space on https://huggingface.co/spaces
2. Upload all files
3. Select Streamlit as SDK
4. Auto-deploy and get URL

## Screenshots for Assignment

Capture these 3 screenshots for your PDF report:

1. **RAG System Test** (from RAG System page)
   - Show question, answer, confidence score, inference time
   
2. **MoE System Test** (from MoE System page)  
   - Show question, answer, confidence score, inference time
   
3. **Side-by-Side Comparison** (from Comparison page)
   - Show both systems answering same question

## File Structure

```
├── streamlit_app.py          # Main Streamlit application
├── rag_system.py            # RAG with Cross-Encoder implementation
├── moe_system.py            # MoE Fine-tuning implementation  
├── data_processor.py        # Data loading and processing
├── requirements.txt         # Python dependencies
├── launch_app.py           # Launch script
├── README_STREAMLIT.md     # This file
└── microsoft_qa_pairs.json # Q&A dataset (if available)
```

## Technical Details

### RAG System
- **Embedding Model:** all-MiniLM-L6-v2
- **Cross-Encoder:** ms-marco-MiniLM-L-6-v2  
- **Vector Store:** FAISS
- **Pipeline:** Bi-Encoder → Cross-Encoder → Answer Generation

### MoE System
- **Base Model:** GPT-2
- **Experts:** 8 per MoE layer
- **MoE Layers:** Positions 4 and 8
- **Top-K:** 2 experts activated per input

## Assignment Compliance

**Advanced Techniques Implemented:**
- Cross-Encoder Re-ranking (RAG)
- Mixture-of-Experts (Fine-tuning)

**Requirements Met:**
- Interactive comparison interface
- Performance metrics and analysis
- Professional hosted app capability
- Clear documentation and screenshots

## Troubleshooting

### Common Issues

1. **Import Errors:**
   - Ensure all files are in same directory
   - Check Python environment and dependencies

2. **Model Loading Issues:**
   - First launch may take time downloading models
   - Ensure internet connection for model downloads

3. **Performance:**
   - RAG system may be slower on first queries (model loading)
   - MoE system loads GPT-2 weights on initialization

### Contact

For technical issues, check the console output and ensure all requirements are installed correctly.

## Deployment Success

Once deployed, your hosted app URL will be something like:
- **Streamlit Cloud:** `https://your-repo-streamlit-app.streamlit.app`
- **Hugging Face:** `https://huggingface.co/spaces/username/space-name`

Include this URL in your PDF report for the "Hosted app link" requirement.