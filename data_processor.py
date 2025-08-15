"""
Financial Data Processor for Microsoft Q&A System
"""

import os
import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict
import pdfplumber

class FinancialDataProcessor:
    """Process Microsoft financial data for Q&A system training and evaluation."""
    
    def __init__(self):
        self.raw_data = {}
        self.processed_data = {}
        
    def load_sample_chunks(self) -> Dict[str, List[Dict]]:
        """Load sample financial chunks for demo purposes."""
        # Sample chunks from Microsoft financial data
        sample_chunks = {
            'chunks_100': [
                {
                    'id': 'chunk_0',
                    'text': 'Microsoft Corporation - Fiscal Year 2024 Financial Highlights. Total Revenue: $245.1 billion (16% increase year-over-year). Operating Income: $109.4 billion (24% increase).',
                    'year': 'fy2024',
                    'chunk_size': 100,
                    'token_count': 32,
                    'section': 'Revenue'
                },
                {
                    'id': 'chunk_1', 
                    'text': 'Microsoft operates through three main business segments: 1) Productivity and Business Processes, 2) Intelligent Cloud, and 3) More Personal Computing.',
                    'year': 'fy2024',
                    'chunk_size': 100,
                    'token_count': 24,
                    'section': 'Segments'
                },
                {
                    'id': 'chunk_2',
                    'text': 'Intelligent Cloud revenue was $87.9 billion in fiscal year 2024, with Azure and other cloud services generating $60.6 billion (29% growth).',
                    'year': 'fy2024', 
                    'chunk_size': 100,
                    'token_count': 28,
                    'section': 'Intelligent Cloud'
                },
                {
                    'id': 'chunk_3',
                    'text': 'Microsoft spent $29.5 billion on research and development in fiscal year 2024, focusing on AI, cloud computing, and productivity solutions.',
                    'year': 'fy2024',
                    'chunk_size': 100,
                    'token_count': 26,
                    'section': 'Research and Development'
                },
                {
                    'id': 'chunk_4',
                    'text': 'Operating margin in fiscal year 2024 was 44.7%, demonstrating strong operational efficiency and pricing power in core markets.',
                    'year': 'fy2024',
                    'chunk_size': 100,
                    'token_count': 22,
                    'section': 'Metrics'
                }
            ],
            'chunks_400': [
                {
                    'id': 'chunk_large_0',
                    'text': 'Microsoft Corporation achieved exceptional financial performance in fiscal year 2024, with total revenue reaching $245.1 billion, representing a 16% increase year-over-year. This growth was driven by strong performance across all three business segments. Operating income increased to $109.4 billion, a 24% year-over-year increase, demonstrating improved operational efficiency. The Intelligent Cloud segment led growth with revenue of $87.9 billion, while Productivity and Business Processes generated $69.3 billion, and More Personal Computing contributed $59.7 billion.',
                    'year': 'fy2024',
                    'chunk_size': 400,
                    'token_count': 85,
                    'section': 'Performance'
                },
                {
                    'id': 'chunk_large_1',
                    'text': 'Microsoft\'s balance sheet remained strong with total assets of $512.1 billion as of June 30, 2024. Cash and cash equivalents totaled $75.5 billion, providing substantial financial flexibility. Total stockholders\' equity was $206.2 billion, while total debt was $97.8 billion. The company returned significant value to shareholders through dividends of $3.00 per share and share repurchases of $16.1 billion during the fiscal year.',
                    'year': 'fy2024',
                    'chunk_size': 400,
                    'token_count': 72,
                    'section': 'Balance Sheet'
                }
            ]
        }
        
        # Add more sample chunks for demonstration
        for i in range(10):
            sample_chunks['chunks_100'].append({
                'id': f'chunk_{i+5}',
                'text': f'Sample financial data chunk {i+5} containing Microsoft financial information and metrics for analysis.',
                'year': 'fy2024' if i % 2 == 0 else 'fy2023',
                'chunk_size': 100,
                'token_count': 15,
                'section': 'general'
            })
        
        return sample_chunks
    
    def load_qa_pairs(self) -> List[Dict]:
        """Load Q&A pairs from JSON file."""
        try:
            with open('microsoft_qa_pairs.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return sample Q&A pairs if file not found
            return [
                {
                    "question": "What was Microsoft's total revenue in fiscal year 2024?",
                    "answer": "Microsoft's total revenue in fiscal year 2024 was $245.1 billion, representing a 16% increase year-over-year.",
                    "category": "performance",
                    "confidence": 0.95,
                    "year": "fy2024"
                },
                {
                    "question": "What are Microsoft's three main business segments?",
                    "answer": "Microsoft operates through three main business segments: 1) Productivity and Business Processes, 2) Intelligent Cloud, and 3) More Personal Computing.",
                    "category": "segments", 
                    "confidence": 0.98,
                    "year": "both"
                },
                {
                    "question": "How did Microsoft's operating income change from 2023 to 2024?",
                    "answer": "Microsoft's operating income increased from $88.5 billion in fiscal year 2023 to $109.4 billion in fiscal year 2024, representing a 24% increase year-over-year.",
                    "category": "comparative",
                    "confidence": 0.93,
                    "year": "both"
                },
                {
                    "question": "What was Microsoft's operating margin in 2024?",
                    "answer": "Microsoft's operating margin in fiscal year 2024 was 44.7%.",
                    "category": "metrics",
                    "confidence": 0.90,
                    "year": "fy2024"
                },
                {
                    "question": "How much did Microsoft spend on research and development in 2024?",
                    "answer": "Microsoft spent $29.5 billion on research and development in fiscal year 2024.",
                    "category": "operations",
                    "confidence": 0.95,
                    "year": "fy2024"
                }
            ]

def get_sample_data():
    """Get sample data for Streamlit app."""
    processor = FinancialDataProcessor()
    chunks_data = processor.load_sample_chunks()
    qa_pairs = processor.load_qa_pairs()
    
    return chunks_data, qa_pairs