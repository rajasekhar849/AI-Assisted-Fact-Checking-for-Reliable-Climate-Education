# SRAG Enhanced YouTube Climate Fact-Checker with Self-Retrieval

import streamlit as st
import openai
from openai import OpenAI
import json
import requests
from typing import List, Dict, Any, Optional, Tuple, Union
import time
from dataclasses import dataclass
import re
from youtube_transcript_api._api import YouTubeTranscriptApi
import chromadb
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
import os
import PyPDF2
from io import BytesIO
import fitz  # PyMuPDF for better PDF handling
from typing import Union
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize OpenAI client
client = None

# Configuration
@dataclass
class SRAGClimateFactCheckerConfig:
    openai_api_key: str = ""
    chroma_persist_directory: str = "./ipcc_ar6_db"
    chunk_size: int = 500
    chunk_overlap: int = 100
    max_chunks_to_analyze: int = 5
    top_k_retrieval: int = 5
    ipcc_pdf_path: str = "ipcc_ar6_synthesis_report.pdf"
    sentence_processing_method: str = "nltk"
    max_srag_iterations: int = 2  # Limit to 2 iterations as requested

class IPCCPDFKNOWLEDGEBASE:
    """IPCC AR6 PDF knowledge base for fact-checking with section extraction"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf_content = []
        self.load_pdf()
    
    def extract_section_info(self, text: str, page_num: int) -> str:
        """Extract section information from text using pattern matching"""
        try:
            section_patterns = [
                r'([A-Z]\.\d+(?:\.\d+)*(?:\.\d+)*)',
                r'(\d+\.\d+(?:\.\d+)*(?:\.\d+)*)',
                r'(Box\s+[A-Z]?\d+(?:\.\d+)*)',
                r'(Figure\s+[A-Z]?\d+(?:\.\d+)*)',
                r'(Table\s+[A-Z]?\d+(?:\.\d+)*)',
                r'(Section\s+[A-Z]?\d+(?:\.\d+)*)',
            ]
            
            sections_found = []
            search_text = text[:1000] if len(text) > 1000 else text
            
            for pattern in section_patterns:
                matches = re.finditer(pattern, search_text, re.IGNORECASE)
                for match in matches:
                    section = match.group(1).strip()
                    if section not in sections_found:
                        sections_found.append(section)
            
            if sections_found:
                return f"Page {page_num}, Section {sections_found[0]}"
            else:
                return f"Page {page_num}"
        except Exception as e:
            return f"Page {page_num}"
    
    def load_pdf(self):
        """Load and process IPCC AR6 PDF with section extraction"""
        try:
            if os.path.exists(self.pdf_path):
                doc = fitz.open(self.pdf_path)
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = ""
                    try:
                        if hasattr(page, 'get_text'):
                            text = getattr(page, 'get_text')()
                        elif hasattr(page, 'getText'):
                            text = getattr(page, 'getText')()
                        else:
                            text_page = page.get_textpage()
                            text = text_page.extractText() if hasattr(text_page, 'extractText') else ""
                    except Exception as e:
                        st.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                        continue
                    
                    if text and text.strip():
                        section_info = self.extract_section_info(text, page_num + 1)
                        self.pdf_content.append({
                            'page_number': page_num + 1,
                            'content': text.strip(),
                            'section_reference': section_info
                        })
                
                doc.close()
                st.success(f"✅ Loaded {len(self.pdf_content)} pages from IPCC AR6 PDF")
            else:
                st.error(f"❌ IPCC AR6 PDF not found at {self.pdf_path}")
                self.pdf_content = []
                
        except Exception as e:
            st.error(f"Error loading IPCC AR6 PDF: {str(e)}")
            self.pdf_content = []
    
    def get_processed_chunks(self, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
        """Process PDF content into chunks for vector database"""
        if not self.pdf_content:
            return self._get_fallback_statements()
        
        processed_chunks = []
        chunk_id = 0
        
        for page_data in self.pdf_content:
            page_num = page_data['page_number']
            content = page_data['content']
            section_ref = page_data['section_reference']
            
            words = content.split()
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                
                if len(chunk_text.strip()) > 50:
                    processed_chunks.append({
                        'id': f"ipcc_ar6_chunk_{chunk_id}",
                        'content': chunk_text.strip(),
                        'page_number': page_num,
                        'section_reference': section_ref,
                        'source': f"IPCC AR6 Synthesis Report - {section_ref}",
                        'chunk_start_word': i,
                        'chunk_end_word': min(i + chunk_size, len(words))
                    })
                    chunk_id += 1
        
        st.info(f"📊 Processed {len(processed_chunks)} chunks from IPCC AR6 PDF")
        return processed_chunks
    
    def _get_fallback_statements(self) -> List[Dict]:
        """Fallback statements if PDF is not available"""
        return [
            {
                "id": "fallback_1",
                "content": "Human influence has warmed the climate at a rate that is unprecedented in at least the last 2000 years.",
                "page_number": 1,
                "section_reference": "Page 1, Section A.1.1",
                "source": "IPCC AR6 Synthesis Report - Page 1, Section A.1.1",
                "chunk_start_word": 0,
                "chunk_end_word": 20
            },
            {
                "id": "fallback_2", 
                "content": "Global surface temperature has increased faster since 1970 than in any other 50-year period over at least the last 2000 years.",
                "page_number": 1,
                "section_reference": "Page 1, Section A.1.2",
                "source": "IPCC AR6 Synthesis Report - Page 1, Section A.1.2",
                "chunk_start_word": 0,
                "chunk_end_word": 25
            }
        ]

class SentenceBasedTranscriptProcessor:
    """Process YouTube transcripts sentence-wise"""
    
    def __init__(self, method: str = "nltk"):
        self.method = method
        self.nlp = None
        
        if method == "spacy":
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    st.warning("spaCy model 'en_core_web_sm' not found. Falling back to NLTK.")
                    self.method = "nltk"
            except Exception as e:
                st.warning(f"Error initializing spaCy: {str(e)}. Falling back to NLTK.")
                self.method = "nltk"
        elif method == "spacy":
            st.warning("spaCy not available. Falling back to NLTK.")
            self.method = "nltk"
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using the specified method"""
        if self.method == "spacy" and self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        elif self.method == "nltk":
            try:
                sentences = sent_tokenize(text)
                return [sent.strip() for sent in sentences if sent.strip()]
            except Exception as e:
                st.warning(f"NLTK sentence tokenization failed: {str(e)}. Using simple method.")
                return self._simple_sentence_split(text)
        else:
            return self._simple_sentence_split(text)
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting using punctuation"""
        sentences = re.split(r'[.!?]+\s+(?=[A-Z])', text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    def process_timed_transcript(self, timed_transcript: List[Dict]) -> List[Dict]:
        """Process timed transcript segments into sentence-level data"""
        if not timed_transcript:
            return []
        
        sentence_data = []
        sentence_id = 0
        
        full_text = ""
        segment_mapping = []
        
        for segment in timed_transcript:
            start_pos = len(full_text)
            segment_text = segment['text'].strip()
            full_text += segment_text + " "
            end_pos = len(full_text) - 1
            
            segment_mapping.append({
                'start_time': segment['start'],
                'duration': segment['duration'],
                'end_time': segment['start'] + segment['duration'],
                'text': segment_text,
                'start_pos': start_pos,
                'end_pos': end_pos
            })
        
        sentences = self.split_into_sentences(full_text.strip())
        
        current_pos = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_start = full_text.find(sentence, current_pos)
            if sentence_start == -1:
                sentence_start = current_pos
            
            sentence_end = sentence_start + len(sentence)
            current_pos = sentence_end
            
            overlapping_segments = []
            for seg in segment_mapping:
                if (sentence_start <= seg['end_pos'] and sentence_end >= seg['start_pos']):
                    overlapping_segments.append(seg)
            
            if overlapping_segments:
                start_time = min(seg['start_time'] for seg in overlapping_segments)
                end_time = max(seg['end_time'] for seg in overlapping_segments)
                duration = end_time - start_time
            else:
                start_time = 0.0
                duration = 0.0
                end_time = 0.0
            
            sentence_data.append({
                'sentence_id': f"sent_{sentence_id}",
                'text': sentence,
                'start_time': start_time,
                'duration': duration,
                'end_time': end_time,
                'sentence_index': sentence_id,
                'word_count': len(sentence.split()),
                'char_count': len(sentence)
            })
            
            sentence_id += 1
        
        return sentence_data
    
    def extract_climate_sentences(self, sentence_data: List[Dict]) -> List[Dict]:
        """Filter sentences that likely contain climate-related content"""
        climate_keywords = [
            'climate', 'global warming', 'greenhouse gas', 'carbon dioxide', 'co2',
            'temperature', 'warming', 'cooling', 'emission', 'fossil fuel',
            'renewable energy', 'solar', 'wind energy', 'sea level', 'ice',
            'glacier', 'permafrost', 'weather', 'extreme weather', 'drought',
            'flood', 'hurricane', 'cyclone', 'precipitation', 'rainfall',
            'carbon footprint', 'sustainability', 'ipcc', 'paris agreement',
            'methane', 'ozone', 'deforestation', 'ocean acidification',
            'biodiversity', 'ecosystem', 'coral reef', 'arctic', 'antarctic'
        ]
        
        climate_sentences = []
        for sent_data in sentence_data:
            text_lower = sent_data['text'].lower()
            if any(keyword in text_lower for keyword in climate_keywords):
                sent_data['climate_relevant'] = True
                sent_data['matched_keywords'] = [
                    keyword for keyword in climate_keywords 
                    if keyword in text_lower
                ]
                climate_sentences.append(sent_data)
            else:
                sent_data['climate_relevant'] = False
                sent_data['matched_keywords'] = []
        
        return climate_sentences

class YouTubeTranscriptExtractor:
    """Extract and process YouTube video transcripts with sentence-based processing"""
    
    def __init__(self, processor: SentenceBasedTranscriptProcessor):
        self.processor = processor
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats"""
        try:
            patterns = [
                r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
                r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)
            return None
        except Exception as e:
            st.error(f"Error extracting video ID: {str(e)}")
            return None
    
    def get_transcript(self, video_url: str) -> Dict[str, Any]:
        """Get transcript from YouTube video and process it sentence-wise"""
        try:
            video_id = self.extract_video_id(video_url)
            if not video_id:
                return {"error": "Invalid YouTube URL", "transcript": None}
            
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            full_transcript = ""
            transcript_with_time = []
            
            for entry in transcript_list:
                full_transcript += entry['text'] + " "
                transcript_with_time.append({
                    'start': entry['start'],
                    'duration': entry['duration'],
                    'text': entry['text']
                })
            
            sentence_data = self.processor.process_timed_transcript(transcript_with_time)
            climate_sentences = self.processor.extract_climate_sentences(sentence_data)
            
            return {
                "error": None,
                "transcript": full_transcript.strip() if full_transcript else "",
                "timed_transcript": transcript_with_time,
                "video_id": video_id,
                "sentence_data": sentence_data,
                "climate_sentences": climate_sentences,
                "total_sentences": len(sentence_data),
                "climate_sentence_count": len(climate_sentences)
            }
            
        except Exception as e:
            return {"error": str(e), "transcript": None}

class SRAGClimateFactChecker:
    """SRAG Enhanced Climate Fact Checker with Self-Retrieval Augmented Generation"""
    
    def __init__(self, config: SRAGClimateFactCheckerConfig):
        self.config = config
        
        global client
        client = OpenAI(api_key=config.openai_api_key)
        openai.api_key = config.openai_api_key
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
        
        self.chroma_client = chromadb.PersistentClient(path=config.chroma_persist_directory)
        self.ipcc_kb = IPCCPDFKNOWLEDGEBASE(config.ipcc_pdf_path)
        self._setup_knowledge_base()
        self.sentence_processor = SentenceBasedTranscriptProcessor(config.sentence_processing_method)
    
    def _setup_knowledge_base(self):
        """Setup the IPCC AR6 knowledge base"""
        try:
            self.collection = self.chroma_client.get_collection(
                name="ipcc_ar6_knowledge"
            )
            st.info("📚 Using existing IPCC AR6 knowledge base")
        except:
            st.info("🔧 Creating new IPCC AR6 knowledge base...")
            self.collection = self.chroma_client.create_collection(
                name="ipcc_ar6_knowledge"
            )
            self._populate_knowledge_base()
    
    def _populate_knowledge_base(self):
        """Populate the vector database with IPCC AR6 chunks"""
        ipcc_chunks = self.ipcc_kb.get_processed_chunks(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )
        
        if not ipcc_chunks:
            st.error("❌ No IPCC AR6 chunks available for knowledge base")
            return
        
        batch_size = 100
        progress_bar = st.progress(0)
        
        for i in range(0, len(ipcc_chunks), batch_size):
            batch = ipcc_chunks[i:i + batch_size]
            
            documents = [chunk["content"] for chunk in batch]
            metadatas = []
            for chunk in batch:
                metadata = {
                    "source": str(chunk["source"]),
                    "page_number": int(chunk["page_number"]),
                    "section_reference": str(chunk["section_reference"]),
                    "chunk_start_word": int(chunk["chunk_start_word"]),
                    "chunk_end_word": int(chunk["chunk_end_word"])
                }
                metadatas.append(metadata)
            ids = [chunk["id"] for chunk in batch]
            
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                st.warning(f"⚠️ Error adding batch {i//batch_size + 1}: {str(e)}")
            
            progress_bar.progress(min((i + batch_size) / len(ipcc_chunks), 1.0))
        
        st.success(f"✅ Successfully added {len(ipcc_chunks)} IPCC AR6 chunks to knowledge base")
    
    def _grade_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Grade retrieved documents for relevance using SRAG grader"""
        if not documents:
            return []
        
        grader_prompt = f"""You are a grader evaluating the relevance of retrieved documents for climate fact-checking.

Query: {query}

Your task is to assess if each document contains information relevant to answering this query about climate science.

For each document, respond with ONLY "YES" if the document is relevant (contains information that helps answer the query) or "NO" if it's not relevant.

Focus on:
- Climate science accuracy and factual content
- Relevance to the specific query
- Scientific credibility of information

Documents to evaluate:
"""
        
        relevant_docs = []
        for i, doc in enumerate(documents):
            try:
                doc_prompt = grader_prompt + f"\nDocument {i+1}: {doc['content'][:800]}...\n\nRelevance (YES/NO):"
                
                if client is None:
                    st.error("OpenAI client is not initialized.")
                    # Include the document to be safe if client is None
                    relevant_docs.append(doc)
                    continue
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": doc_prompt}],
                    temperature=0.1,
                    max_tokens=10
                )
                
                if response.choices and response.choices[0].message.content:
                    grade = response.choices[0].message.content.strip().upper()
                    
                    if "YES" in grade:
                        relevant_docs.append(doc)
                else:
                    # If no content in response, include document to be safe
                    relevant_docs.append(doc)
                
            except Exception as e:
                # If grading fails, include the document to be safe
                relevant_docs.append(doc)
                continue
        
        return relevant_docs
    
    def _rewrite_query(self, original_query: str, iteration: int) -> str:
        """Rewrite query to improve retrieval effectiveness for climate fact-checking"""
        rewriter_prompt = f"""You are a query rewriter that optimizes queries for climate science fact-checking and retrieval.

Original query: {original_query}
Iteration: {iteration + 1}

Your task is to rewrite this query to improve retrieval of relevant climate science information from IPCC reports and scientific documents.

Guidelines for climate fact-checking queries:
- Make queries more specific to climate science terminology
- Include relevant scientific concepts and keywords
- Focus on factual, verifiable climate information
- Ensure the query targets measurable climate phenomena
- Use terminology consistent with IPCC and climate science literature

Provide ONLY the rewritten query, no explanations:"""
        
        try:
            if client is None:
                st.warning("OpenAI client not available for query rewriting.")
                return original_query
                
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": rewriter_prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            if response.choices and response.choices[0].message.content:
                rewritten_query = response.choices[0].message.content.strip()
                return rewritten_query if rewritten_query else original_query
            else:
                return original_query
            
        except Exception as e:
            st.warning(f"Query rewriting failed: {str(e)}")
            return original_query
    
    def search_ipcc_evidence_srag(self, claim: str) -> List[Dict]:
        """Search for relevant IPCC AR6 evidence using SRAG approach"""
        current_query = claim
        best_docs = []
        
        for iteration in range(self.config.max_srag_iterations):
            try:
                # Retrieve documents using current query
                results = self.collection.query(
                    query_texts=[current_query],
                    n_results=self.config.top_k_retrieval * 2  # Get more initially for filtering
                )
                
                # Convert to document format
                iteration_docs = []
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                        distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 1.0
                        
                        iteration_docs.append({
                            "content": doc,
                            "metadata": metadata,
                            "distance": distance
                        })
                
                # Grade documents for relevance
                relevant_docs = self._grade_documents(current_query, iteration_docs)
                
                # Check if we have good enough results
                if len(relevant_docs) >= self.config.top_k_retrieval:
                    best_docs = relevant_docs[:self.config.top_k_retrieval]
                    st.info(f"🎯 SRAG iteration {iteration + 1}: Found {len(relevant_docs)} relevant documents")
                    break
                elif len(relevant_docs) > len(best_docs):
                    best_docs = relevant_docs
                
                # If not the last iteration, rewrite query
                if iteration < self.config.max_srag_iterations - 1:
                    current_query = self._rewrite_query(claim, iteration)
                    st.info(f"🔄 SRAG iteration {iteration + 1}: Rewriting query for better retrieval")
                
            except Exception as e:
                st.error(f"SRAG iteration {iteration + 1} failed: {str(e)}")
                break
        
        # Fallback to regular search if SRAG didn't find enough documents
        if len(best_docs) < 3:
            try:
                results = self.collection.query(
                    query_texts=[claim],
                    n_results=self.config.top_k_retrieval
                )
                
                fallback_docs = []
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                        distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 1.0
                        
                        fallback_docs.append({
                            "content": doc,
                            "metadata": metadata,
                            "distance": distance
                        })
                
                if len(fallback_docs) > len(best_docs):
                    best_docs = fallback_docs
                    st.info("🔄 Using fallback retrieval due to insufficient SRAG results")
            
            except Exception as e:
                st.error(f"Fallback retrieval failed: {str(e)}")
        
        return best_docs
    
    def identify_top_climate_claims_from_sentences(self, climate_sentences: List[Dict]) -> List[Tuple[str, str, Dict]]:
        """Identify and rank top climate claims from sentence-level data"""
        if not climate_sentences:
            return []
        
        top_sentences = sorted(climate_sentences, key=lambda x: len(x['matched_keywords']), reverse=True)[:10]
        combined_text = " ".join([sent['text'] for sent in top_sentences])
        
        # Enhanced prompt for climate fact-checking
        prompt = f"""
        Analyze the following climate-related sentences from a video transcript and identify the TOP CLIMATE CLAIMS that can be fact-checked.
        
        Focus on:
        - Specific factual assertions about climate change, global warming, or climate science
        - Quantitative claims about greenhouse gases, temperatures, or climate trends  
        - Statements about climate impacts, solutions, or scientific findings
        - Claims about climate policies, international agreements, or scientific consensus
        - Assertions that can be verified against IPCC reports and climate science literature
        
        AVOID vague or subjective statements. Focus on SPECIFIC, VERIFIABLE claims.
        
        Return the result as a JSON array where each item has:
        - "claim_id": "Claim1", "Claim2", etc.
        - "claim_text": The exact, specific factual claim
        
        Climate sentences: {combined_text[:2000]}
        
        Top climate claims for fact-checking:
        """
        
        try:
            if client is None:
                st.error("OpenAI client is not initialized.")
                return []
                
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            if not response.choices or not response.choices[0].message.content:
                return []
                
            claims_text = response.choices[0].message.content.strip()
            
            try:
                claims_json = json.loads(claims_text)
                if isinstance(claims_json, list):
                    results = []
                    for i, claim in enumerate(claims_json):
                        if claim.get("claim_text"):
                            claim_text = claim.get("claim_text", "")
                            source_sentences = self._find_source_sentences(claim_text, climate_sentences)
                            
                            results.append((
                                claim.get("claim_id", f"Claim{i+1}"), 
                                claim_text,
                                {
                                    'source_sentences': source_sentences,
                                    'sentence_count': len(source_sentences)
                                }
                            ))
                    return results[:5]
                else:
                    return []
            except json.JSONDecodeError:
                lines = claims_text.split('\n')
                claims = []
                claim_count = 1
                for line in lines:
                    line = line.strip()
                    if line and any(keyword in line.lower() for keyword in 
                                   ['climate', 'temperature', 'warming', 'carbon', 'greenhouse', 'emission']):
                        line = re.sub(r'^[-*•]\s*', '', line)
                        line = re.sub(r'^["\']|["\']$', '', line)
                        line = re.sub(r'^\d+\.\s*', '', line)
                        if len(line) > 20:
                            source_sentences = self._find_source_sentences(line, climate_sentences)
                            claims.append((
                                f"Claim{claim_count}", 
                                line,
                                {
                                    'source_sentences': source_sentences,
                                    'sentence_count': len(source_sentences)
                                }
                            ))
                            claim_count += 1
                return claims[:5]
                
        except Exception as e:
            st.error(f"Error identifying claims: {str(e)}")
            return []
    
    def _find_source_sentences(self, claim_text: str, climate_sentences: List[Dict]) -> List[Dict]:
        """Find the source sentences that likely contain this claim"""
        claim_words = set(claim_text.lower().split())
        source_sentences = []
        
        for sent_data in climate_sentences:
            sent_words = set(sent_data['text'].lower().split())
            overlap = len(claim_words.intersection(sent_words))
            if overlap >= 3:
                sent_data_copy = sent_data.copy()
                sent_data_copy['word_overlap'] = overlap
                source_sentences.append(sent_data_copy)
        
        source_sentences.sort(key=lambda x: x['word_overlap'], reverse=True)
        return source_sentences[:3]
    
    def fact_check_claim_with_user_response(self, claim_id: str, claim_text: str, user_response: str, 
                                          ipcc_evidence: List[Dict], source_sentences: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Fact-check claim using SRAG-enhanced evidence and user response"""
        
        if not ipcc_evidence:
            return self._fact_check_without_ipcc_with_user_response(claim_id, claim_text, user_response, source_sentences)
        
        ipcc_evidence = sorted(ipcc_evidence, key=lambda x: x['distance'])[:3]
        
        evidence_text = "\n\n".join([
            f"IPCC AR6 Evidence {i+1}:\n"
            f"Source: {doc['metadata'].get('source', 'IPCC AR6 Report')}\n"
            f"Section Reference: {doc['metadata'].get('section_reference', 'N/A')}\n"
            f"Content: {doc['content'][:500]}..."
            for i, doc in enumerate(ipcc_evidence)
        ])
        
        sentence_context = ""
        if source_sentences:
            sentence_context = "\n\nORIGINAL VIDEO CONTEXT:\n"
            for i, sent in enumerate(source_sentences[:3]):
                sentence_context += f"Sentence {i+1} (at {sent.get('start_time', 0):.1f}s): {sent['text']}\n"
        
        # Enhanced climate fact-checking prompt with SRAG context
        prompt = f"""
        You are a climate science educational assistant providing formative feedback on climate fact-checking responses.
        
        A student has analyzed this climate claim and provided their response:
        STUDENT RESPONSE: "{user_response}"
        
        ORIGINAL CLIMATE CLAIM: {claim_text}
        {sentence_context}
        
        IPCC AR6 SCIENTIFIC EVIDENCE (Retrieved via Self-Retrieval Augmented Generation):
        {evidence_text}
        
        Your task is to provide educational feedback following these principles:
        
        **STEP 1 - UNDERSTANDING**: Restate what the student is trying to communicate, showing you understand their perspective.
        → If the student expresses uncertainty (e.g., "I don't know", "not sure", or leaves it blank), acknowledge their honesty without pretending they engaged deeply. Briefly explain the concept they were expected to address, and encourage them to re-engage with a low-pressure question or suggestion.

        **STEP 2 - SCIENTIFIC ANALYSIS**: 
        - Evaluate the scientific accuracy of their response against IPCC evidence
        - Identify any misconceptions or gaps in climate science understanding
        - Note what they got right about climate science
        
        **STEP 3 - FORMATIVE FEEDBACK**:
        - **Feed Up**: State the climate science learning goal
        - **Feed Back**: Highlight strengths and gently address scientific inaccuracies
        - **Feed Forward**: Provide a thought-provoking question or hint to deepen their climate science understanding
        
        **GUIDELINES**:
        - Be constructive and encouraging about their scientific reasoning process
        - Use accurate IPCC-based climate science (reference the evidence provided)
        - Don't simply give the correct answer - guide them toward better understanding
        - Focus on building climate science literacy and critical thinking
        - Acknowledge uncertainty where appropriate in climate science
        
        Provide your educational feedback:
        """
        
        try:
            if client is None:
                st.error("OpenAI client is not initialized.")
                return self._fact_check_without_ipcc_with_user_response(claim_id, claim_text, user_response, source_sentences)
                
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            
            if not response.choices or not response.choices[0].message.content:
                return self._fact_check_without_ipcc_with_user_response(claim_id, claim_text, user_response, source_sentences)
                
            result = response.choices[0].message.content.strip()
            
            return {
                "success": True,
                "claim_id": claim_id,
                "fact_check_result": result.strip(),
                "ipcc_sections": [doc['metadata'].get('section_reference', 'N/A') for doc in ipcc_evidence],
                "ipcc_sources": [doc['metadata'].get('source', 'IPCC AR6 Report') for doc in ipcc_evidence],
                "relevance_scores": [round(1.0 - doc['distance'], 3) for doc in ipcc_evidence],
                "evidence_type": "SRAG_ENHANCED_IPCC_AR6",
                "source_sentences": source_sentences[:3] if source_sentences else [],
                "srag_enhanced": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "claim_id": claim_id,
                "fact_check_result": f"Error in SRAG fact-checking: {str(e)}",
                "ipcc_sections": [],
                "ipcc_sources": [],
                "relevance_scores": [],
                "evidence_type": "ERROR",
                "source_sentences": [],
                "srag_enhanced": False
            }
    
    def _fact_check_without_ipcc_with_user_response(self, claim_id: str, claim_text: str, user_response: str, 
                                                   source_sentences: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Fact-check when no relevant IPCC evidence is found"""
        
        sentence_context = ""
        if source_sentences:
            sentence_context = "\n\nORIGINAL VIDEO CONTEXT:\n"
            for i, sent in enumerate(source_sentences[:3]):
                sentence_context += f"Sentence {i+1} (at {sent.get('start_time', 0):.1f}s): {sent['text']}\n"
        
        prompt = f"""
        You are a climate science educational assistant providing formative feedback on climate fact-checking responses.
        
        A student has analyzed this climate claim and provided their response:
        STUDENT RESPONSE: "{user_response}"
        
        ORIGINAL CLIMATE CLAIM: {claim_text}
        {sentence_context}
        
        Note: This claim was not found in the available IPCC AR6 report sections, so you should rely on general climate science knowledge.
        
        Your task is to provide educational feedback following these principles:
        
        **STEP 1 - UNDERSTANDING**: Restate what the student is trying to communicate.
        → If the student expresses uncertainty (e.g., "I don't know", "not sure", or leaves it blank), acknowledge their honesty without pretending they engaged deeply. Briefly explain the concept they were expected to address, and encourage them to re-engage with a low-pressure question or suggestion.

        **STEP 2 - SCIENTIFIC ANALYSIS**: 
        - Evaluate their reasoning against established climate science principles
        - Identify any misconceptions about climate systems or processes
        - Note strengths in their scientific thinking
        
        **STEP 3 - FORMATIVE FEEDBACK**:
        - **Feed Up**: State the climate science learning goal
        - **Feed Back**: Highlight good scientific reasoning and address misconceptions
        - **Feed Forward**: Guide them toward authoritative climate science sources (IPCC, NASA, NOAA)
        
        **GUIDELINES**:
        - Be supportive of their learning process
        - Use established climate science principles
        - Don't provide definitive answers - encourage them to consult authoritative sources
        - Build their climate science evaluation skills
        - Acknowledge when claims need verification from expert sources
        
        Provide your educational feedback:
        """
        
        try:
            if client is None:
                return {
                    "success": False,
                    "claim_id": claim_id,
                    "fact_check_result": "OpenAI client is not initialized.",
                    "ipcc_sections": [],
                    "ipcc_sources": [],
                    "relevance_scores": [],
                    "evidence_type": "ERROR",
                    "source_sentences": [],
                    "srag_enhanced": False
                }
                
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600
            )
            
            result = ""
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
            
            return {
                "success": True,
                "claim_id": claim_id,
                "fact_check_result": result if result else "Unable to provide educational feedback on this response.",
                "ipcc_sections": [],
                "ipcc_sources": [],
                "relevance_scores": [],
                "evidence_type": "GENERAL_CLIMATE_KNOWLEDGE",
                "source_sentences": source_sentences[:3] if source_sentences else [],
                "srag_enhanced": False
            }
            
        except Exception as e:
            return {
                "success": False,
                "claim_id": claim_id,
                "fact_check_result": f"Error in educational feedback generation: {str(e)}",
                "ipcc_sections": [],
                "ipcc_sources": [],
                "relevance_scores": [],
                "evidence_type": "ERROR",
                "source_sentences": [],
                "srag_enhanced": False
            }

def main():
    st.set_page_config(
        page_title="SRAG Climate Fact-Checker",
        page_icon="🌍",
        layout="wide"
    )
    
    st.title("🌍 SRAG Climate Fact-Checker")
    st.markdown("**Self-Retrieval Augmented Generation for Climate Fact-Checking using IPCC AR6 Reports**")
    
    # File upload section
    st.sidebar.header("📄 IPCC AR6 PDF")
    uploaded_file = st.sidebar.file_uploader(
        "Upload IPCC AR6 Synthesis Report PDF",
        type=['pdf'],
        help="Upload the IPCC AR6 Synthesis Report PDF file"
    )
    
    # Handle file upload
    pdf_path = "ipcc_ar6_synthesis_report.pdf"
    if uploaded_file is not None:
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("✅ IPCC AR6 PDF uploaded successfully!")
    elif not os.path.exists(pdf_path):
        st.sidebar.warning("⚠️ Please upload the IPCC AR6 PDF file to continue")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ SRAG Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        
        st.header("🔄 SRAG Settings")
        max_iterations = st.selectbox(
            "Max SRAG Iterations:",
            [1, 2, 3],
            index=1,
            help="Number of self-retrieval iterations (2 recommended)"
        )
        
        st.header("📝 Sentence Processing Settings")
        sentence_method = st.selectbox(
            "Sentence splitting method:",
            ["nltk", "spacy", "simple"],
            index=0,
            help="Choose method for splitting transcripts into sentences"
        )
        
        st.header("📺 YouTube Video Input")
        sidebar_video_url = st.text_input(
            "YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter YouTube video URL for analysis",
            key="sidebar_video_url"
        )
        
        st.header("📊 Analysis Settings")
        max_chunks = st.slider("Max chunks to analyze", 1, 10, 5)
        chunk_size = st.slider("Chunk size (words)", 200, 800, 500)
        top_k_retrieval = st.slider("Evidence retrieval", 3, 10, 5, help="Number of IPCC chunks to retrieve per SRAG iteration")
        
        # Initialize system only when all requirements are met
        config = None
        if api_key and os.path.exists(pdf_path):
            config = SRAGClimateFactCheckerConfig(
                openai_api_key=api_key,
                max_chunks_to_analyze=max_chunks,
                chunk_size=chunk_size,
                top_k_retrieval=top_k_retrieval,
                ipcc_pdf_path=pdf_path,
                sentence_processing_method=sentence_method,
                max_srag_iterations=max_iterations
            )
            
            if 'srag_fact_checker' not in st.session_state or st.session_state.get('last_config') != (sentence_method, max_iterations):
                with st.spinner("🔧 Initializing SRAG Climate Fact-Checker..."):
                    try:
                        st.session_state.srag_fact_checker = SRAGClimateFactChecker(config)
                        st.session_state.transcript_extractor = YouTubeTranscriptExtractor(
                            st.session_state.srag_fact_checker.sentence_processor
                        )
                        st.session_state.last_config = (sentence_method, max_iterations)
                        st.success("✅ SRAG Climate Fact-Checker ready with enhanced self-retrieval!")
                    except Exception as e:
                        st.error(f"❌ Error initializing SRAG fact-checker: {str(e)}")
                        st.stop()
    
    # Check if system is ready
    if 'srag_fact_checker' not in st.session_state:
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
        if not os.path.exists(pdf_path):
            st.warning("⚠️ Please upload the IPCC AR6 PDF file in the sidebar.")
        st.stop()
    
    # Initialize session state for interactive claims
    if 'identified_claims' not in st.session_state:
        st.session_state.identified_claims = []
    if 'current_claim_index' not in st.session_state:
        st.session_state.current_claim_index = 0
    if 'user_responses' not in st.session_state:
        st.session_state.user_responses = {}
    if 'fact_check_results' not in st.session_state:
        st.session_state.fact_check_results = {}
    
    # Main video analysis interface
    st.header("📺 YouTube Video Analysis with SRAG Enhancement")
    
    # Use sidebar URL if provided, otherwise show main input
    if sidebar_video_url:
        st.info(f"🔗 Using video URL from sidebar: {sidebar_video_url}")
        video_url = sidebar_video_url
    else:
        video_url = st.text_input(
            "YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=... (or use sidebar)",
            help="Paste YouTube video URL here or use the sidebar input"
        )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        identify_claims_button = st.button("🔍 Analyze Video with SRAG", type="primary")
    
    with col2:
        if st.button("🧹 Clear All Results"):
            for key in ['transcript_data', 'identified_claims', 'current_claim_index', 'user_responses', 'fact_check_results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_claim_index = 0
            st.session_state.user_responses = {}
            st.session_state.fact_check_results = {}
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Current Claim"):
            if st.session_state.identified_claims and st.session_state.current_claim_index < len(st.session_state.identified_claims):
                current_claim_id = st.session_state.identified_claims[st.session_state.current_claim_index][0]
                if current_claim_id in st.session_state.user_responses:
                    del st.session_state.user_responses[current_claim_id]
                if current_claim_id in st.session_state.fact_check_results:
                    del st.session_state.fact_check_results[current_claim_id]
            st.rerun()
    
    # Step 1: Process Video and Identify Claims with SRAG
    if identify_claims_button and video_url:
        # Extract transcript and process sentences
        with st.spinner("📝 Extracting video transcript and processing sentences..."):
            transcript_data = st.session_state.transcript_extractor.get_transcript(video_url)
        
        if transcript_data['error']:
            st.error(f"❌ Error: {transcript_data['error']}")
            st.stop()
        
        st.success("✅ Transcript extracted and processed into sentences!")
        st.session_state.transcript_data = transcript_data
        
        # Display sentence processing statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sentences", transcript_data['total_sentences'])
        with col2:
            st.metric("Climate-Related Sentences", transcript_data['climate_sentence_count'])
        with col3:
            climate_percentage = (transcript_data['climate_sentence_count'] / transcript_data['total_sentences'] * 100) if transcript_data['total_sentences'] > 0 else 0
            st.metric("Climate Content %", f"{climate_percentage:.1f}%")
        
        # Identify claims from sentences
        with st.spinner("🔍 Identifying climate claims using SRAG-enhanced analysis..."):
            claims = st.session_state.srag_fact_checker.identify_top_climate_claims_from_sentences(
                transcript_data['climate_sentences']
            )
        
        if claims:
            st.session_state.identified_claims = claims
            st.session_state.current_claim_index = 0
            st.session_state.user_responses = {}
            st.session_state.fact_check_results = {}
            st.success(f"✅ Identified {len(claims)} climate claims for SRAG-enhanced analysis!")
        else:
            st.info("🔍 No specific climate claims were identified in the video content.")
    
    # Display sentence analysis if available
    if 'transcript_data' in st.session_state and st.session_state.transcript_data.get('sentence_data'):
        with st.expander("📊 Sentence Analysis Results"):
            sentence_data = st.session_state.transcript_data['sentence_data']
            climate_sentences = st.session_state.transcript_data['climate_sentences']
            
            # Show sample climate sentences
            st.subheader("🌍 Top Climate-Related Sentences")
            for i, sent in enumerate(climate_sentences[:5]):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Sentence {i+1}:** {sent['text']}")
                        st.caption(f"Keywords: {', '.join(sent['matched_keywords'])}")
                    with col2:
                        st.caption(f"Time: {sent['start_time']:.1f}s")
                        st.caption(f"Words: {sent['word_count']}")
    
    # Step 2: Interactive Claims Analysis with SRAG Enhancement
    if st.session_state.identified_claims:
        st.header("🎯 Interactive SRAG-Enhanced Climate Claims Analysis")
        
        total_claims = len(st.session_state.identified_claims)
        current_index = st.session_state.current_claim_index
        
        if current_index < total_claims:
            claim_id, claim_text, claim_metadata = st.session_state.identified_claims[current_index]
            
            # Progress indicator
            progress_value = (current_index + 1) / total_claims
            st.progress(progress_value)
            st.write(f"**Claim {current_index + 1} of {total_claims}** (SRAG Enhanced)")
            
            # Display current claim with sentence context
            st.markdown("### 📋 Current Climate Claim:")
            st.info(f"**{claim_id}:** {claim_text}")
            
            # Show source sentences if available
            if claim_metadata.get('source_sentences'):
                with st.expander("📝 Source Context from Video"):
                    for i, sent in enumerate(claim_metadata['source_sentences'][:3]):
                        st.write(f"**Sentence {i+1}** (at {sent.get('start_time', 0):.1f}s): {sent['text']}")
            
            # Check if user has already responded to this claim
            if claim_id not in st.session_state.user_responses:
                st.markdown("### 💭 Your Response:")
                st.write("Please share your thoughts on this climate claim. Do you think it's accurate? Why or why not?")
                
                user_response = st.text_area(
                    "Your analysis of this claim:",
                    placeholder="Share your thoughts on whether this climate claim is accurate, partially accurate, or inaccurate, and explain your scientific reasoning...",
                    height=100,
                    key=f"response_{claim_id}_{current_index}"
                )
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("✅ Submit Response", disabled=not user_response.strip(), key=f"submit_{current_index}"):
                        if user_response.strip():
                            st.session_state.user_responses[claim_id] = user_response.strip()
                            
                            # Perform SRAG-enhanced fact-checking with user response
                            with st.spinner("🔄 Analyzing your response with SRAG-enhanced IPCC AR6 evidence..."):
                                try:
                                    ipcc_evidence = st.session_state.srag_fact_checker.search_ipcc_evidence_srag(claim_text)
                                    fact_check_result = st.session_state.srag_fact_checker.fact_check_claim_with_user_response(
                                        claim_id, claim_text, user_response.strip(), ipcc_evidence, 
                                        claim_metadata.get('source_sentences', [])
                                    )
                                    st.session_state.fact_check_results[claim_id] = fact_check_result
                                except Exception as e:
                                    st.error(f"Error during SRAG fact-checking: {str(e)}")
                            
                            st.rerun()
                
                with col2:
                    if st.button("⏭️ Skip This Claim", key=f"skip_{current_index}"):
                        st.session_state.current_claim_index = min(current_index + 1, total_claims)
                        st.rerun()
            
            else:
                # Display user response and SRAG fact-check result
                st.markdown("### 💭 Your Response:")
                st.write(st.session_state.user_responses[claim_id])
                
                if claim_id in st.session_state.fact_check_results:
                    fact_check = st.session_state.fact_check_results[claim_id]
                    
                    st.markdown("### 🎓 SRAG-Enhanced Educational Feedback:")
                    
                    # Status indicator with SRAG enhancement
                    if fact_check['success']:
                        if fact_check.get('srag_enhanced'):
                            st.success("✅ Feedback based on SRAG-Enhanced IPCC AR6 Evidence")
                        elif fact_check['evidence_type'] == 'SRAG_ENHANCED_IPCC_AR6':
                            st.success("✅ Feedback based on SRAG-Enhanced IPCC AR6 Evidence")
                        elif fact_check['evidence_type'] == 'GENERAL_CLIMATE_KNOWLEDGE':
                            st.info("ℹ️ General climate science educational feedback")
                        else:
                            st.info("ℹ️ Educational feedback provided")
                    else:
                        st.error("❌ Error in generating feedback")
                    
                    # Display educational feedback result
                    st.markdown(fact_check['fact_check_result'])
                    
                    # Show sentence context if available
                    if fact_check.get('source_sentences'):
                        st.markdown("**🎬 Video Context:**")
                        for sent in fact_check['source_sentences']:
                            st.markdown(f"• At {sent.get('start_time', 0):.1f}s: {sent['text']}")
                    
                    # Show IPCC AR6 references with sections if available
                    if fact_check.get('ipcc_sections') and fact_check.get('srag_enhanced'):
                        st.markdown("**📚 SRAG-Enhanced IPCC AR6 References:**")
                        for section, source in zip(fact_check['ipcc_sections'], fact_check['ipcc_sources']):
                            st.markdown(f"• {source} - {section}")
                    
                    # Show relevance scores
                    if fact_check.get('relevance_scores') and fact_check.get('srag_enhanced'):
                        avg_relevance = np.mean(fact_check['relevance_scores'])
                        st.markdown(f"**📊 SRAG Evidence Relevance Score:** {avg_relevance:.3f}")
                
                # Navigation buttons
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if current_index > 0:
                        if st.button("⬅️ Previous Claim", key=f"prev_{current_index}"):
                            st.session_state.current_claim_index = current_index - 1
                            st.rerun()
                
                with col2:
                    if st.button("🔄 Revise Response", key=f"revise_{current_index}"):
                        if claim_id in st.session_state.user_responses:
                            del st.session_state.user_responses[claim_id]
                        if claim_id in st.session_state.fact_check_results:
                            del st.session_state.fact_check_results[claim_id]
                        st.rerun()
                
                with col3:
                    if current_index < total_claims - 1:
                        if st.button("➡️ Next Claim", key=f"next_{current_index}"):
                            st.session_state.current_claim_index = current_index + 1
                            st.rerun()
                    else:
                        if st.button("🏁 Complete Analysis", key=f"complete_{current_index}"):
                            st.session_state.current_claim_index = total_claims
                            st.rerun()
        
        else:
            # Analysis complete - show summary
            st.header("🎉 SRAG-Enhanced Analysis Complete!")
            st.success("You have completed the interactive SRAG-enhanced analysis of all identified climate claims.")
            
            # Summary of all claims and responses
            st.header("📋 Complete SRAG Analysis Summary")
            
            srag_enhanced_count = 0
            for i, (claim_id, claim_text, claim_metadata) in enumerate(st.session_state.identified_claims):
                with st.expander(f"📄 {claim_id}: {claim_text[:100]}{'...' if len(claim_text) > 100 else ''}"):
                    st.markdown(f"**Full Claim:** {claim_text}")
                    
                    # Show source sentences
                    if claim_metadata.get('source_sentences'):
                        st.markdown("**Video Context:**")
                        for sent in claim_metadata['source_sentences'][:3]:
                            st.markdown(f"• At {sent.get('start_time', 0):.1f}s: {sent['text']}")
                    
                    if claim_id in st.session_state.user_responses:
                        st.markdown(f"**Your Response:** {st.session_state.user_responses[claim_id]}")
                        
                        if claim_id in st.session_state.fact_check_results:
                            fact_check = st.session_state.fact_check_results[claim_id]
                            
                            if fact_check.get('srag_enhanced'):
                                st.markdown("**🔄 SRAG-Enhanced Educational Feedback:**")
                                srag_enhanced_count += 1
                            else:
                                st.markdown("**Educational Feedback:**")
                                
                            st.markdown(fact_check['fact_check_result'])
                            
                            # Show references
                            if fact_check.get('ipcc_sections') and fact_check.get('srag_enhanced'):
                                st.markdown("**SRAG-Enhanced References:**")
                                for section in fact_check['ipcc_sections']:
                                    st.markdown(f"• IPCC AR6 - {section}")
                    else:
                        st.markdown("*No response provided*")
            
            # SRAG Statistics
            st.info(f"🔄 **SRAG Enhancement:** {srag_enhanced_count} out of {len(st.session_state.identified_claims)} claims were analyzed using Self-Retrieval Augmented Generation")
            
            # Option to restart
            if st.button("🔄 Analyze Another Video", key="restart_analysis"):
                for key in ['transcript_data', 'identified_claims', 'current_claim_index', 'user_responses', 'fact_check_results']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.current_claim_index = 0
                st.session_state.user_responses = {}
                st.session_state.fact_check_results = {}
                st.rerun()
    
    # Display transcript if available
    if 'transcript_data' in st.session_state:
        with st.expander("📄 View Full Transcript"):
            st.text_area(
                "Transcript:",
                st.session_state.transcript_data['transcript'],
                height=200,
                disabled=True,
                key="transcript_display"
            )
    
    # Statistics and Analytics with SRAG metrics
    if st.session_state.identified_claims and st.session_state.user_responses:
        st.header("📊 SRAG Analysis Statistics")
        
        total_claims = len(st.session_state.identified_claims)
        responded_claims = len(st.session_state.user_responses)
        fact_checked_claims = len(st.session_state.fact_check_results)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Claims Identified", total_claims)
        with col2:
            st.metric("Claims Responded To", responded_claims)
        with col3:
            st.metric("Educational Feedback Generated", fact_checked_claims)
        
        # SRAG enhancement breakdown
        if st.session_state.fact_check_results:
            srag_enhanced = sum(1 for result in st.session_state.fact_check_results.values() 
                              if result.get('srag_enhanced') or result.get('evidence_type') == 'SRAG_ENHANCED_IPCC_AR6')
            general_checks = sum(1 for result in st.session_state.fact_check_results.values() 
                               if result.get('evidence_type') == 'GENERAL_CLIMATE_KNOWLEDGE')
            
            st.subheader("🔄 SRAG Enhancement Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("SRAG-Enhanced IPCC AR6 Feedback", srag_enhanced)
            with col2:
                st.metric("General Climate Knowledge Feedback", general_checks)
        
        # Sentence processing statistics
        if 'transcript_data' in st.session_state:
            st.subheader("📝 Sentence Processing Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sentences Processed", st.session_state.transcript_data['total_sentences'])
            with col2:
                st.metric("Climate-Related Sentences", st.session_state.transcript_data['climate_sentence_count'])
            with col3:
                processing_method = st.session_state.get('last_config', ('nltk', 2))[0]
                st.metric("Processing Method", processing_method.upper())
    
    # Help section with SRAG information
    with st.expander("ℹ️ How to Use This SRAG Climate Fact-Checking Tool"):
        st.markdown("""
        ### 🔄 SRAG-Enhanced Climate Fact-Checking Learning Process:
        
        **What is SRAG (Self-Retrieval Augmented Generation)?**
        SRAG enhances traditional RAG by using an AI system to iteratively improve its own information retrieval through:
        - **Self-Assessment**: The AI evaluates the relevance of retrieved documents
        - **Query Rewriting**: If documents aren't relevant enough, the AI rewrites the query for better results
        - **Iterative Improvement**: This process repeats for up to 2 iterations to find the best evidence
        
        **Step 1: Upload & Configure**
        1. Upload the IPCC AR6 Synthesis Report PDF
        2. Enter your OpenAI API key
        3. Choose SRAG iterations (2 recommended for optimal balance of quality and speed)
        4. Select sentence processing method (NLTK recommended)
        5. Paste a YouTube video URL
        
        **Step 2: SRAG-Enhanced Analysis**
        1. Click "Analyze Video with SRAG" to process the video
        2. The system extracts climate-related sentences with timestamps
        3. AI identifies specific, verifiable climate claims
        4. Each claim is enhanced through SRAG retrieval from IPCC documents
        
        **Step 3: Interactive Learning with SRAG Evidence**
        1. For each claim, see the original video context with timestamps
        2. Share your analysis - is the claim accurate? Why?
        3. Receive educational feedback based on SRAG-enhanced IPCC evidence
        4. Compare your reasoning with authoritative climate science
        5. Navigate through claims at your own pace
        
        **Step 4: Learn & Reflect**
        1. Review complete analysis summary with SRAG enhancement statistics
        2. Learn from formative feedback grounded in IPCC science
        3. Understand how claims connect to specific video moments
        4. See precise IPCC references retrieved through SRAG
        
        ### 🌟 Key SRAG Improvements:
        - **Higher Quality Evidence**: SRAG finds more relevant IPCC content through iterative retrieval
        - **Better Fact-Checking**: Enhanced evidence leads to more accurate educational feedback
        - **Adaptive Retrieval**: System learns and improves its search strategy for each claim
        - **Climate-Focused**: Query rewriting specifically targets climate science terminology
        - **Transparency**: See when SRAG enhancement was used vs. general knowledge
        
        ### 📊 SRAG Process:
        1. **Initial Retrieval**: Search IPCC database with original claim
        2. **Relevance Grading**: AI evaluates if retrieved documents are relevant
        3. **Query Rewriting**: If needed, rewrite query with better climate science terms
        4. **Enhanced Retrieval**: Search again with improved query
        5. **Best Evidence Selection**: Choose most relevant documents from all iterations
        
        ### 🎯 Learning Benefits:
        - **Higher Accuracy**: SRAG finds better evidence than simple retrieval
        - **Scientific Grounding**: Feedback based on most relevant IPCC content
        - **Iterative Learning**: Understand how search strategies can be improved
        - **Climate Expertise**: Build knowledge using authoritative climate science
        - **Critical Thinking**: Compare different levels of evidence quality
        
        ### ⚡ Performance Notes:
        - SRAG may take slightly longer due to iterative processing
        - Maximum 2 iterations balances quality with speed
        - System falls back to regular search if SRAG doesn't find enough evidence
        - Enhanced evidence marked with special indicators
        
        ### 🔧 Troubleshooting:
        - If SRAG seems slow, reduce max iterations to 1
        - Check "SRAG Enhancement Breakdown" in statistics for success rate  
        - SRAG works best with specific, factual climate claims
        - Some claims may use general knowledge if IPCC evidence insufficient
        """)

if __name__ == "__main__":
    main()
        