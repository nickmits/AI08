import os
from typing import List
import PyPDF2
import io
from typing import List, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi
import re

class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])

class EnhancedTextFileLoader:
    """Enhanced loader that supports TXT, PDF files and YouTube transcripts"""
    
    def __init__(self, path_or_url: str, encoding: str = "utf-8"):
        self.documents = []
        self.metadata = []
        self.path_or_url = path_or_url
        self.encoding = encoding
        
    def load_documents(self) -> List[str]:
        """Load documents from various sources"""
        if self._is_youtube_url(self.path_or_url):
            return self._load_youtube_transcript()
        elif self.path_or_url.lower().endswith('.pdf'):
            return self._load_pdf()
        elif self.path_or_url.lower().endswith('.txt'):
            return self._load_txt()
        else:
            raise ValueError(f"Unsupported file type: {self.path_or_url}")
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL"""
        youtube_patterns = [
            r'youtube\.com/watch\?v=',
            r'youtu\.be/',
            r'youtube\.com/embed/',
            r'youtube\.com/v/'
        ]
        return any(re.search(pattern, url) for pattern in youtube_patterns)
    
    def _extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from URL"""
        patterns = [
            r'youtube\.com/watch\?v=([^&]+)',
            r'youtu\.be/([^?]+)',
            r'youtube\.com/embed/([^?]+)',
            r'youtube\.com/v/([^?]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract video ID from URL: {url}")
    
    def _load_youtube_transcript(self) -> List[str]:
        """Load transcript from YouTube video"""
        try:
            video_id = self._extract_video_id(self.path_or_url)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine transcript into text
            full_text = " ".join([entry['text'] for entry in transcript])
            self.documents.append(full_text)
            self.metadata.append({
                'source': 'youtube',
                'video_id': video_id,
                'url': self.path_or_url,
                'duration': len(transcript)
            })
            
            return self.documents
            
        except Exception as e:
            raise ValueError(f"Failed to load YouTube transcript: {str(e)}")
    
    def _load_pdf(self) -> List[str]:
        """Load PDF file using PyPDF2 with improved text extraction"""
        try:
            with open(self.path_or_url, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        
                        # Clean and validate extracted text
                        if page_text and page_text.strip():
                            # Remove null bytes and other problematic characters
                            cleaned_text = page_text.replace('\x00', '').replace('\ufffd', '')
                            
                            # Check if text is mostly readable (not binary/encoded)
                            if self._is_readable_text(cleaned_text):
                                text += f"\n--- Page {page_num + 1} ---\n"
                                text += cleaned_text + "\n"
                            else:
                                print(f"Warning: Page {page_num + 1} contains unreadable text, skipping")
                                
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                        continue
                
                if not text.strip():
                    raise ValueError("No readable text could be extracted from the PDF")
                
                self.documents.append(text)
                self.metadata.append({
                    'source': 'pdf',
                    'file_path': self.path_or_url,
                    'num_pages': len(pdf_reader.pages),
                    'file_size': os.path.getsize(self.path_or_url)
                })
                
                return self.documents
                
        except FileNotFoundError:
            raise ValueError(f"PDF file not found: {self.path_or_url}")
        except Exception as e:
            raise ValueError(f"Failed to load PDF: {str(e)}")

    def _is_readable_text(self, text: str, min_readable_ratio: float = 0.7) -> bool:
        """Check if text contains mostly readable characters"""
        if not text or len(text) < 10:
            return False
        
        # Count printable characters
        printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
        readable_ratio = printable_chars / len(text)
        
        # Additional check: ensure it's not mostly encoded/base64-like
        base64_chars = sum(1 for c in text if c.isalnum() or c in '+/=')
        if base64_chars / len(text) > 0.9 and ' ' not in text[:100]:
            return False
        
        return readable_ratio >= min_readable_ratio

    def _load_txt(self) -> List[str]:
        """Load text file"""
        try:
            with open(self.path_or_url, 'r', encoding=self.encoding) as file:
                content = file.read()
                self.documents.append(content)
                self.metadata.append({
                    'source': 'txt',
                    'file_path': self.path_or_url,
                    'encoding': self.encoding,
                    'file_size': os.path.getsize(self.path_or_url)
                })
                
                return self.documents
                
        except FileNotFoundError:
            raise ValueError(f"Text file not found: {self.path_or_url}")
        except Exception as e:
            raise ValueError(f"Failed to load text file: {str(e)}")