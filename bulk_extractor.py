import os
import asyncio
import json
import logging
import pandas as pd
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from pathlib import Path
from typing import List, Set, Literal
from ollama import AsyncClient
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
ROOT_FOLDER = "/home/pyatrix/Downloads/rfp-sam-gov/output"              # Root directory
OUTPUT_CSV = "rfp_extraction_results.csv"
PROGRESS_FILE = "extraction_progress.json"
SAVE_INTERVAL = 5                      

# EXTRACTION METHOD: 'docling' (High Fidelity/Slow) or 'fast' (Text Only/Fast)
EXTRACTION_METHOD: Literal["docling", "fast"] = "fast" 
# EXTRACTION_METHOD = os.getenv("EXTRACTION_METHOD", "fast")

# Ollama Cloud Settings
OLLAMA_HOST = "https://ollama.com"     
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
MODEL_NAME = "gpt-oss:120b-cloud"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("rfp_batch_extraction.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==========================================
# CORE EXTRACTION CLASS
# ==========================================

class BatchRFPExtractor:
    def __init__(self, method: str = "fast"):
        self.method = method
        self.client = AsyncClient(
            host=OLLAMA_HOST,
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
        )
        # Only initialize Docling if selected to save startup time/memory
        self.converter = DocumentConverter() if method == "docling" else None
        
        self.processed_files = self._load_progress()
        self.results = self._load_existing_csv()

    def _load_progress(self) -> Set[str]:
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, "r") as f:
                    return set(json.load(f))
            except Exception as e:
                logger.error(f"Error loading progress file: {e}")
        return set()

    def _save_progress(self):
        with open(PROGRESS_FILE, "w") as f:
            json.dump(list(self.processed_files), f, indent=4)

    def _load_existing_csv(self) -> List[dict]:
        if os.path.exists(OUTPUT_CSV):
            try:
                return pd.read_csv(OUTPUT_CSV).to_dict("records")
            except Exception as e:
                logger.error(f"Error loading CSV: {e}")
        return []

    def _save_csv(self):
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(OUTPUT_CSV, index=False)
            logger.info(f"Checkpoint: {len(self.results)} total records saved to {OUTPUT_CSV}")

    # --- Fast Extraction Methods ---
    def _extract_fast_pdf(self, path: Path) -> str:
        text = ""
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def _extract_fast_docx(self, path: Path) -> str:
        doc = DocxDocument(path)
        return "\n".join([para.text for para in doc.paragraphs])

    async def get_llm_analysis(self, text: str) -> dict:
        """Query Ollama with RFP definitions."""
        context = text[:40000] # Safe context window limit
        
        prompt = f"""
        Act as a professional RFP Analyst. Analyze the document provided below based on the following definitions:

        DEFINITIONS:
        1. REQUIREMENT: A clearly defined need, expectation, or condition that the organization wants a vendor to fulfill. It specifies functionality, performance, technical capabilities, or compliance standards. It outlines what must be delivered and any constraints.
        
        2. EVALUATION CRITERIA: Predefined standards used to assess vendor proposals. They explain how proposals will be scored (e.g., technical capability, experience, pricing, timeline).

        TASK:
        Identify and extract all Requirements and Evaluation Criteria found in the text.

        DOCUMENT CONTENT:
        {context}

        RESPONSE FORMAT (Strict JSON):
        {{
            "requirements": "Detailed list of extracted requirements...",
            "evaluation_criteria": "Detailed list of extracted evaluation criteria..."
        }}
        """

        try:
            response = await self.client.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                format="json" 
            )
            data = json.loads(response['message']['content'])
            return {
                "requirement": data.get("requirements", "None identified"),
                "evaluation_criteria": data.get("evaluation_criteria", "None identified")
            }
        except Exception as e:
            logger.error(f"LLM processing error: {e}")
            return {"requirement": "ERROR_IN_LLM", "evaluation_criteria": "ERROR_IN_LLM"}

    async def process_file(self, file_path: Path):
        try:
            logger.info(f"--- [{self.method.upper()}] Processing: {file_path.name} ---")
            
            raw_text = ""
            if self.method == "docling":
                conv_result = self.converter.convert(file_path)
                raw_text = conv_result.document.export_to_markdown()
            else:
                # Fast Method
                if file_path.suffix.lower() == ".pdf":
                    raw_text = self._extract_fast_pdf(file_path)
                elif file_path.suffix.lower() == ".docx":
                    raw_text = self._extract_fast_docx(file_path)

            if not raw_text.strip():
                logger.warning(f"File {file_path.name} resulted in empty text.")
                return False

            analysis = await self.get_llm_analysis(raw_text)
            
            self.results.append({
                "source": str(file_path.absolute()),
                "requirement": analysis["requirement"],
                "evaluation_criteria": analysis["evaluation_criteria"]
            })
            
            self.processed_files.add(str(file_path.absolute()))
            return True

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {str(e)}")
            return False

    async def run(self):
        all_files = []
        for ext in ["*.pdf", "*.docx"]:
            all_files.extend(list(Path(ROOT_FOLDER).rglob(ext)))

        files_to_process = [f for f in all_files if str(f.absolute()) not in self.processed_files]
        
        logger.info(f"Method: {self.method} | Found: {len(all_files)} | Remaining: {len(files_to_process)}")

        if not files_to_process:
            logger.info("Nothing to process.")
            return

        count = 0
        for file_path in files_to_process:
            success = await self.process_file(file_path)
            
            if success:
                count += 1
                if count % SAVE_INTERVAL == 0:
                    self._save_csv()
                    self._save_progress()
            
            await asyncio.sleep(0.2) # Throttling for API

        self._save_csv()
        self._save_progress()
        logger.info("Batch extraction finished.")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Change EXTRACTION_METHOD at the top of the file to switch loaders
    extractor = BatchRFPExtractor(method=EXTRACTION_METHOD)
    try:
        asyncio.run(extractor.run())
    except KeyboardInterrupt:
        logger.info("Process stopped. Progress saved.")