import os
import asyncio
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Set, Optional
from ollama import AsyncClient
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
ROOT_FOLDER = "/home/pyatrix/Downloads/rfp-sam-gov/output"             # Folder containing subfolders of documents
OUTPUT_CSV = "rfp_extraction_results.csv"
PROGRESS_FILE = "extraction_progress.json"
SAVE_INTERVAL = 5                      # Save files every 5 iterations

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
    def __init__(self):
        self.client = AsyncClient(
            host=OLLAMA_HOST,
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
        )
        self.converter = DocumentConverter()
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

    async def get_llm_analysis(self, text: str) -> dict:
        """Query Ollama with user-defined RFP Requirement and Criteria definitions."""
        
        # Take a large chunk of text (approx 40k characters)
        # RFPs are usually long; this captures the most critical sections
        context = text[:40000] 
        
        prompt = f"""
        Act as a professional RFP Analyst. Analyze the document provided below based on the following definitions:

        DEFINITIONS:
        1. REQUIREMENT: A clearly defined need, expectation, or condition that the organization wants a vendor to fulfill. It specifies what is being looked for in terms of functionality, performance, technical capabilities, business outcomes, or compliance standards. It outlines what must be delivered, how it should perform, and any constraints.
        
        2. EVALUATION CRITERIA: Predefined standards and factors used to assess and compare vendor proposals. They explain how proposals will be scored and what aspects are most important (e.g., technical capability, experience, methodology, pricing, timeline).

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
                format="json" # Ensures structured output
            )
            
            content = response['message']['content']
            data = json.loads(content)
            return {
                "requirement": data.get("requirements", "None identified"),
                "evaluation_criteria": data.get("evaluation_criteria", "None identified")
            }
        except Exception as e:
            logger.error(f"LLM processing error: {e}")
            return {"requirement": "ERROR_IN_EXTRACTION", "evaluation_criteria": "ERROR_IN_EXTRACTION"}

    async def process_file(self, file_path: Path):
        """Convert and analyze an individual file."""
        try:
            logger.info(f"--- Processing: {file_path.name} ---")
            
            # 1. High-fidelity conversion using Docling
            conv_result = self.converter.convert(file_path)
            md_text = conv_result.document.export_to_markdown()

            if not md_text.strip():
                logger.warning(f"File {file_path.name} resulted in empty text.")
                return False

            # 2. Extract specific RFP components using LLM
            analysis = await self.get_llm_analysis(md_text)
            
            # 3. Store in results
            self.results.append({
                "source": str(file_path.absolute()),
                "requirement": analysis["requirement"],
                "evaluation_criteria": analysis["evaluation_criteria"]
            })
            
            # 4. Mark as processed
            self.processed_files.add(str(file_path.absolute()))
            return True

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {str(e)}")
            return False

    async def run(self):
        """Traverse folders and manage batching."""
        # Find all PDF and DOCX files recursively in all subfolders
        all_files = []
        for ext in ["*.pdf", "*.docx"]:
            all_files.extend(list(Path(ROOT_FOLDER).rglob(ext)))

        files_to_process = [f for f in all_files if str(f.absolute()) not in self.processed_files]
        
        logger.info(f"Total files found: {len(all_files)}")
        logger.info(f"Already processed: {len(self.processed_files)}")
        logger.info(f"Remaining to process: {len(files_to_process)}")

        if not files_to_process:
            logger.info("No new files to process.")
            return

        count = 0
        for file_path in files_to_process:
            success = await self.process_file(file_path)
            
            if success:
                count += 1
                # Periodic Save Logic
                if count % SAVE_INTERVAL == 0:
                    self._save_csv()
                    self._save_progress()
                    logger.info(f"Saved progress after {count} files.")
            
            # Gentle delay for Cloud API stability
            await asyncio.sleep(0.5)

        # Final terminal save
        self._save_csv()
        self._save_progress()
        logger.info("Batch extraction successfully completed.")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    extractor = BatchRFPExtractor()
    try:
        asyncio.run(extractor.run())
    except KeyboardInterrupt:
        logger.info("Extraction paused by user. Progress has been saved.")
    except Exception as e:
        logger.critical(f"Unhandled system error: {e}")