import os
from typing import List
from sentence_transformers import SentenceTransformer

from .pdf_ingest import (
     find_pdfs_recursively,
     process_pdf,
     save_vectorstore,
     DATA_DIR,
     VECTORSTORE_DIR,
     EMBEDDING_MODEL,
)

def _vector_path_for(pdf_path: str) -> str:
     rel_path = os.path.relpath(pdf_path, DATA_DIR)
     out_filename = rel_path.replace(".pdf", "_vectors.json")
     return os.path.join(VECTORSTORE_DIR, out_filename)

def _needs_processing(pdf_path: str) -> bool:
     out_path = _vector_path_for(pdf_path)
     if not os.path.exists(out_path):
          return True
     return os.path.getmtime(pdf_path) > os.path.getmtime(out_path)

def auto_vectorize() -> int:
     """
     Automatically vectorize all eligible PDF files.
     """

     os.makedirs(VECTORSTORE_DIR, exist_ok=True)
     pdf_files: List[str] = find_pdfs_recursively(DATA_DIR)
     if not pdf_files:
          print("[auto] No PDF files found.")
          return 0
     
     try:
          model = SentenceTransformer(EMBEDDING_MODEL)
     except Exception as e:
          print(f"[auto] Failed to load embedding model '{EMBEDDING_MODEL}': {e}")
          return 0
     
     processed_count = 0
     for pdf_path in pdf_files:
          if not _needs_processing(pdf_path):
               continue

          try:
               out_path = _vector_path_for(pdf_path)
               records = process_pdf(pdf_path, model)
               if not records:
                    print(f"[auto] Skipped (no extractable text): {os.path.relpath(pdf_path, DATA_DIR)}")
                    continue

               os.makedirs(os.path.dirname(out_path), exist_ok = True)
               save_vectorstore(records, out_path)
               print(f"[auto] Processed {os.path.relpath(pdf_path, DATA_DIR)} -> {out_path} ({len(records)} pages)")
               processed_count += 1
          except Exception as e:
               print(f"[auto] Error processing {pdf_path}: {e}")

     if processed_count == 0:
          print("[auto] No new or updated PDF files to process.")
     else: 
          print(f"[auto] Completed vectorization. {processed_count} files processed successfully.")
     return processed_count