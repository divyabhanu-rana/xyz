import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sentence_transformers import SentenceTransformer

from .pdf_ingest import (
    process_pdf,
    save_vectorstore,
    DATA_DIR,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
)

class PDFHandler(FileSystemEventHandler):
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(".pdf"):
            self._process_new_pdf(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        dest = getattr(event, "dest_path", "")
        if dest and dest.lower().endswith(".pdf"):
            self._process_new_pdf(dest)

    def _process_new_pdf(self, pdf_path: str):
        try:
            time.sleep(1.0)

            rel_path = os.path.relpath(pdf_path, DATA_DIR)
            out_filename = rel_path.replace(".pdf", "_vectors.json")
            out_path = os.path.join(VECTORSTORE_DIR, out_filename)

            if os.path.exists(out_path):
                print(f"[watcher] Vectors already exist for: {rel_path}")
                return

            records = process_pdf(pdf_path, self.model)
            if not records:
                print(f"[watcher] Skipped (no extractable text): {rel_path}")
                return

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_vectorstore(records, out_path)
            print(f"[watcher] Processed {rel_path} -> {out_path} ({len(records)} pages)")
        except Exception as e:
            print(f"[watcher] Error processing {pdf_path}: {e}")

def start_watcher():
    """
    Start a file system watcher to monitor for new PDF files under DATA_DIR.
    Returns the Observer instance so the caller can keep it alive and stop it on shutdown.
    """
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        print(f"[watcher] Failed to load embedding model '{EMBEDDING_MODEL}': {e}")
        return None

    event_handler = PDFHandler(model)
    observer = Observer()
    observer.schedule(event_handler, DATA_DIR, recursive=True)
    observer.start()
    print(f"[watcher] Watching for new PDF files in: {DATA_DIR}")
    return observer

if __name__ == "__main__":
    obs = start_watcher()
    try:
        while obs and obs.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        if obs:
            obs.stop()
            obs.join()