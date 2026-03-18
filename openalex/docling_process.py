"""
docling_process.py — Module 2: Process downloaded fulltexts through Docling.

Standalone — run directly:
    python docling_process.py
    python docling_process.py --incremental

Reads:   data/raw/works_with_fulltext.jsonl   (output of fetch_and_download.py)
         data/fulltext/{work_id}.pdf  OR  data/fulltext/{work_id}.tei.xml

Writes:  data/docling/{work_id}.json          — structured sections per work
         data/raw/works_with_docling.jsonl     — annotated with docling_status + docling_path

docling_status values:
    'docling_ok'   — Docling succeeded, rich structured sections available
    'fallback_pdf' — Docling failed, fell back to pdfminer raw text
    'none'         — no fulltext available (title+abstract only)

Install:
    pip install docling pdfminer.six
    pip install torch --index-url https://download.pytorch.org/whl/cu121
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Suppress Docling's per-document initialization noise
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("docling_core").setLevel(logging.WARNING)
logging.getLogger("docling.document_converter").setLevel(logging.WARNING)
logging.getLogger("docling.pipeline").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

DOCLING_DIR = Path(config.DOCLING_DIR)

STATUS_OK       = "docling_ok"
STATUS_FALLBACK = "fallback_pdf"
STATUS_NONE     = "none"

BATCH_SIZE = 8   # number of PDFs per Docling batch — tune if OOM


# ═══════════════════════════════════════════════════════════════════════════════
# Docling converter — GPU-aware, built once
# ═══════════════════════════════════════════════════════════════════════════════

_CONVERTER = None


def _build_converter():
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
    except ImportError:
        logger.error("Docling not installed — run: pip install docling")
        return None

    # Detect GPU
    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("CUDA available: %s (%.1f GB VRAM)",
                        torch.cuda.get_device_name(0),
                        torch.cuda.get_device_properties(0).total_memory / 1e9)
    except ImportError:
        pass

    # Resolve accelerator device
    accelerator_device = "cpu"
    if use_gpu:
        try:
            from docling.datamodel.pipeline_options import AcceleratorDevice
            accelerator_device = AcceleratorDevice.CUDA
        except (ImportError, AttributeError):
            os.environ["DOCLING_DEVICE"] = "cuda"
            accelerator_device = "cuda"

    # Build pipeline options — explicit and consistent so the hash never changes
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr             = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    try:
        from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=accelerator_device if accelerator_device != "cpu" else AcceleratorDevice.CPU,
        )
        logger.info("Docling accelerator: %s via AcceleratorOptions", "GPU (CUDA)" if use_gpu else "CPU")
    except (ImportError, AttributeError):
        if use_gpu:
            try:
                pipeline_options.accelerator_device = AcceleratorDevice.CUDA
                logger.info("Docling accelerator: GPU via accelerator_device")
            except Exception:
                logger.info("Docling accelerator: GPU via env var")
        else:
            logger.warning("CUDA not available — Docling running on CPU (slow)")

    try:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        logger.info("Docling converter ready")
        return converter
    except Exception as e:
        logger.error("Failed to build Docling converter: %s", e)
        return None


def _get_converter():
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = _build_converter()
    return _CONVERTER


# ═══════════════════════════════════════════════════════════════════════════════
# Docling output → sections
# ═══════════════════════════════════════════════════════════════════════════════

def _docling_to_sections(conv_res) -> List[Dict]:
    """
    Convert a Docling ConversionResult into a flat list of section dicts.
    Handles Docling >= 2.x (iterate_items) and 1.x (export_to_dict).
    """
    sections: List[Dict] = []
    current_heading = "Body"
    current_level   = 1
    current_paras:  List[str] = []

    def _flush():
        nonlocal current_paras
        text = " ".join(current_paras).strip()
        if text and len(text) > 30:
            sections.append({
                "heading":      current_heading,
                "level":        current_level,
                "text":         text,
                "element_type": "section",
            })
        current_paras = []

    try:
        # Docling >= 2.x
        for item, _ in conv_res.document.iterate_items():
            itype = type(item).__name__

            if itype in ("SectionHeaderItem", "HeadingItem"):
                _flush()
                current_heading = (getattr(item, "text", "") or "").strip() or current_heading
                current_level   = getattr(item, "level", 1) or 1

            elif itype == "TextItem":
                text = (getattr(item, "text", "") or "").strip()
                if text and len(text) > 10:
                    current_paras.append(text)

            elif itype == "TableItem":
                _flush()
                try:
                    # Pass doc to avoid deprecation warning in newer Docling
                    try:
                        table_md = item.export_to_markdown(doc=conv_res.document)
                    except TypeError:
                        table_md = item.export_to_markdown()
                    if table_md and len(table_md.strip()) > 10:
                        sections.append({
                            "heading":      f"{current_heading} [Table]",
                            "level":        current_level,
                            "text":         table_md.strip(),
                            "element_type": "table",
                        })
                except Exception:
                    pass

            elif itype == "FigureItem":
                caption = ""
                try:
                    caption = " ".join(
                        c.text for c in (getattr(item, "captions", []) or [])
                        if hasattr(c, "text")
                    ).strip()
                except Exception:
                    pass
                if caption:
                    _flush()
                    sections.append({
                        "heading":      f"{current_heading} [Figure]",
                        "level":        current_level,
                        "text":         caption,
                        "element_type": "figure_caption",
                    })

        _flush()

    except AttributeError:
        # Docling 1.x fallback
        try:
            d = conv_res.document.export_to_dict()
            for item in d.get("body", []):
                label = item.get("label", "")
                text  = (item.get("text") or "").strip()
                if label in ("section-header", "title"):
                    _flush()
                    current_heading = text or current_heading
                elif label == "text" and text and len(text) > 10:
                    current_paras.append(text)
                elif label == "table" and text:
                    _flush()
                    sections.append({
                        "heading":      f"{current_heading} [Table]",
                        "level":        1,
                        "text":         text,
                        "element_type": "table",
                    })
            _flush()
        except Exception as e:
            logger.debug("Docling 1.x export failed: %s", e)

    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# pdfminer fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _pdfminer_text(pdf_path: str) -> str:
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
        if text and len(text.strip()) > 100:
            return text.strip()
    except ImportError:
        logger.warning("pdfminer.six not installed — run: pip install pdfminer.six")
    except Exception as e:
        logger.debug("pdfminer failed for %s: %s", pdf_path, e)

    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        pages  = [p.extract_text() or "" for p in reader.pages]
        text   = "\n\n".join(p.strip() for p in pages if p.strip())
        if text and len(text.strip()) > 100:
            return text.strip()
    except Exception as e:
        logger.debug("pypdf failed for %s: %s", pdf_path, e)

    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Save helper
# ═══════════════════════════════════════════════════════════════════════════════

def _save_sections(work_id: str, ft_path: str, sections: List[Dict], status: str) -> str:
    DOCLING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCLING_DIR / f"{work_id}.json"
    out_path.write_text(
        json.dumps({
            "work_id":  work_id,
            "source":   ft_path,
            "status":   status,
            "sections": sections,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(out_path)


def _pdfminer_fallback(wid: str, ft_path: str) -> Tuple[str, str]:
    """Try pdfminer on a PDF. Returns (status, saved_path)."""
    if not ft_path.endswith(".pdf"):
        return STATUS_NONE, ""
    text = _pdfminer_text(ft_path)
    if text:
        saved = _save_sections(wid, ft_path, [{
            "heading": "Full Text", "level": 1,
            "text": text, "element_type": "section",
        }], STATUS_FALLBACK)
        return STATUS_FALLBACK, saved
    return STATUS_NONE, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run(incremental: bool = False) -> Dict[str, int]:
    """
    Process all works with downloaded fulltexts through Docling using batch
    processing (convert_all) to keep the pipeline warm across documents.

    Safe to interrupt and resume — already-processed files in data/docling/
    are detected and skipped automatically.
    """
    works_in  = Path(config.RAW_DIR) / "works_with_fulltext.jsonl"
    works_out = Path(config.RAW_DIR) / "works_with_docling.jsonl"

    if not works_in.exists():
        logger.error("No works_with_fulltext.jsonl — run fetch_and_download.py first")
        return {STATUS_OK: 0, STATUS_FALLBACK: 0, STATUS_NONE: 0, "total": 0}

    # Load all works
    all_works: List[Dict] = []
    with open(works_in, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_works.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Build already_done from JSONL (incremental mode)
    already_done: Dict[str, Dict] = {}
    if incremental and works_out.exists():
        with open(works_out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    w   = json.loads(line)
                    wid = (w.get("id") or "").rsplit("/", 1)[-1]
                    if wid and "docling_status" in w:
                        already_done[wid] = w
                except json.JSONDecodeError:
                    continue

    # Detect works already processed on disk (safe resume after crash)
    pre_recovered = 0
    to_process    = []
    for w in all_works:
        wid      = (w.get("id") or "").rsplit("/", 1)[-1]
        out_path = DOCLING_DIR / f"{wid}.json"
        if wid in already_done:
            continue
        if out_path.exists() and out_path.stat().st_size > 100:
            w["docling_status"] = STATUS_OK
            w["docling_path"]   = str(out_path)
            already_done[wid]   = w
            pre_recovered      += 1
            continue
        to_process.append(w)

    if pre_recovered:
        logger.info("Auto-recovered %d already-processed works from data/docling/", pre_recovered)

    has_ft = [w for w in to_process if w.get("fulltext_status", STATUS_NONE) != STATUS_NONE]
    no_ft  = [w for w in to_process if w.get("fulltext_status", STATUS_NONE) == STATUS_NONE]

    logger.info(
        "Total: %d | Already done: %d | To process: %d (with fulltext: %d, no fulltext: %d)",
        len(all_works), len(already_done), len(to_process), len(has_ft), len(no_ft),
    )

    counts:  Dict[str, int]             = {STATUS_OK: 0, STATUS_FALLBACK: 0, STATUS_NONE: 0}
    results: Dict[str, Tuple[str, str]] = {}

    # Build lookup maps
    path_to_wid:  Dict[str, str]  = {}
    path_to_work: Dict[str, Dict] = {}
    pdf_paths:    List[str]       = []

    for work in has_ft:
        wid     = (work.get("id") or "").rsplit("/", 1)[-1]
        ft_path = work.get("fulltext_path", "")
        if ft_path and Path(ft_path).exists():
            path_to_wid[ft_path]  = wid
            path_to_work[ft_path] = work
            pdf_paths.append(ft_path)
        else:
            results[wid]        = (STATUS_NONE, "")
            counts[STATUS_NONE] += 1

    converter = _get_converter()
    t0        = time.time()
    done      = 0

    if converter is not None and pdf_paths:
        logger.info("Processing %d files through Docling (batch_size=%d)...", len(pdf_paths), BATCH_SIZE)

        # Pass ALL paths to convert_all at once — Docling keeps the pipeline
        # warm across all documents, no re-initialization per file.
        try:
            for conv_res in converter.convert_all(pdf_paths, raises_on_error=False):
                # Match result back to work_id
                try:
                    ft_path = str(conv_res.input.file)
                except Exception:
                    ft_path = ""

                wid = path_to_wid.get(ft_path, "")
                if not wid:
                    for p, w in path_to_wid.items():
                        if Path(p).name == Path(ft_path).name:
                            wid     = w
                            ft_path = p
                            break

                if not wid or wid in results:
                    continue

                done += 1

                try:
                    sections = _docling_to_sections(conv_res)
                    if not sections:
                        raise ValueError("0 sections extracted")
                    saved             = _save_sections(wid, ft_path, sections, STATUS_OK)
                    results[wid]      = (STATUS_OK, saved)
                    counts[STATUS_OK] += 1

                except Exception as e:
                    logger.debug("Extraction failed %s: %s — pdfminer fallback", wid, e)
                    status, saved   = _pdfminer_fallback(wid, ft_path)
                    results[wid]    = (status, saved)
                    counts[status] += 1

                if done % 50 == 0 or done == len(pdf_paths):
                    elapsed = time.time() - t0
                    rate    = done / max(1, elapsed)
                    eta     = (len(pdf_paths) - done) / max(0.01, rate)
                    logger.info(
                        "Progress: %d/%d (%.1f/min) | OK: %d | Fallback: %d | None: %d | ETA: %.0fs",
                        done, len(pdf_paths), rate * 60,
                        counts[STATUS_OK], counts[STATUS_FALLBACK], counts[STATUS_NONE], eta,
                    )

        except Exception as err:
            logger.warning("convert_all failed: %s — falling back per-file", err)
            for ft_path in pdf_paths:
                wid = path_to_wid.get(ft_path, "")
                if not wid or wid in results:
                    continue
                try:
                    conv_res = converter.convert(ft_path)
                    sections = _docling_to_sections(conv_res)
                    if not sections:
                        raise ValueError("0 sections")
                    saved             = _save_sections(wid, ft_path, sections, STATUS_OK)
                    results[wid]      = (STATUS_OK, saved)
                    counts[STATUS_OK] += 1
                except Exception:
                    status, saved   = _pdfminer_fallback(wid, ft_path)
                    results[wid]    = (status, saved)
                    counts[status] += 1
                done += 1

    elif pdf_paths:
        # Docling not available — pdfminer for everything
        logger.warning("Docling unavailable — using pdfminer fallback for all %d files", len(pdf_paths))
        for ft_path in pdf_paths:
            wid                = path_to_wid.get(ft_path, "")
            status, saved      = _pdfminer_fallback(wid, ft_path)
            results[wid]       = (status, saved)
            counts[status]    += 1

    # Works with no fulltext
    for work in no_ft:
        wid = (work.get("id") or "").rsplit("/", 1)[-1]
        results[wid]        = (STATUS_NONE, "")
        counts[STATUS_NONE] += 1

    # Write output JSONL
    with open(works_out, "w", encoding="utf-8") as f:
        for w in already_done.values():
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
        for work in to_process:
            wid                    = (work.get("id") or "").rsplit("/", 1)[-1]
            status, path           = results.get(wid, (STATUS_NONE, ""))
            work["docling_status"] = status
            work["docling_path"]   = path
            f.write(json.dumps(work, ensure_ascii=False) + "\n")

    summary = {**counts, "total": len(all_works)}
    logger.info("━" * 50)
    logger.info(
        "DONE: OK=%d | Fallback=%d | None=%d → %s",
        counts[STATUS_OK], counts[STATUS_FALLBACK], counts[STATUS_NONE], works_out,
    )
    logger.info("━" * 50)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Process downloaded fulltexts with Docling")
    parser.add_argument("--incremental", action="store_true", help="Skip already-processed works")
    args = parser.parse_args()
    print(run(incremental=args.incremental))