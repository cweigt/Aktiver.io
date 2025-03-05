from fastapi import FastAPI, UploadFile, File, Depends, Form
from sqlalchemy.orm import Session
import fitz  # PyMuPDF
from typing import List
import math
import spacy  # or transformers from HuggingFace
import re
import json
from models import Base, engine, get_db, TextChunk, MolecularTarget, Therapy

app = FastAPI()

Base.metadata.create_all(bind=engine)

# Example spaCy pipeline:
nlp = spacy.load("en_core_web_sm")  # Replace with your custom biomedical model


def compute_pfs(conf: float):
    """
    Example function to compute Pythagorean Fuzzy Set membership (MY),
    non-membership (MN), and hesitancy (H).
    This is a simplistic approach; adapt as needed.
    """
    MY = conf
    MN = 0.0 if conf > 0.8 else (1.0 - conf) * 0.2
    H = math.sqrt(abs(1 - MY**2 - MN**2))
    return MY, MN, H


def chunk_text(text: str, chunk_size=50) -> List[str]:
    """
    Naive approach to split text into 50-word segments.
    """
    words = re.split(r"\s+", text)
    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


@app.post("/upload/")
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    1) Upload PDF
    2) Parse text
    3) Chunk into ~50 words
    4) Store each chunk in the DB with is_annotated=False
    5) Optionally run an initial NER pass, store auto-labeled results
    """
    content = await file.read()
    #streaming to Prodigy...?
    doc = fitz.open(stream=content, filetype="pdf")
    full_text = " ".join([page.get_text() for page in doc])
    text_chunks = chunk_text(full_text, chunk_size=50)

    for chunk in text_chunks:
        # Create the chunk record:
        new_chunk = TextChunk(
            text=chunk,
            is_annotated=False,
            annotation_json=None,  # We’ll fill this in after manual annotation
        )
        db.add(new_chunk)

    db.commit()
    return {
        "message": "PDF uploaded and text chunked",
        "total_chunks": len(text_chunks),
    }


@app.get("/annotation/next")
def get_next_unlabeled_chunk(db: Session = Depends(get_db)):
    """
    Returns the next chunk that is not yet annotated
    so the user can label it in a custom UI.
    """
    chunk = db.query(TextChunk).filter_by(is_annotated=False).first()
    if chunk:
        return {"chunk_id": chunk.id, "text": chunk.text}
    return {"message": "No unannotated chunks left!"}


@app.post("/annotation/submit")
def submit_annotation(
    chunk_id: int = Form(...),
    annotation_json: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Endpoint that receives user-submitted annotation in JSON form.
    """
    chunk = db.query(TextChunk).filter_by(id=chunk_id).first()
    if not chunk:
        return {"error": "No chunk with this ID"}

    # Save the annotation
    chunk.annotation_json = annotation_json
    chunk.is_annotated = True
    db.commit()

    return {"message": f"Chunk {chunk_id} annotated successfully"}


@app.post("/ner-auto/")
def run_auto_ner(db: Session = Depends(get_db)):
    """
    Example of an auto-labelling step that runs spaCy NER on all unannotated chunks.
    """
    unannotated = db.query(TextChunk).filter_by(is_annotated=False).all()

    for chunk in unannotated:
        doc = nlp(chunk.text)
        results = []

        for ent in doc.ents:
            # Fake confidence
            raw_conf = 0.9
            my, mn, h = compute_pfs(raw_conf)

            results.append({
                "text": ent.text,
                "label": ent.label_,
                "confidence": raw_conf,
                "my": my,
                "mn": mn,
                "hesitancy": h,
            })

        chunk.annotation_json = json.dumps(results)
        # We might still want a human to review, so is_annotated=False
        db.commit()

    return {"message": "Auto-NER complete", "count": len(unannotated)}


@app.post("/extract-entities/")
def extract_entities_into_tables(db: Session = Depends(get_db)):
    """
    Parse annotation JSON and store recognized MolecularTargets or Therapies.
    """
    annotated = db.query(TextChunk).filter_by(is_annotated=True).all()
    count_targets = 0
    count_therapies = 0

    for chunk in annotated:
        if not chunk.annotation_json:
            continue

        data = json.loads(chunk.annotation_json)

        for ent_info in data:
            label = ent_info.get("label", "")
            text_value = ent_info.get("text", "")
            raw_conf = ent_info.get("confidence", 0.5)
            my = ent_info.get("my", 0.5)
            mn = ent_info.get("mn", 0.0)
            h = ent_info.get("hesitancy", 0.0)

            if label in ("THERAPY", "DRUG", "TREATMENT"):
                db_obj = Therapy(
                    name=text_value,
                    my=my,
                    mn=mn,
                    hesitancy=h,
                    confidence=raw_conf,
                )
                db.add(db_obj)
                count_therapies += 1

            elif label in ("MOLECULAR_TARGET", "GENE", "PROTEIN"):
                db_obj = MolecularTarget(
                    name=text_value,
                    my=my,
                    mn=mn,
                    hesitancy=h,
                    confidence=raw_conf,
                )
                db.add(db_obj)
                count_targets += 1

    db.commit()
    return {
        "message": "Entities extracted",
        "targets": count_targets,
        "therapies": count_therapies,
    }


@app.get("/results/molecular-targets")
def get_molecular_targets(db: Session = Depends(get_db)):
    return db.query(MolecularTarget).all()


@app.get("/results/therapies")
def get_therapies(db: Session = Depends(get_db)):
    return db.query(Therapy).all()
