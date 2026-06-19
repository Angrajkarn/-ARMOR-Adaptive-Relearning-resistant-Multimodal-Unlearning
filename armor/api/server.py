"""
armor/api/server.py
===================
Real-time Online Unlearning API Service with Automated GDPR Compliance Certification.
"""

import os
import uuid
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, HTTPException

import sys
import io

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

from armor.config import ARMORConfig
from armor.model import get_model_and_tokenizer
from armor.unlearn.hdi import HolographicInterference
from armor.data import make_dataloader, TOFUSample
from armor.eval.zk_verify import ZKVerifier, UnlearningCommitment
from armor.eval.privacy_audit import PrivacyAuditor
from armor.attack.reconstruction import TextReconstructionAttack
from armor.eval.certificate import AuditCertificateGenerator

# --- Data Models ---
class ForgetJobRequest(BaseModel):
    text_to_forget: str
    rephrases: Optional[List[str]] = None

class ForgetJobResponse(BaseModel):
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

class GenerateResponse(BaseModel):
    prompt: str
    completion: str

# --- Global State ---
app = FastAPI(title="ARMOR Online Unlearning API")
job_store: Dict[str, dict] = {}
certificate_store: Dict[str, dict] = {}
queue = asyncio.Queue()

# Globals for model
GLOBAL_MODEL: Optional[PreTrainedModel] = None
GLOBAL_TOKENIZER: Optional[PreTrainedTokenizer] = None
GLOBAL_CFG: Optional[ARMORConfig] = None
GLOBAL_LOCK = asyncio.Lock() # Protect model during HDI weight updates

def _text_to_samples(texts: List[str]) -> List[TOFUSample]:
    """Converts a list of strings into a list of TOFUSample objects."""
    samples = []
    for t in texts:
        samples.append(TOFUSample(
            question="What is the secret?", 
            answer=t,
            rephrases=[],
            split="forget"
        ))
    return samples

def _text_to_loader(texts: List[str], bsz: int = 4):
    """Converts a list of strings into an ARMOR dataloader."""
    samples = _text_to_samples(texts)
    # create dataloader using armor's helper
    loader = make_dataloader(samples, GLOBAL_TOKENIZER, GLOBAL_CFG, shuffle=False)
    return loader

async def unlearning_worker():
    """Background worker that continuously processes the unlearning queue and audits the updates."""
    print("[Worker] Online Unlearning Worker started.")
    while True:
        job_id, texts_to_forget = await queue.get()
        print(f"[Worker] Processing Job {job_id}...")
        job_store[job_id] = {"status": "processing", "message": "Applying Cryptographic Pre-Unlearn Commitment..."}
        
        try:
            # Prepare loaders and samples
            forget_samples = _text_to_samples(texts_to_forget)
            forget_loader = make_dataloader(forget_samples, GLOBAL_TOKENIZER, GLOBAL_CFG, shuffle=False)
            
            # Dummy retain set to orthogonalize against basic English structure
            retain_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subfield of artificial intelligence.",
                "Python is a popular programming language.",
                "The sky is blue and the grass is green."
            ]
            retain_loader = _text_to_loader(retain_texts)

            # Compute pre-hashes and commitments
            pre_model_hash = UnlearningCommitment.hash_model_weights(GLOBAL_MODEL)
            forget_set_hash = UnlearningCommitment.hash_forget_set(forget_loader)

            # ZK commitment Phase 1
            verifier = ZKVerifier(GLOBAL_CFG)
            verifier.commit_pre(GLOBAL_MODEL, forget_loader, retain_loader, method="HDI")

            # We must lock the model while capturing activations and injecting interference
            job_store[job_id] = {"status": "processing", "message": "Applying HDI Projection weight update..."}
            async with GLOBAL_LOCK:
                hdi = HolographicInterference(GLOBAL_MODEL, GLOBAL_CFG)
                # HDI unlearn runs synchronously, but it's very fast
                hdi.unlearn(forget_loader, retain_loader)

            # Compute post-hashes
            post_model_hash = UnlearningCommitment.hash_model_weights(GLOBAL_MODEL)

            # Audit Phase
            job_store[job_id] = {"status": "processing", "message": "Executing compliance audits (ZK, MIA, Inversion)..."}
            
            # ZK Verification Phase 2
            zk_report = verifier.verify_post(GLOBAL_MODEL, forget_loader, retain_loader)

            # Privacy Audit (MIA AUROC & DP Bound)
            auditor = PrivacyAuditor(GLOBAL_CFG, GLOBAL_MODEL, GLOBAL_TOKENIZER)
            privacy_report = auditor.audit(forget_loader, retain_loader, method_name="HDI")

            # Reconstruction Audit (Model Inversion)
            attacker = TextReconstructionAttack(GLOBAL_CFG, GLOBAL_MODEL, GLOBAL_TOKENIZER)
            reconstruction_report = attacker.run_reconstruction_attack(forget_samples, method_name="HDI")

            # Compile and Sign GDPR Compliance Certificate
            job_store[job_id] = {"status": "processing", "message": "Generating signed compliance certificate..."}
            cert_gen = AuditCertificateGenerator(GLOBAL_CFG, private_signing_key="ARMOR_ONLINE_API_KEY_2026")
            certificate, html_path = cert_gen.generate_certificate(
                model_name=GLOBAL_CFG.model_name,
                method_name="HDI",
                pre_model_hash=pre_model_hash,
                post_model_hash=post_model_hash,
                forget_set_hash=forget_set_hash,
                zk_report=zk_report,
                privacy_report=privacy_report,
                reconstruction_report=reconstruction_report,
                output_dir="outputs/audit"
            )

            # Store the final certificate
            certificate_store[job_id] = certificate
            
            job_store[job_id] = {"status": "completed", "message": f"Knowledge Eradicated. Signed compliance certificate ready."}
            print(f"[Worker] Job {job_id} Completed and compliance certificate generated.")
        except Exception as e:
            job_store[job_id] = {"status": "failed", "message": str(e)}
            print(f"[Worker] Job {job_id} Failed: {e}")
            
        queue.task_done()

@app.on_event("startup")
async def startup_event():
    global GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_CFG
    print("=" * 60)
    print("  ARMOR Online Unlearning API Booting...")
    
    # In production, these would come from env vars
    # We use debug=True to load distilgpt2 quickly
    GLOBAL_CFG = ARMORConfig(
        model_key="debug",
        debug=True,
    )
    # We target output layers for HDI
    GLOBAL_CFG.hdi_target_layers = ['c_proj', 'out_proj']
    
    print(f"  Loading Base Model: {GLOBAL_CFG.model_name}")
    GLOBAL_MODEL, GLOBAL_TOKENIZER = get_model_and_tokenizer(GLOBAL_CFG)
    
    # Start the background unlearning worker
    asyncio.create_task(unlearning_worker())
    
    print("  API Server Ready.")
    print("=" * 60)

@app.post("/unlearn", response_model=ForgetJobResponse)
async def submit_unlearn_job(req: ForgetJobRequest):
    """Submit a request to forget specific text or concepts."""
    job_id = str(uuid.uuid4())
    
    texts = [req.text_to_forget]
    if req.rephrases:
        texts.extend(req.rephrases)
        
    job_store[job_id] = {"status": "queued", "message": "Waiting in line..."}
    await queue.put((job_id, texts))
    
    return ForgetJobResponse(job_id=job_id, status="queued")

@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Check the status of an unlearning job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    info = job_store[job_id]
    return JobStatusResponse(job_id=job_id, status=info["status"], message=info["message"])

@app.get("/certificate/{job_id}")
async def get_certificate(job_id: str):
    """Retrieve the signed GDPR compliance certificate for a completed unlearning job."""
    if job_id not in certificate_store:
        if job_id in job_store:
            return {
                "job_id": job_id, 
                "status": job_store[job_id]["status"], 
                "detail": f"Certificate is currently being compiled (Status: {job_store[job_id]['message']}). Please wait."
            }
        raise HTTPException(status_code=404, detail="Certificate or job not found")
    return certificate_store[job_id]

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(req: GenerateRequest):
    """Query the live model to check its knowledge."""
    async with GLOBAL_LOCK:
        inputs = GLOBAL_TOKENIZER(req.prompt, return_tensors="pt").to(GLOBAL_CFG.device)
        with torch.no_grad():
            outputs = GLOBAL_MODEL.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                pad_token_id=GLOBAL_TOKENIZER.eos_token_id
            )
        
        completion = GLOBAL_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        # return only the generated part
        completion = completion[len(req.prompt):].strip()
        
    return GenerateResponse(prompt=req.prompt, completion=completion)

