"""
Julie Web Interface - FastAPI Backend

Provides REST API and WebSocket endpoints for:
- Admin panel (document upload, configuration, stats)
- Voice/chat interaction with Julie
"""

import os
import io
import json
import uuid
import asyncio
import base64
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Julie imports
from julie.core.agent import Agent
from julie.core.logging import CallLogger
from julie.rag.retriever import RAGRetriever
from julie.rag.loader import DocumentLoader
from julie.claims.service import ClaimsService


# ============================================================
# App Configuration
# ============================================================

app = FastAPI(
    title="Julie - CNP Assurances Voice Assistant",
    description="Interface d'administration et d'interaction vocale",
    version="1.0.0"
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Knowledge base directory
KNOWLEDGE_DIR = Path("knowledge/documents")
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

# Global instances (lazy loaded)
_agent: Optional[Agent] = None
_rag: Optional[RAGRetriever] = None
_logger: Optional[CallLogger] = None
_claims: Optional[ClaimsService] = None


def get_agent() -> Agent:
    """Get or create the Julie agent."""
    global _agent
    if _agent is None:
        _agent = Agent(
            enable_rag=True,
            knowledge_dir=str(KNOWLEDGE_DIR),
            enable_logging=True,
            use_streaming_tts=False,  # No streaming for web
        )
    return _agent


def get_rag() -> RAGRetriever:
    """Get or create the RAG retriever."""
    global _rag
    if _rag is None:
        _rag = RAGRetriever(in_memory=True)
        # Load existing documents
        if KNOWLEDGE_DIR.exists():
            _rag.ingest_directory(str(KNOWLEDGE_DIR))
    return _rag


def get_logger() -> CallLogger:
    """Get or create the call logger."""
    global _logger
    if _logger is None:
        _logger = CallLogger()
    return _logger


def get_claims() -> ClaimsService:
    """Get or create the claims service."""
    global _claims
    if _claims is None:
        _claims = ClaimsService()
        _claims.seed_mock_data()
    return _claims


# ============================================================
# Pydantic Models
# ============================================================

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: Optional[str] = None


class DocumentInfo(BaseModel):
    filename: str
    size: int
    uploaded_at: str
    chunks: int


class StatsResponse(BaseModel):
    total_calls: int
    total_documents: int
    total_claims: int
    resolution_rate: float
    avg_duration: Optional[float]


class ConfigUpdate(BaseModel):
    relevance_threshold: Optional[float] = None
    greeting_message: Optional[str] = None


# ============================================================
# API Routes - Admin
# ============================================================

@app.get("/api/stats")
async def get_stats():
    """Get dashboard statistics."""
    logger = get_logger()
    stats = logger.get_stats(days=30)
    
    # Count documents
    doc_count = sum(1 for _ in KNOWLEDGE_DIR.glob("*.*"))
    
    # Count claims
    claims = get_claims()
    
    return {
        "total_calls": stats.get("total_calls", 0),
        "total_documents": doc_count,
        "resolution_rate": stats.get("resolution_rate", 0),
        "avg_duration": stats.get("avg_duration_seconds"),
        "outcomes": stats.get("outcomes", {}),
        "calls_with_claims": stats.get("calls_with_claims_created", 0),
    }


@app.get("/api/documents")
async def list_documents():
    """List all documents in knowledge base."""
    documents = []
    
    for file_path in KNOWLEDGE_DIR.glob("*.*"):
        if file_path.suffix.lower() in [".txt", ".md", ".pdf"]:
            stat = file_path.stat()
            documents.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "type": file_path.suffix[1:].upper(),
            })
    
    return {"documents": sorted(documents, key=lambda x: x["uploaded_at"], reverse=True)}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the knowledge base."""
    # Validate file type
    allowed_extensions = [".txt", ".md", ".pdf"]
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Type de fichier non supporté. Types autorisés: {', '.join(allowed_extensions)}"
        )
    
    # Save file
    file_path = KNOWLEDGE_DIR / file.filename
    content = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Ingest into RAG
    try:
        rag = get_rag()
        loader = DocumentLoader()
        docs = loader.load_file(str(file_path))
        
        if docs:
            rag.add_documents(docs)
            chunk_count = len(docs)
        else:
            chunk_count = 0
        
        return {
            "success": True,
            "message": f"Document '{file.filename}' uploadé avec succès",
            "chunks": chunk_count,
        }
    except Exception as e:
        # Remove file if ingestion failed
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document from the knowledge base."""
    file_path = KNOWLEDGE_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document non trouvé")
    
    file_path.unlink()
    
    # Note: We'd need to rebuild the RAG index to fully remove
    # For now, just delete the file
    
    return {"success": True, "message": f"Document '{filename}' supprimé"}


@app.get("/api/calls")
async def list_calls(limit: int = 20):
    """Get recent call logs."""
    logger = get_logger()
    calls = logger.get_recent(limit=limit)
    
    return {"calls": calls}


@app.get("/api/calls/{call_id}")
async def get_call(call_id: str):
    """Get details of a specific call."""
    logger = get_logger()
    call = logger.get_by_id(call_id)
    
    if not call:
        raise HTTPException(status_code=404, detail="Appel non trouvé")
    
    return call


@app.get("/api/claims")
async def list_claims():
    """Get all claims."""
    claims = get_claims()
    all_claims = claims.get_all_claims()
    return {"claims": all_claims}


# ============================================================
# API Routes - Chat
# ============================================================

# Session storage (in production, use Redis)
sessions: dict = {}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Send a message to Julie and get a response."""
    agent = get_agent()
    
    # Create or get session
    session_id = message.session_id or str(uuid.uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "turns": [],
        }
    
    # Process message
    response = agent.process_text(message.message)
    
    # Classify intent
    intent, _ = agent.intent_classifier.classify(message.message)
    
    # Store turn
    sessions[session_id]["turns"].append({
        "user": message.message,
        "agent": response,
        "intent": intent.name,
        "timestamp": datetime.now().isoformat(),
    })
    
    return ChatResponse(
        response=response,
        session_id=session_id,
        intent=intent.name,
    )


@app.post("/api/chat/reset")
async def reset_chat(session_id: Optional[str] = None):
    """Reset a chat session."""
    if session_id and session_id in sessions:
        del sessions[session_id]
    
    agent = get_agent()
    agent.reset_conversation()
    
    return {"success": True, "message": "Session réinitialisée"}


# ============================================================
# WebSocket for Voice Call (Real Audio)
# ============================================================

class WebCallSession:
    """Manages a single web call session with audio processing."""
    
    def __init__(self, websocket: WebSocket, agent: Agent):
        self.websocket = websocket
        self.agent = agent
        self.call_id = str(uuid.uuid4())
        self.audio_buffer = io.BytesIO()
        self.is_active = True
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.last_audio_time = datetime.now()
        
        # Start call logging
        if agent.call_logger:
            agent.call_logger.start_call()
    
    async def send_status(self, status: str):
        """Send status update to client."""
        try:
            await self.websocket.send_json({"type": "status", "status": status})
        except:
            pass
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to client."""
        try:
            base64_audio = base64.b64encode(audio_data).decode('utf-8')
            await self.websocket.send_json({"type": "audio", "audio": base64_audio})
        except Exception as e:
            print(f"Error sending audio: {e}")
    
    async def send_end(self):
        """Signal call end to client."""
        try:
            await self.websocket.send_json({"type": "end"})
        except:
            pass


@app.websocket("/ws/call")
async def call_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for full voice call simulation.
    
    This replicates the main.py CLI experience on the web:
    - Receives audio chunks from browser
    - Uses VAD to detect speech end
    - Processes with STT -> LLM -> TTS
    - Streams audio back to browser
    """
    await websocket.accept()
    
    agent = get_agent()
    session = WebCallSession(websocket, agent)
    
    # Audio accumulator
    audio_chunks = []
    silence_count = 0
    speech_started = False
    SILENCE_THRESHOLD = 8  # ~2 seconds at 250ms chunks
    MIN_SPEECH_CHUNKS = 2  # Minimum chunks to consider as speech
    
    try:
        # Send welcome audio
        await session.send_status("speaking")
        welcome_text = agent.welcome()
        welcome_audio = await synthesize_speech(agent, welcome_text)
        if welcome_audio:
            await session.send_audio(welcome_audio)
        
        await session.send_status("listening")
        
        while session.is_active:
            try:
                # Receive with timeout
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=60.0  # 1 minute timeout
                )
            except asyncio.TimeoutError:
                # Timeout - end call
                goodbye = "Au revoir, à bientôt!"
                await session.send_status("speaking")
                goodbye_audio = await synthesize_speech(agent, goodbye)
                if goodbye_audio:
                    await session.send_audio(goodbye_audio)
                await session.send_end()
                break
            
            if data.get("type") == "audio":
                # Decode audio chunk
                audio_base64 = data.get("audio", "")
                if not audio_base64:
                    continue
                
                try:
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_chunks.append(audio_bytes)
                    
                    # Simple voice activity detection based on chunk size
                    # Larger chunks typically indicate more audio activity
                    if len(audio_bytes) > 500:  # Has some audio
                        silence_count = 0
                        speech_started = True
                    elif speech_started:
                        silence_count += 1
                    
                    # Process when silence detected after speech
                    if speech_started and silence_count >= SILENCE_THRESHOLD:
                        if len(audio_chunks) >= MIN_SPEECH_CHUNKS:
                            await session.send_status("processing")
                            
                            # Combine audio chunks
                            combined_audio = b''.join(audio_chunks)
                            
                            # Process the audio
                            response_audio = await process_voice_turn(
                                agent, session, combined_audio
                            )
                            
                            if response_audio is None:
                                # Call ended
                                await session.send_end()
                                break
                        
                        # Reset for next turn
                        audio_chunks = []
                        silence_count = 0
                        speech_started = False
                        await session.send_status("listening")
                
                except Exception as e:
                    print(f"Error processing audio chunk: {e}")
            
            elif data.get("type") == "playback_done":
                # Client finished playing audio, ready to listen
                await session.send_status("listening")
            
            elif data.get("type") == "hangup":
                # User hung up
                break
    
    except WebSocketDisconnect:
        print(f"Call {session.call_id} disconnected")
    except Exception as e:
        print(f"Call error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        # End call logging
        if agent.call_logger:
            agent.call_logger.end_call(session.call_id, "completed")


async def process_voice_turn(agent: Agent, session: WebCallSession, audio_data: bytes) -> Optional[bytes]:
    """
    Process a single voice turn: STT -> LLM -> TTS
    
    Returns audio bytes or None if call should end.
    """
    try:
        # Save audio to temp file for STT
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name
        
        try:
            # Transcribe with STT
            user_text = agent.stt.transcribe_file(temp_path)
        finally:
            # Clean up temp file
            os.unlink(temp_path)
        
        if not user_text or len(user_text.strip()) < 2:
            # No speech detected
            await session.send_status("listening")
            return b''
        
        print(f"User: {user_text}")
        
        # Check for exit
        if agent.should_exit(user_text):
            goodbye = "Au revoir, merci de votre appel!"
            await session.send_status("speaking")
            goodbye_audio = await synthesize_speech(agent, goodbye)
            if goodbye_audio:
                await session.send_audio(goodbye_audio)
            return None  # Signal end
        
        # Process with LLM
        response_text = agent.process_text(user_text)
        print(f"Julie: {response_text}")
        
        # Log the turn
        if agent.call_logger:
            intent, confidence = agent.intent_classifier.classify(user_text)
            agent.call_logger.log_turn(
                session.call_id, user_text, response_text,
                intent=intent.name, confidence=confidence
            )
        
        # Synthesize response
        await session.send_status("speaking")
        response_audio = await synthesize_speech(agent, response_text)
        
        if response_audio:
            await session.send_audio(response_audio)
        
        return response_audio or b''
    
    except Exception as e:
        print(f"Error in voice turn: {e}")
        error_msg = "Désolée, je n'ai pas compris. Pouvez-vous répéter?"
        await session.send_status("speaking")
        error_audio = await synthesize_speech(agent, error_msg)
        if error_audio:
            await session.send_audio(error_audio)
        return b''


async def synthesize_speech(agent: Agent, text: str) -> Optional[bytes]:
    """
    Synthesize speech using the agent's TTS provider.
    Returns MP3/WAV bytes suitable for browser playback.
    """
    try:
        # Use TTS to synthesize
        audio_data = agent.tts.synthesize(text)
        
        if audio_data:
            return audio_data
        
        return None
    except Exception as e:
        print(f"TTS error: {e}")
        return None


# Keep old text-based WebSocket for backward compatibility
@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket endpoint for text-based voice interaction (legacy)."""
    await websocket.accept()
    
    agent = get_agent()
    session_id = str(uuid.uuid4())
    
    try:
        # Send welcome message
        welcome = agent.welcome()
        await websocket.send_json({
            "type": "welcome",
            "text": welcome,
            "session_id": session_id,
        })
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get("type") == "text":
                # Text message
                user_text = data.get("text", "")
                
                if agent.should_exit(user_text):
                    await websocket.send_json({
                        "type": "goodbye",
                        "text": "Au revoir!",
                    })
                    break
                
                response = agent.process_text(user_text)
                intent, confidence = agent.intent_classifier.classify(user_text)
                
                await websocket.send_json({
                    "type": "response",
                    "text": response,
                    "intent": intent.name,
                    "confidence": confidence,
                })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })


# ============================================================
# HTML Pages
# ============================================================

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon."""
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin", response_class=HTMLResponse)
async def admin():
    """Serve the admin panel."""
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/voice", response_class=HTMLResponse)
async def voice():
    """Serve the voice interaction page."""
    return FileResponse(STATIC_DIR / "voice.html")


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
