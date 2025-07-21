import os
import time
import uuid
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from pinecone import Pinecone, ServerlessSpec
# import redis
from botocore.exceptions import NoCredentialsError, ClientError

# ---- ENVIRONMENT SETUP ----
from dotenv import load_dotenv
load_dotenv()

# ---- CONFIG ----
MAX_FILE_SIZE_MB = 30
MAX_FILES_PER_SESSION = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SESSION_EXPIRE_HOURS = 24

# Environment variables with validation
REQUIRED_VARS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "PINECONE_API_KEY", "S3_BUCKET_NAME"]
for var in REQUIRED_VARS:
    if not os.getenv(var):
        raise ValueError(f"Required environment variable {var} not set")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
REGION = os.getenv("AWS_REGION", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "document-qa-index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama3-8b-8192")

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

# ---- LOGGING SETUP ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---- GLOBAL SERVICES ----
class Services:
    def __init__(self):
        self.s3 = None
        self.pinecone = None
        self.embedding = None
        self.llm = None
        self.redis = None
        self.vector_store = None
    
    async def initialize(self):
        """Initialize all services with proper error handling"""
        try:
            # S3 Client
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=REGION
            )
            
            # Verify S3 bucket exists
            try:
                self.s3.head_bucket(Bucket=S3_BUCKET_NAME)
            except ClientError:
                logger.error(f"S3 bucket {S3_BUCKET_NAME} not accessible")
                raise
            
            # Pinecone
            self.pinecone = Pinecone(api_key=PINECONE_API_KEY)
            if PINECONE_INDEX not in [i.name for i in self.pinecone.list_indexes().indexes]:
                self.pinecone.create_index(
                    name=PINECONE_INDEX,
                    dimension=768,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region=REGION)
                )
                # Wait for index to be ready
                while not self.pinecone.describe_index(PINECONE_INDEX).status['ready']:
                    await asyncio.sleep(1)
            
            # Initialize embedding and LLM
            self.embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
            self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)
        
            
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            raise

services = Services()

# ---- LIFESPAN MANAGER ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await services.initialize()
    yield
    # Cleanup
    if services.redis:
        await services.redis.close()

# ---- FASTAPI APP ----
app = FastAPI(
    title="Smart Document QA API",
    description="Upload PDFs and ask questions based on document content",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- ENHANCED MODELS ----
class UploadResponse(BaseModel):
    message: str
    files: List[str]
    s3_keys: List[str]
    user_id: str
    expires_at: Optional[datetime] = None

class IngestRequest(BaseModel):
    s3_keys: List[str]
    
    @validator('s3_keys')
    def validate_keys(cls, v):
        if not v:
            raise ValueError("At least one S3 key required")
        return v

class IngestResponse(BaseModel):
    message: str
    chunks: int
    user_id: str
    processing_time: float

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = True

class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    user_id: str
    confidence: Optional[float] = None
    processing_time: float

class SessionInfo(BaseModel):
    user_id: str
    documents_count: int
    chunks_count: int
    created_at: datetime
    expires_at: datetime

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# ---- ENHANCED HELPERS ----
def get_user_id(request: Request) -> str:
    """Get or create user session ID"""
    if 'x-session-id' in request.headers:
        user_id = request.headers['x-session-id']
    else:
        user_id = str(uuid.uuid4())
    
    # Store session info in Redis if available
    if services.redis:
        try:
            expire_time = datetime.now() + timedelta(hours=SESSION_EXPIRE_HOURS)
            services.redis.setex(
                f"session:{user_id}",
                SESSION_EXPIRE_HOURS * 3600,
                expire_time.isoformat()
            )
        except Exception as e:
            logger.warning(f"Failed to store session in Redis: {e}")
    
    return user_id

async def validate_uploads(files: List[UploadFile]) -> None:
    """Enhanced file validation"""
    if len(files) > MAX_FILES_PER_SESSION:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FILES_PER_SESSION} files allowed per session"
        )
    
    for file in files:
        # Check file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a PDF"
            )
        
        # Check file size
        if file.size and file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File '{file.filename}' exceeds {MAX_FILE_SIZE_MB}MB limit"
            )
        
        # Check if file is empty
        if file.size == 0:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is empty"
            )

async def cleanup_temp_files(*file_paths: str) -> None:
    """Clean up temporary files"""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to cleanup {path}: {e}")

# ---- ENHANCED PROMPT TEMPLATE ----
enhanced_prompt = ChatPromptTemplate.from_template("""
You are an expert AI assistant specialized in analyzing documents and providing accurate, helpful responses.

INSTRUCTIONS:
- Provide detailed, accurate answers based ONLY on the provided context
- If the answer isn't in the context, clearly state "I cannot find this information in the provided documents"
- Use specific quotes or references from the documents when possible
- Structure your response clearly with relevant details
- If asked about multiple topics, address each one separately

CONTEXT:
{context}

QUESTION: {input}

RESPONSE:
""")

# ---- ENHANCED ENDPOINTS ----
@app.get("/")
def read_root():
    return {"message": "Backend is up and running."}


@app.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    request: Request = None
):
    """Upload PDF files to S3"""
    start_time = time.time()
    
    try:
        await validate_uploads(files)
        user_id = get_user_id(request)
        
        s3_keys = []
        uploaded_files = []
        
        for file in files:
            # Generate unique S3 key
            file_extension = Path(file.filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            key = f"{user_id}/{unique_filename}"
            
            # Upload to S3
            try:
                services.s3.upload_fileobj(file.file, S3_BUCKET_NAME, key)
                s3_keys.append(key)
                uploaded_files.append(file.filename)
                logger.info(f"Uploaded {file.filename} as {key}")
            except Exception as e:
                logger.error(f"Failed to upload {file.filename}: {e}")
                raise HTTPException(500, f"Failed to upload {file.filename}")
        
        expires_at = datetime.now() + timedelta(hours=SESSION_EXPIRE_HOURS)
        
        return UploadResponse(
            message=f"Successfully uploaded {len(files)} files",
            files=uploaded_files,
            s3_keys=s3_keys,
            user_id=user_id,
            expires_at=expires_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Upload error: {e}")
        raise HTTPException(500, "Upload failed due to internal error")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    payload: IngestRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """Process and ingest documents into vector store"""
    start_time = time.time()
    
    try:
        user_id = get_user_id(request)
        
        # Verify all S3 keys belong to this user
        for key in payload.s3_keys:
            if not key.startswith(f"{user_id}/"):
                raise HTTPException(403, f"Access denied to file: {key}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        all_chunks = []
        temp_files = []
        
        try:
            for key in payload.s3_keys:
                # Download file from S3
                local_path = f"/tmp/{uuid.uuid4()}.pdf"
                temp_files.append(local_path)
                
                services.s3.download_file(S3_BUCKET_NAME, key, local_path)
                
                # Load and process PDF
                loader = PyPDFLoader(local_path)
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata.update({
                        'source': key,
                        'user_id': user_id,
                        'ingested_at': datetime.now().isoformat()
                    })
                
                # Split into chunks
                chunks = splitter.split_documents(documents)
                for chunk in chunks:
                    chunk.metadata['user_id'] = user_id
                
                all_chunks.extend(chunks)
                logger.info(f"Processed {len(chunks)} chunks from {key}")
            
            # Store in Pinecone
            if all_chunks:
                vector_store = PineconeVectorStore(
                    index_name=PINECONE_INDEX,
                    embedding=services.embedding
                )
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    vector_store.add_documents,
                    all_chunks
                )
                logger.info(f"Added {len(all_chunks)} chunks to vector store")
            
        finally:
            # Schedule cleanup of temp files
            background_tasks.add_task(cleanup_temp_files, *temp_files)
        
        processing_time = time.time() - start_time
        
        return IngestResponse(
            message="Documents successfully ingested",
            chunks=len(all_chunks),
            user_id=user_id,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Ingestion error: {e}")
        raise HTTPException(500, "Document ingestion failed")

@app.post("/question", response_model=QuestionResponse)
async def ask_question(
    payload: QuestionRequest,
    request: Request
):
    """Ask questions based on ingested documents"""
    start_time = time.time()
    
    try:
        user_id = get_user_id(request)
        
        # Create vector store and retriever
        vector_store = PineconeVectorStore(
            index_name=PINECONE_INDEX,
            embedding=services.embedding
        )
        
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": payload.top_k,
                "filter": {"user_id": user_id}
            }
        )
        
        # Create RAG chain
        combine_docs_chain = create_stuff_documents_chain(
            services.llm,
            enhanced_prompt
        )
        
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        # Execute query
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            rag_chain.invoke,
            {"input": payload.question}
        )
        
        # Extract sources
        sources = []
        if payload.include_sources and 'context' in result:
            sources = list({
                doc.metadata.get('source', 'Unknown')
                for doc in result['context']
            })
        
        processing_time = time.time() - start_time
        
        return QuestionResponse(
            answer=result['answer'],
            sources=sources,
            user_id=user_id,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.exception(f"Question processing error: {e}")
        raise HTTPException(500, "Failed to process question")

@app.get("/session", response_model=SessionInfo)
async def get_session_info(request: Request):
    """Get current session information"""
    try:
        user_id = get_user_id(request)
        
        # Get document count from S3
        try:
            response = services.s3.list_objects_v2(
                Bucket=S3_BUCKET_NAME,
                Prefix=f"{user_id}/"
            )
            documents_count = len(response.get('Contents', []))
        except Exception:
            documents_count = 0
        
        # Get chunks count from Pinecone (approximation)
        chunks_count = documents_count * 10  # Rough estimate
        
        # Get session timing
        created_at = datetime.now() - timedelta(hours=1)  # Placeholder
        expires_at = created_at + timedelta(hours=SESSION_EXPIRE_HOURS)
        
        return SessionInfo(
            user_id=user_id,
            documents_count=documents_count,
            chunks_count=chunks_count,
            created_at=created_at,
            expires_at=expires_at
        )
        
    except Exception as e:
        logger.exception(f"Session info error: {e}")
        raise HTTPException(500, "Failed to retrieve session information")

@app.delete("/session")
async def delete_session(request: Request):
    """Delete current session and all associated data"""
    try:
        user_id = get_user_id(request)
        
        # Delete from S3
        try:
            response = services.s3.list_objects_v2(
                Bucket=S3_BUCKET_NAME,
                Prefix=f"{user_id}/"
            )
            
            if 'Contents' in response:
                delete_keys = [{'Key': obj['Key']} for obj in response['Contents']]
                services.s3.delete_objects(
                    Bucket=S3_BUCKET_NAME,
                    Delete={'Objects': delete_keys}
                )
                logger.info(f"Deleted {len(delete_keys)} files from S3")
        except Exception as e:
            logger.warning(f"S3 deletion failed: {e}")
        
        # Delete from Pinecone
        try:
            index = services.pinecone.Index(PINECONE_INDEX)
            index.delete(filter={"user_id": {"$eq": user_id}})
            logger.info(f"Deleted vectors for user {user_id}")
        except Exception as e:
            logger.warning(f"Pinecone deletion failed: {e}")
        
        # Clear Redis session if available
        if services.redis:
            try:
                services.redis.delete(f"session:{user_id}")
            except Exception as e:
                logger.warning(f"Redis cleanup failed: {e}")
        
        return {
            "message": f"Session {user_id} successfully deleted",
            "user_id": user_id
        }
        
    except Exception as e:
        logger.exception(f"Session deletion error: {e}")
        raise HTTPException(500, "Failed to delete session")

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Check S3
    try:
        services.s3.head_bucket(Bucket=S3_BUCKET_NAME)
        health_status["services"]["s3"] = "healthy"
    except Exception:
        health_status["services"]["s3"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check Pinecone
    try:
        services.pinecone.list_indexes()
        health_status["services"]["pinecone"] = "healthy"
    except Exception:
        health_status["services"]["pinecone"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check Redis
    if services.redis:
        try:
            services.redis.ping()
            health_status["services"]["redis"] = "healthy"
        except Exception:
            health_status["services"]["redis"] = "unhealthy"
    else:
        health_status["services"]["redis"] = "not_configured"
    
    return health_status

# ---- ERROR HANDLERS ----
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            details=f"HTTP {exc.status_code}"
        ).dict()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            details="An unexpected error occurred"
        ).dict()
    )

# ---- STARTUP MESSAGE ----
@app.on_event("startup")
async def startup_message():
    logger.info("🚀 Smart Document QA API is starting up...")
    logger.info(f"📚 Using model: {MODEL_NAME}")
    logger.info(f"🔗 Pinecone index: {PINECONE_INDEX}")
    logger.info(f"📁 S3 bucket: {S3_BUCKET_NAME}")