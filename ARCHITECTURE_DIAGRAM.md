# Biểu Đồ Kiến Trúc Hệ Thống KGChat-03

## 1. Kiến Trúc Tổng Quan

```mermaid
graph TB
    subgraph "Frontend"
        UI[React Frontend]
        API_Call[API Calls]
    end
    
    subgraph "Backend API"
        FastAPI[FastAPI Server<br/>main.py]
        RM[RetrievalManager<br/>baseline.py]
    end
    
    subgraph "Query Processing Pipeline"
        QP[KGQueryProcessor<br/>kg_query_processor.py]
        QA[QueryAnalyzer<br/>query_analyzer.py]
        KE[KeywordExtractor<br/>keyword_extractor.py]
        TLR[TripleLevelRetrieval<br/>triple_level_retrieval.py]
    end
    
    subgraph "LLM Services"
        GC[GeminiClient]
        TE[TransformerEncoder]
        PP[PipelinePrompts<br/>pipeline_prompts.py]
    end
    
    subgraph "Databases"
        Neo4j[(Neo4j<br/>Graph Database)]
        Qdrant[(Qdrant<br/>Vector Database)]
    end
    
    subgraph "Data Processing"
        GE[GraphExtractor]
        NE[NodeExtractor]
        EE[EdgeExtractor]
    end
    
    UI --> API_Call
    API_Call --> FastAPI
    FastAPI --> RM
    RM --> QP
    QP --> QA
    QP --> KE
    QP --> TLR
    QA --> GC
    KE --> GC
    TLR --> TE
    TLR --> Neo4j
    TLR --> Qdrant
    QP --> GC
    QP --> PP
    GE --> Neo4j
    GE --> Qdrant
    NE --> GE
    EE --> GE
```

## 2. Luồng Xử Lý Query Chi Tiết

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant FastAPI
    participant KGQueryProcessor
    participant QueryAnalyzer
    participant KeywordExtractor
    participant TripleLevelRetrieval
    participant GeminiClient
    participant TransformerEncoder
    participant Neo4j
    participant Qdrant
    
    User->>Frontend: Nhập câu hỏi
    Frontend->>FastAPI: POST /api/chat
    FastAPI->>KGQueryProcessor: process_query_full_pipeline()
    
    Note over KGQueryProcessor: BƯỚC 1: Phân loại Intent
    KGQueryProcessor->>QueryAnalyzer: analyze_query()
    QueryAnalyzer->>GeminiClient: Phân loại intent
    GeminiClient-->>QueryAnalyzer: QueryIntent
    QueryAnalyzer-->>KGQueryProcessor: HEALTHCARE_RELATED/GENERAL/GREETING
    
    alt Intent = HEALTHCARE_RELATED
        Note over KGQueryProcessor: BƯỚC 2: Trích xuất Keywords
        KGQueryProcessor->>KeywordExtractor: extract_keywords()
        KeywordExtractor->>GeminiClient: Trích xuất keywords
        GeminiClient-->>KeywordExtractor: High-level & Low-level keywords
        KeywordExtractor-->>KGQueryProcessor: Keywords list
        
        Note over KGQueryProcessor: BƯỚC 3: Truy vấn Knowledge Graph
        KGQueryProcessor->>TripleLevelRetrieval: retrieve_from_knowledge_graph()
        
        Note over TripleLevelRetrieval: Level 1: Vector Search
        TripleLevelRetrieval->>TransformerEncoder: Tạo embeddings cho keywords
        TransformerEncoder-->>TripleLevelRetrieval: Embeddings
        TripleLevelRetrieval->>Qdrant: Vector similarity search
        Qdrant-->>TripleLevelRetrieval: Level 1 nodes
        
        Note over TripleLevelRetrieval: Level 2: Graph Traversal
        TripleLevelRetrieval->>Neo4j: Traverse relationships
        Neo4j-->>TripleLevelRetrieval: Level 2 nodes & relationships
        
        TripleLevelRetrieval->>TripleLevelRetrieval: format_retrieval_results()
        TripleLevelRetrieval-->>KGQueryProcessor: Formatted context
        
        Note over KGQueryProcessor: BƯỚC 4: Tạo RAG Prompt
        KGQueryProcessor->>TripleLevelRetrieval: create_rag_prompt()
        TripleLevelRetrieval-->>KGQueryProcessor: RAG prompt
        
        Note over KGQueryProcessor: BƯỚC 5: Generate Answer
        KGQueryProcessor->>GeminiClient: generate_response_from_rag_prompt()
        GeminiClient-->>KGQueryProcessor: Final answer
        
        KGQueryProcessor->>TripleLevelRetrieval: save_retrieval_result()
        KGQueryProcessor->>KGQueryProcessor: save_llm_answer()
        
    else Intent = GENERAL
        KGQueryProcessor->>GeminiClient: Generate general response
        GeminiClient-->>KGQueryProcessor: General answer
        
    else Intent = GREETING
        KGQueryProcessor->>GeminiClient: Generate greeting response
        GeminiClient-->>KGQueryProcessor: Greeting message
    end
    
    KGQueryProcessor-->>FastAPI: Result dict
    FastAPI-->>Frontend: Streaming response
    Frontend-->>User: Hiển thị kết quả
```

## 3. Cấu Trúc Module Backend

```mermaid
graph LR
    subgraph "backend/"
        subgraph "retrieval/"
            QP[kg_query_processor.py<br/>Main Pipeline Orchestrator]
            QA[query_analyzer.py<br/>Intent Classification]
            KE[keyword_extractor.py<br/>Keyword Extraction]
            TLR[triple_level_retrieval.py<br/>Graph Retrieval]
            CA[context_assembler.py<br/>Context Assembly]
        end
        
        subgraph "llm/"
            subgraph "providers/gemini/"
                GC[gemini_client.py<br/>Gemini API Client]
                GConfig[gemini_config.py<br/>Configuration]
            end
            subgraph "providers/ollama/"
                OC[ollama_client.py]
            end
            subgraph "providers/openai/"
                OAC[openai_client.py]
            end
            BF[base_factory.py<br/>LLM Factory]
        end
        
        subgraph "db/"
            NC[neo4j_client.py<br/>Neo4j Client]
            VC[vector_db.py<br/>Vector DB Client]
        end
        
        subgraph "encoders/"
            TE[transformer_encoder.py<br/>Transformer Embeddings]
        end
        
        subgraph "graph_extractor/"
            GE[graph_extract.py<br/>Graph Extraction]
            NE[node_extractor.py<br/>Node Extraction]
            EE[edge_extractor.py<br/>Edge Extraction]
        end
        
        PP[pipeline_prompts.py<br/>Centralized Prompts]
    end
    
    QP --> QA
    QP --> KE
    QP --> TLR
    QP --> GC
    QP --> PP
    QA --> GC
    KE --> GC
    TLR --> NC
    TLR --> VC
    TLR --> TE
    TLR --> PP
    GE --> NE
    GE --> EE
    GE --> NC
    GE --> VC
```

## 4. Data Flow - Knowledge Graph Retrieval

```mermaid
flowchart TD
    Start[User Query] --> Intent{Query Intent?}
    
    Intent -->|HEALTHCARE_RELATED| Extract[Extract Keywords]
    Intent -->|GENERAL| General[Generate General Response]
    Intent -->|GREETING| Greeting[Generate Greeting]
    Intent -->|PERSONAL_INFO| Personal[Extract Personal Info]
    
    Extract --> HighLevel[High-level Keywords]
    Extract --> LowLevel[Low-level Keywords]
    
    HighLevel --> Embed1[Create Embeddings]
    LowLevel --> Embed2[Create Embeddings]
    
    Embed1 --> QdrantSearch1[Qdrant Vector Search<br/>Collection: kg_lv1_nodes]
    Embed2 --> QdrantSearch2[Qdrant Vector Search<br/>Collection: kg_lv1_nodes]
    
    QdrantSearch1 --> Level1Nodes1[Level 1 Nodes]
    QdrantSearch2 --> Level1Nodes2[Level 1 Nodes]
    
    Level1Nodes1 --> Neo4jTraverse[Neo4j Graph Traversal<br/>Find Level 2 Nodes]
    Level1Nodes2 --> Neo4jTraverse
    
    Neo4jTraverse --> Level2Nodes[Level 2 Nodes]
    Neo4jTraverse --> Relationships[Relationships]
    
    Level1Nodes1 --> Format[Format Retrieval Results]
    Level1Nodes2 --> Format
    Level2Nodes --> Format
    Relationships --> Format
    
    Format --> RAGPrompt[Create RAG Prompt]
    RAGPrompt --> LLM[Generate Answer with Gemini]
    LLM --> SavePrompt[Save Prompt File]
    LLM --> SaveAnswer[Save Answer File]
    SavePrompt --> End[Return Result]
    SaveAnswer --> End
    
    General --> End
    Greeting --> End
    Personal --> End
```

## 5. Component Dependencies

```mermaid
graph TD
    subgraph "Core Components"
        QP[KGQueryProcessor]
        PP[PipelinePrompts]
    end
    
    subgraph "Analysis Components"
        QA[QueryAnalyzer]
        KE[KeywordExtractor]
    end
    
    subgraph "Retrieval Components"
        TLR[TripleLevelRetrieval]
        CA[ContextAssembler]
    end
    
    subgraph "LLM Clients"
        GC[GeminiClient]
        OC[OllamaClient]
        OAC[OpenAIClient]
    end
    
    subgraph "Database Clients"
        NC[Neo4jClient]
        VC[VectorDBClient]
    end
    
    subgraph "Encoders"
        TE[TransformerEncoder]
    end
    
    QP --> QA
    QP --> KE
    QP --> TLR
    QP --> GC
    QP --> PP
    
    QA --> GC
    KE --> GC
    TLR --> NC
    TLR --> VC
    TLR --> TE
    TLR --> PP
    
    CA --> TLR
    CA --> NC
    
    style QP fill:#ff9999
    style PP fill:#99ff99
    style GC fill:#9999ff
    style NC fill:#ffff99
    style VC fill:#ffff99
```

## 6. Test Flow

```mermaid
flowchart TD
    Start[test_full_encoder.py<br/>main function] --> Init[initialize_clients]
    
    Init --> Gemini[Initialize GeminiClient]
    Init --> Transformer[Initialize TransformerEncoder]
    Init --> Qdrant[Initialize QdrantClient]
    Init --> Neo4j[Initialize Neo4jClient]
    
    Gemini --> Check1{All clients OK?}
    Transformer --> Check1
    Qdrant --> Check1
    Neo4j --> Check1
    
    Check1 -->|No| Error[Exit with Error]
    Check1 -->|Yes| Loop[For each test query]
    
    Loop --> Process[process_query_full_pipeline]
    
    Process --> Step1[Step 1: Intent Classification]
    Step1 --> Step2[Step 2: Keyword Extraction]
    Step2 --> Step3[Step 3: Graph Retrieval]
    Step3 --> Step4[Step 4: RAG Prompt Creation]
    Step4 --> Step5[Step 5: LLM Answer Generation]
    
    Step5 --> SaveFiles[Save Prompt & Answer Files]
    SaveFiles --> Summary[Print Summary]
    
    Summary --> NextQuery{More queries?}
    NextQuery -->|Yes| Loop
    NextQuery -->|No| Close[Close Connections]
    Close --> End[End]
```

## 7. File Structure Overview

```
KGChat-03/
├── main.py                          # FastAPI entry point
├── backend/
│   ├── retrieval/
│   │   ├── kg_query_processor.py    # Main pipeline orchestrator
│   │   ├── query_analyzer.py       # Intent classification
│   │   ├── keyword_extractor.py    # Keyword extraction
│   │   └── triple_level_retrieval.py # Graph retrieval
│   ├── llm/
│   │   └── providers/gemini/        # Gemini LLM client
│   ├── db/
│   │   ├── neo4j_client.py          # Neo4j graph database
│   │   └── vector_db.py             # Qdrant vector database
│   ├── encoders/
│   │   └── transformer_encoder.py  # Transformer embeddings
│   ├── graph_extractor/             # Graph extraction tools
│   └── pipeline_prompts.py          # Centralized prompts
├── test_full/
│   └── test_full_encoder.py         # Full pipeline test
└── frontend/                        # React frontend
```

## 8. Key Functions và Responsibilities

### kg_query_processor.py
- `process_query_full_pipeline()`: Main entry point cho toàn bộ pipeline
- `process_kg_query()`: Xử lý query với knowledge graph
- `generate_response_from_rag_prompt()`: Generate LLM response từ RAG prompt
- `_generate_healthcare_response_with_rag()`: Healthcare response generation
- `_generate_general_response_with_rag()`: General response generation
- `_generate_greeting_response()`: Greeting response generation
- `save_llm_answer()`: Lưu LLM answer vào file

### triple_level_retrieval.py
- `retrieve_from_knowledge_graph()`: Retrieve từ knowledge graph
- `format_retrieval_results()`: Format kết quả retrieval
- `create_rag_prompt()`: Tạo RAG prompt từ context
- `save_retrieval_result()`: Lưu retrieval result vào file

### query_analyzer.py
- `analyze_query()`: Phân loại intent của query
- `QueryIntent`: Enum định nghĩa các loại intent

### keyword_extractor.py
- `extract_keywords()`: Trích xuất high-level và low-level keywords

## 9. Database Schema

```mermaid
erDiagram
    Level1 ||--o{ Level2 : "has"
    Level1 ||--o{ Relationship : "connects"
    Level2 ||--o{ Relationship : "connects"
    
    Level1 {
        string id
        string name
        string description
        vector embedding
    }
    
    Level2 {
        string id
        string name
        string description
        string level1_id
    }
    
    Relationship {
        string id
        string source_id
        string target_id
        string type
        string description
    }
```

## 10. Configuration và Environment

```mermaid
graph LR
    subgraph "Environment Variables"
        GEMINI_API_KEY[GEMINI_API_KEY]
        NEO4J_URI[NEO4J_URI]
        NEO4J_USER[NEO4J_USER_NAME]
        NEO4J_PASS[NEO4J_PASSWORD]
        QDRANT_HOST[QDRANT_HOST]
        QDRANT_PORT[QDRANT_PORT]
    end
    
    subgraph "Config Files"
        CONFIG[configs.yml]
        PROMPTS[pipeline_prompts.py]
    end
    
    GEMINI_API_KEY --> GeminiClient
    NEO4J_URI --> Neo4jClient
    NEO4J_USER --> Neo4jClient
    NEO4J_PASS --> Neo4jClient
    QDRANT_HOST --> QdrantClient
    QDRANT_PORT --> QdrantClient
    CONFIG --> RetrievalManager
    PROMPTS --> KGQueryProcessor
```

---

**Ghi chú:**
- Tất cả các biểu đồ được tạo bằng Mermaid syntax
- Có thể render bằng các công cụ như Mermaid Live Editor, GitHub, hoặc VS Code extensions
- Biểu đồ mô tả kiến trúc và luồng xử lý của hệ thống KGChat-03

