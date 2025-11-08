# Greek Curriculum Ontology Extractor

**Αυτόματη εξαγωγή οντολογιών από Ελληνικά Αναλυτικά Προγράμματα με Τεχνητή Νοημοσύνη και Σημασιολογική Ανάλυση**

Advanced AI-powered system for extracting structured ontologies from Greek curriculum documents with multiple LLM providers, RAG enhancement, contradiction detection, and MITOS annotation support.

---

## 🎯 Features

### **6 Extraction Modes**

| Mode | Description | Use Case |
|------|-------------|----------|
| **1. LLM Only** | Pure LLM extraction | Fast baseline |
| **2. LLM + Knowledge Enhancement** | With CEDS ontology | Structured output |
| **3. LLM + RAG** | With retrieval augmentation | Context-rich |
| **4. LLM + MITOS** | With legal text annotation | Legal compliance |
| **5. Focused Ontology** | Targeted extraction | Precision mode |
| **6. Full Pipeline** | All features enabled | Maximum quality |

### **Multi-LLM Support**
- **OpenAI**: GPT-4o, GPT-4-turbo, GPT-3.5
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **Google**: Gemini Pro, Gemini 1.5 Flash
- Automatic fallback on failures

### **Advanced Features**
- 📚 **RAG (Retrieval-Augmented Generation)**: FAISS vectorstore για εμπλουτισμό context
- 🎓 **CEDS Integration**: Common Education Data Standards alignment
- ⚖️ **MITOS Annotation**: Legal text alignment με νομικό πλαίσιο
- 🔍 **Contradiction Detection**: Αυτόματη ανίχνευση αντιφάσεων
- 📊 **Ontology Analysis**: Comprehensive statistics και validation
- 🌐 **RDF/TTL Output**: Semantic Web standard formats

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/StergiosCha/greek-curriculum-ontology.git
cd greek-curriculum-ontology

# Install dependencies
pip install -r requirements.txt

# Set up API keys
echo "OPENAI_API_KEY=your_key" > .env
echo "ANTHROPIC_API_KEY=your_key" >> .env
echo "GOOGLE_API_KEY=your_key" >> .env

# Run the application
python -m app.main
```

### Usage

**Web Interface:**
```bash
# Start server
uvicorn app.main:app --reload --port 8000

# Open browser
open http://localhost:8000
```

**API Usage:**
```python
from app.services.enhanced_curriculum_extractor import EnhancedCurriculumExtractor
from app.core.config import ExtractionMode
from app.services.llm_service import LLMProvider

# Initialize extractor
extractor = EnhancedCurriculumExtractor(
    source_type="pdf",
    source_path="data/curricula/ΠΣ_Γραμματική.pdf",
    extraction_mode=ExtractionMode.RAG,
    llm_provider=LLMProvider.OPENAI,
    llm_model="gpt-4o"
)

# Extract ontology
ontology = extractor.extract()

# Save output
extractor.save_output("output.ttl")
```

---

## 📁 Project Structure

```
greek-curriculum-ontology7/
├── app/
│   ├── main.py                                # FastAPI application
│   ├── api/
│   │   ├── routes.py                          # API endpoints
│   │   └── models.py                          # Request/response models
│   ├── core/
│   │   ├── config.py                          # Configuration
│   │   └── extraction_modes.py                # Mode definitions
│   ├── services/
│   │   ├── enhanced_curriculum_extractor.py   # Main extractor
│   │   ├── llm_service.py                     # Multi-LLM interface
│   │   ├── rag_service.py                     # RAG implementation
│   │   ├── knowledge_enhancer.py              # CEDS integration
│   │   ├── mitos_annotator.py                 # Legal annotation
│   │   ├── contradiction_detector.py          # Contradiction detection
│   │   ├── focused_ontology.py                # Targeted extraction
│   │   └── complete_mitos_pipeline.py         # Full pipeline
│   ├── utils/
│   │   ├── file_handler.py                    # PDF/text processing
│   │   ├── ontology_analyzer.py               # Ontology analysis
│   │   └── text_processing.py                 # Text utilities
│   └── static/
│       └── frontend/                          # Web interface
├── data/
│   ├── curricula/                             # Input curricula
│   ├── outputs/                               # Generated ontologies
│   ├── cache/                                 # RAG embeddings
│   └── ceds_cache/                            # CEDS ontology
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration

### Extraction Modes

```python
from app.core.config import ExtractionMode

# Mode 1: Fast extraction
ExtractionMode.LLM_ONLY

# Mode 2: With CEDS standards
ExtractionMode.KNOWLEDGE_ENHANCED

# Mode 3: With RAG (best quality)
ExtractionMode.RAG

# Mode 4: With legal compliance
ExtractionMode.MITOS

# Mode 5: Focused extraction
ExtractionMode.FOCUSED

# Mode 6: Full pipeline (all features)
ExtractionMode.FULL_PIPELINE
```

### LLM Providers

```python
from app.services.llm_service import LLMProvider

# OpenAI
LLMProvider.OPENAI, model="gpt-4o"

# Anthropic
LLMProvider.ANTHROPIC, model="claude-3-5-sonnet-20241022"

# Google
LLMProvider.GOOGLE, model="gemini-1.5-pro"
```

---

## 📊 Output Format

Ontologies are generated in RDF Turtle format following curriculum ontology standards:

```turtle
@prefix currkg: <http://curriculum-kg.org/ontology/> .
@prefix proto-okn: <http://proto-okn.net/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

currkg:LearningObjective_1 a currkg:LearningObjective ;
    rdfs:label "Αναγνώριση γραμματικών κατηγοριών"@el ;
    currkg:hasEducationLevel currkg:PrimaryEducation ;
    currkg:hasSubjectArea currkg:GreekLanguage ;
    currkg:hasPrerequisite currkg:LearningObjective_0 ;
    currkg:alignsWithStandard ceds:Standard_123 .
```

---

## 🎓 Use Cases

1. **Curriculum Analysis**: Αυτόματη εξαγωγή μαθησιακών στόχων
2. **Standards Alignment**: Ευθυγράμμιση με CEDS και διεθνή πρότυπα  
3. **Legal Compliance**: Έλεγχος συμβατότητας με νομικό πλαίσιο (MITOS)
4. **Contradiction Detection**: Εύρεση ασυνεπειών στα Αναλυτικά Προγράμματα
5. **Knowledge Graph Construction**: Δημιουργία curriculum knowledge graphs
6. **Comparative Analysis**: Σύγκριση μεταξύ διαφορετικών ΑΠ

---

## 🔬 Research Applications

- **Digital Humanities**: Ανάλυση εκπαιδευτικών προγραμμάτων
- **Educational Policy**: Evidence-based policy making
- **Curriculum Design**: Βελτιστοποίηση μαθησιακών στόχων
- **Standards Development**: Alignment με διεθνή πρότυπα

---

## 🛠️ Dependencies

```
fastapi
uvicorn
rdflib
langchain
langchain-openai
langchain-anthropic
langchain-google-genai
pypdf2
faiss-cpu
sentence-transformers
```

---

## 📝 API Endpoints

### Extraction
- `POST /api/extract` - Extract ontology from curriculum
- `GET /api/results/{task_id}` - Get extraction results
- `GET /api/download/{task_id}` - Download TTL file

### Analysis
- `POST /api/analyze` - Analyze ontology quality
- `POST /api/detect-contradictions` - Find contradictions
- `GET /api/statistics/{ontology_id}` - Get ontology stats

### Health
- `GET /health` - Health check endpoint

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional LLM providers
- More extraction modes
- Enhanced CEDS alignment
- Improved contradiction detection

---

## 📧 Contact

**Stergios Chatzikyriakidis**  
Email: stergios.chatzikyriakidis@uoc.gr  
University of Crete

For questions, issues, or collaboration inquiries, please contact via email or open an issue on GitHub.

---

## 📝 License

MIT License

Copyright (c) 2025 Stergios Chatzikyriakidis

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

**Built for Greek Educational Standards Analysis and Ontology Engineering**

