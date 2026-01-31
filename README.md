# 🎙️ Julie - Assistant Vocal Intelligent pour l'Assurance

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Powered%20by-Groq-orange)](https://groq.com/)

Assistant vocal IA avancé pour CNP Assurances (assurance vie française), combinant reconnaissance vocale de pointe, traitement du langage naturel et synthèse vocale pour offrir une expérience client exceptionnelle.

---

##  Table des Matières

- [ Caractéristiques Principales](#-caractéristiques-principales)
- [ Architecture](#️-architecture)
- [ Installation Rapide](#-installation-rapide)
- [ Configuration](#️-configuration)
- [ Utilisation](#-utilisation)
- [ Structure du Projet](#-structure-du-projet)
- [ Technologies Utilisées](#-technologies-utilisées)
- [ Tests](#-tests)
- [ Interface Web](#-interface-web)
- [ Documentation API](#-documentation-api)
- [ Contribution](#-contribution)
- [ Licence](#-licence)

---

##  Caractéristiques Principales

###  Fonctionnalités Intelligentes

- **Reconnaissance Vocale Avancée**
  - WebRTC VAD (Voice Activity Detection) pour une détection fiable de la parole
  - Groq Whisper STT (whisper-large-v3-turbo) pour une transcription précise en français
  - Gestion automatique du silence et du bruit ambiant

- **Traitement du Langage Naturel**
  - Groq LLM (llama-3.3-70b-versatile) pour des réponses contextuelles
  - Système de classification d'intentions (salutations, réclamations, transferts)
  - Historique de conversation avec mémoire contextuelle
  - Prompts optimisés pour le domaine de l'assurance

- **Synthèse Vocale de Qualité**
  - ElevenLabs TTS pour une voix naturelle et professionnelle (option premium)
  - gTTS comme solution de secours gratuite
  - Voix françaises optimisées (Charlotte, Thomas)

- **RAG (Retrieval-Augmented Generation)**
  - Base vectorielle Qdrant pour la recherche sémantique
  - LangChain pour l'orchestration des récupérations
  - Embeddings HuggingFace pour une compréhension contextuelle
  - Chargement automatique de documents d'assurance

- **Gestion des Sinistres**
  - Consultation du statut des sinistres en temps réel
  - Déclaration guidée de nouveaux sinistres
  - Base de données intégrée pour le suivi
  - Workflow conversationnel intelligent

- **Interface Web Moderne**
  - Panel d'administration complet
  - Simulation d'appel vocal push-to-talk
  - Upload et gestion de documents
  - Statistiques et analytics en temps réel

### Expérience Utilisateur

- **Conversation Naturelle** : Dialogue fluide et contextuel en français
- **Réponses Concises** : Optimisées pour l'interaction vocale (2-3 phrases)
- **Détection Intelligente** : Fin automatique basée sur les silences
- **Multi-Interface** : CLI, Web, et API REST

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    JULIE VOICE ASSISTANT                     │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌─────────┐     ┌──────────┐    ┌──────────┐
        │   CLI   │     │   WEB    │    │    API   │
        │Interface│     │Interface │    │  (REST)  │
        └─────────┘     └──────────┘    └──────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │         CORE ORCHESTRATION              │
        │  ┌──────────────────────────────────┐   │
        │  │    Intent Classification         │   │
        │  │  (greeting, claim, transfer...)  │   │
        │  └──────────────────────────────────┘   │
        └─────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│  AUDIO       │      │     NLU      │     │     DATA     │
├──────────────┤      ├──────────────┤     ├──────────────┤
│ WebRTC VAD   │─────▶│  Groq STT    │     │   Qdrant     │
│ Microphone   │      │  (Whisper)   │     │  Vector DB   │
│ Playback     │      └──────────────┘     │              │
└──────────────┘              │             │ LangChain    │
        │                     ▼             │ Retrieval    │
        │             ┌──────────────┐     │              │
        │             │   Groq LLM   │◀────┤ Claims DB    │
        │             │ (Llama 3.3)  │     │              │
        │             └──────────────┘     └──────────────┘
        │                     │
        ▼                     ▼
┌──────────────┐      ┌──────────────┐
│     TTS      │      │   Response   │
├──────────────┤      │  Generation  │
│ ElevenLabs   │◀─────┤              │
│    (or)      │      │  • Context   │
│    gTTS      │      │  • History   │
└──────────────┘      └──────────────┘
```

###  Pipeline de Traitement

1. **Capture Audio** : WebRTC VAD détecte la parole et enregistre
2. **Transcription** : Groq Whisper convertit l'audio en texte français
3. **Classification** : Identification de l'intention utilisateur
4. **Récupération** : RAG recherche les informations pertinentes dans Qdrant
5. **Génération** : Groq LLM produit une réponse contextuelle
6. **Synthèse** : ElevenLabs/gTTS convertit la réponse en audio
7. **Lecture** : L'audio est joué à l'utilisateur

---

##  Installation Rapide

### Prérequis

- **Python** : 3.11 ou supérieur
- **Système** : Linux, macOS, ou Windows (WSL recommandé)
- **Audio** : Microphone et haut-parleurs fonctionnels
- **API Keys** : Compte Groq (gratuit) - ElevenLabs optionnel

### Installation en 4 Étapes

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/julie-voice-assistant.git
cd julie-voice-assistant

# 2. Créer et activer l'environnement virtuel
python3.11 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
nano .env  # Ajouter vos clés API
```

### Vérification de l'Installation

```bash
# Test rapide
python main.py --version

# Lancer l'interface CLI
python main.py

# Lancer l'interface Web
python run_web.py
```

---

##  Configuration

### Variables d'Environnement

Créez un fichier `.env` à la racine du projet :

```bash
# ========================================
# GROQ API (OBLIGATOIRE)
# ========================================
# Obtenez votre clé gratuite sur https://console.groq.com
GROQ_API_KEY=gsk_your_groq_api_key_here

# ========================================
# ELEVENLABS API (OPTIONNEL)
# ========================================
# Pour une voix naturelle et professionnelle
# Obtenez votre clé sur https://elevenlabs.io
# Si non configurée, utilise gTTS (gratuit mais robotique)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# ========================================
# CONFIGURATION AUDIO (OPTIONNEL)
# ========================================
# Ajustez selon vos besoins
SAMPLE_RATE=16000
SILENCE_DURATION=1.5
MIN_SPEECH_DURATION=0.3
MAX_RECORD_DURATION=30.0

# ========================================
# CONFIGURATION RAG (OPTIONNEL)
# ========================================
# Qdrant Vector Database
QDRANT_PATH=./data/qdrant
QDRANT_COLLECTION=insurance_docs

# HuggingFace Embeddings
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# ========================================
# CONFIGURATION WEB (OPTIONNEL)
# ========================================
WEB_HOST=0.0.0.0
WEB_PORT=8000
DEBUG=False
```

### Obtention des Clés API

#### Groq API (Gratuit - Obligatoire)

1. Visitez [console.groq.com](https://console.groq.com)
2. Créez un compte gratuit
3. Naviguez vers **API Keys**
4. Cliquez sur **Create API Key**
5. Copiez la clé et ajoutez-la dans `.env`

**Modèles utilisés** :
- STT : `whisper-large-v3-turbo` (le plus économique)
- LLM : `llama-3.3-70b-versatile` (rapide et puissant)

#### ElevenLabs API (Optionnel - Recommandé)

1. Visitez [elevenlabs.io](https://elevenlabs.io)
2. Créez un compte (essai gratuit : 10,000 caractères/mois)
3. Allez dans **Profile** → **API Keys**
4. Copiez la clé et ajoutez-la dans `.env`

**Voix françaises disponibles** :
- `charlotte` : Voix féminine professionnelle (par défaut)
- `thomas` : Voix masculine professionnelle

---

##  Utilisation

### 1. Interface CLI (Ligne de Commande)

L'interface CLI offre une interaction vocale directe avec Julie.

```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Lancer Julie en mode CLI
python main.py

# Avec options
python main.py --interface cli
```

**Workflow CLI** :
1. Julie vous accueille vocalement
2. Parlez dans votre microphone (détection automatique)
3. Julie répond vocalement
4. Pour quitter, dites "au revoir" ou appuyez sur Ctrl+C

**Exemple de conversation** :
```
Julie: Bonjour, je suis Julie, votre assistante AssuranceVie. 
       Comment puis-je vous aider?

Vous: J'ai eu un accident de voiture hier

Julie: Je suis désolée d'apprendre cela. Pour déclarer votre 
       sinistre auto, appelez le 0 800 123 456, disponible 24h/24. 
       Vous aurez besoin du constat amiable et des photos.

Vous: Quels documents dois-je préparer?

Julie: Pour votre déclaration, préparez le constat amiable signé 
       par les deux parties, des photos des dégâts, et vos papiers 
       du véhicule. Le délai est de 5 jours ouvrés.
```

### 2. Interface Web

L'interface web offre deux espaces distincts :

```bash
# Lancer le serveur web
python run_web.py
```

Le serveur démarre sur `http://localhost:8000`

####  Page d'Accueil (`/`)

- Présentation de Julie
- Liens vers les différentes interfaces
- Informations sur les fonctionnalités

####  Interface Voix (`/voice`)

Interface de simulation d'appel avec Julie :

- **Push-to-Talk** : Maintenez le bouton pour parler
- **Historique** : Visualisation de la conversation
- **Réponses Vocales** : Lecture automatique des réponses
- **Interface Intuitive** : Design moderne et responsive

**Utilisation** :
1. Cliquez et maintenez le bouton 
2. Parlez clairement
3. Relâchez le bouton
4. Julie analyse et répond vocalement

####  Panel Admin (`/admin`)

Interface d'administration complète :

**Fonctionnalités** :
-  **Upload de Documents** : Ajoutez des PDF d'assurance à la base de connaissances
-  **Statistiques** : Nombre d'appels, taux de satisfaction, temps de réponse
-  **Gestion des Sinistres** : Visualisation et gestion des déclarations
-  **Utilisateurs** : Gestion des clients et historique
-  **Configuration** : Paramètres système et API

**Tableau de bord** :
```
┌─────────────────────────────────────────────────────┐
│   STATISTIQUES                                     │
├─────────────────────────────────────────────────────┤
│  🔹 Appels traités aujourd'hui : 247                 │
│  🔹 Temps de réponse moyen : 1.2s                    │
│  🔹 Taux de satisfaction : 94%                       │
│  🔹 Documents indexés : 156                          │
└─────────────────────────────────────────────────────┘
```

### 3. API REST

L'API REST permet l'intégration avec d'autres systèmes.

#### Documentation Interactive

- **Swagger UI** : `http://localhost:8000/docs`
- **ReDoc** : `http://localhost:8000/redoc`

#### Endpoints Principaux

```python
# POST /api/transcribe
# Transcrit un fichier audio en texte
curl -X POST "http://localhost:8000/api/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@recording.wav"

# Response:
{
  "text": "Je voudrais déclarer un sinistre",
  "language": "fr",
  "duration": 2.5
}

# POST /api/chat
# Envoie un message texte et reçoit une réponse
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Quels sont les horaires du service client?"}'

# Response:
{
  "response": "Nous sommes disponibles du lundi au vendredi de 8h à 20h, et le samedi de 9h à 17h.",
  "intent": "information_request",
  "confidence": 0.95
}

# POST /api/synthesize
# Convertit du texte en audio
curl -X POST "http://localhost:8000/api/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour, je suis Julie"}' \
  --output response.mp3

# GET /api/claims/{claim_id}
# Récupère le statut d'un sinistre
curl -X GET "http://localhost:8000/api/claims/SIN-2024-001"

# Response:
{
  "claim_id": "SIN-2024-001",
  "status": "in_progress",
  "type": "auto",
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_processing": "30 days"
}
```

---

##  Structure du Projet

```
julie-voice-assistant/
│
├──  README.md                 # Documentation principale (ce fichier)
├──  requirements.txt          # Dépendances Python
├──  .env.example              # Template de configuration
├──  .gitignore                # Fichiers à ignorer par Git
├──  pytest.ini                # Configuration des tests
├──  LICENSE                   # Licence MIT
│
├──  main.py                   # Point d'entrée CLI
├──  run_web.py                # Point d'entrée Web
│
├──  julie/                    # Package principal
│   ├── __init__.py              # Version et exports
│   │
│   ├──  core/                 # Orchestration centrale
│   │   ├── __init__.py
│   │   ├── agent.py             # Agent principal
│   │   ├── intents.py           # Classification d'intentions
│   │   ├── context.py           # Gestion du contexte
│   │   └── logging.py           # Système de logs
│   │
│   ├──  audio/                # Traitement audio
│   │   ├── __init__.py
│   │   ├── vad.py               # Voice Activity Detection (WebRTC)
│   │   ├── recorder.py          # Enregistrement audio
│   │   └── player.py            # Lecture audio
│   │
│   ├──  stt/                  # Speech-to-Text
│   │   ├── __init__.py
│   │   ├── groq_whisper.py      # Groq Whisper STT
│   │   └── base.py              # Interface abstraite
│   │
│   ├──  llm/                  # Large Language Model
│   │   ├── __init__.py
│   │   ├── groq_llm.py          # Groq LLM (Llama 3.3)
│   │   ├── prompts.py           # Templates de prompts
│   │   └── base.py              # Interface abstraite
│   │
│   ├──  tts/                  # Text-to-Speech
│   │   ├── __init__.py
│   │   ├── elevenlabs.py        # ElevenLabs TTS
│   │   ├── gtts.py              # Google TTS (fallback)
│   │   └── base.py              # Interface abstraite
│   │
│   ├──  rag/                  # Retrieval-Augmented Generation
│   │   ├── __init__.py
│   │   ├── vectorstore.py       # Qdrant vector database
│   │   ├── embeddings.py        # HuggingFace embeddings
│   │   ├── retriever.py         # LangChain retriever
│   │   └── document_loader.py   # Chargement de documents
│   │
│   ├──  claims/               # Gestion des sinistres
│   │   ├── __init__.py
│   │   ├── database.py          # Base de données des sinistres
│   │   ├── filing.py            # Processus de déclaration
│   │   └── status.py            # Consultation de statut
│   │
│   ├──  web/                  # Interface web
│   │   ├── __init__.py
│   │   ├── app.py               # Application FastAPI
│   │   ├── routes/              # Routes API
│   │   │   ├── api.py           # Endpoints REST
│   │   │   ├── admin.py         # Routes admin
│   │   │   └── voice.py         # Interface vocale
│   │   ├── static/              # Fichiers statiques
│   │   │   ├── css/
│   │   │   ├── js/
│   │   │   └── images/
│   │   └── templates/           # Templates HTML
│   │       ├── index.html
│   │       ├── admin.html
│   │       └── voice.html
│   │
│   └──  interfaces/           # Interfaces utilisateur
│       ├── __init__.py
│       ├── cli.py               # Interface ligne de commande
│       └── telephony.py         # Interface téléphonie (futur)
│
├──  tests/                    # Tests unitaires et d'intégration
│   ├── __init__.py
│   ├── test_stt.py              # Tests STT
│   ├── test_llm.py              # Tests LLM
│   ├── test_tts.py              # Tests TTS
│   ├── test_rag.py              # Tests RAG
│   ├── test_claims.py           # Tests sinistres
│   └── test_integration.py      # Tests d'intégration
│
├──  data/                     # Données et ressources
│   ├── qdrant/                  # Base vectorielle Qdrant
│   ├── claims.db                # SQLite pour les sinistres
│   ├── documents/               # Documents d'assurance (PDF)
│   └── logs/                    # Fichiers de logs
│
└──  docs/                     # Documentation supplémentaire
    ├── ARCHITECTURE.md          # Architecture détaillée
    ├── API.md                   # Documentation API
    ├── DEPLOYMENT.md            # Guide de déploiement
    └── CONTRIBUTING.md          # Guide de contribution
```

---

##  Technologies Utilisées

###  Audio & Voix

| Technologie | Version | Rôle | Raison |
|-------------|---------|------|--------|
| **WebRTC VAD** | 2.0.14 | Voice Activity Detection | Détection robuste et fiable de la parole |
| **sounddevice** | 0.5.5 | Capture audio | Interface simple pour PyAudio |
| **Groq Whisper** | API v1 | Speech-to-Text | Précision excellente en français, rapide |
| **ElevenLabs** | API v1 | Text-to-Speech Premium | Voix naturelles et professionnelles |
| **gTTS** | 2.5.4 | Text-to-Speech Fallback | Solution gratuite et simple |

### Intelligence Artificielle

| Technologie | Version | Rôle | Raison |
|-------------|---------|------|--------|
| **Groq** | 1.0.0 | Plateforme LLM | Inférence ultra-rapide (500+ tokens/s) |
| **Llama 3.3 70B** | - | Modèle de langage | Excellent en français, contexte large |
| **LangChain** | 1.2.7 | Orchestration LLM | Framework robuste pour RAG |
| **Qdrant** | 1.16.2 | Base vectorielle | Recherche sémantique performante |
| **HuggingFace** | 0.36.0 | Embeddings | Modèles multilingues de qualité |
| **sentence-transformers** | 5.2.2 | Embeddings sémantiques | Support français optimisé |

###  Web & API

| Technologie | Version | Rôle | Raison |
|-------------|---------|------|--------|
| **FastAPI** | 0.128.0 | Framework Web | Moderne, rapide, async natif |
| **Uvicorn** | 0.40.0 | Serveur ASGI | Performance excellente |
| **Starlette** | 0.50.0 | Web toolkit | Base solide pour FastAPI |
| **Jinja2** | 3.1.6 | Templates | Rendu HTML dynamique |
| **python-multipart** | 0.0.22 | Upload fichiers | Gestion des formulaires |

###  Tests & Qualité

| Technologie | Version | Rôle | Raison |
|-------------|---------|------|--------|
| **pytest** | 9.0.2 | Framework de tests | Standard Python moderne |
| **python-dotenv** | 1.2.1 | Variables d'env | Gestion sécurisée des secrets |

###  Data & Processing

| Technologie | Version | Rôle | Raison |
|-------------|---------|------|--------|
| **NumPy** | 1.26.4 | Calcul numérique | Traitement audio efficace |
| **PyMuPDF** | 1.26.7 | Lecture PDF | Extraction de texte rapide |
| **scikit-learn** | 1.8.0 | ML traditionnel | Prétraitement et features |

---

##  Tests

### Exécution des Tests

```bash
# Tous les tests
pytest -v

# Tests spécifiques
pytest tests/test_stt.py -v
pytest tests/test_llm.py -v
pytest tests/test_rag.py -v

# Avec coverage
pytest --cov=julie --cov-report=html

# Tests lents exclus
pytest -m "not slow"

# Tests d'intégration seulement
pytest -m integration
```

### Structure des Tests

```python
# tests/test_stt.py
def test_stt_transcription():
    """Test de transcription audio."""
    stt = STT()
    audio = load_test_audio("sample.wav")
    text = stt.transcribe(audio)
    assert "bonjour" in text.lower()

def test_stt_error_handling():
    """Test de gestion d'erreurs STT."""
    stt = STT()
    result = stt.transcribe(b"invalid")
    assert result == ""

# tests/test_llm.py
def test_llm_response():
    """Test de génération de réponse."""
    llm = LLM()
    response = llm.respond("Quels sont vos horaires?")
    assert "lundi" in response.lower()

# tests/test_rag.py
def test_vectorstore_search():
    """Test de recherche vectorielle."""
    vs = VectorStore()
    results = vs.search("sinistre auto")
    assert len(results) > 0
```

### Couverture de Tests

Objectif : **>80% de couverture**

```
Module          Statements   Missing   Coverage
------------------------------------------------
julie/stt           45          3         93%
julie/llm           67          8         88%
julie/tts           52          6         88%
julie/rag           89         12         87%
julie/claims        56          9         84%
------------------------------------------------
TOTAL              309         38         88%
```

---

##  Interface Web

### Architecture Web

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (HTML/JS)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │     Home     │  │    Voice     │  │    Admin     │  │
│  │   Landing    │  │  Push-to-    │  │  Dashboard   │  │
│  │     Page     │  │     Talk     │  │   & Mgmt     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
                           │ HTTP/WebSocket
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI)                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │              API Routes                          │   │
│  │  • /api/transcribe  (STT)                        │   │
│  │  • /api/chat        (LLM)                        │   │
│  │  • /api/synthesize  (TTS)                        │   │
│  │  • /api/claims      (Sinistres)                  │   │
│  │  • /api/documents   (RAG upload)                 │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │           WebSocket Handlers                     │   │
│  │  • Real-time voice streaming                     │   │
│  │  • Live transcription updates                    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │ Julie Core   │
                   │   Engine     │
                   └──────────────┘
```

### Pages Disponibles

####  Page d'Accueil (`/`)

**Fonctionnalités** :
- Présentation de Julie et de ses capacités
- Cartes de navigation vers Voice et Admin
- Informations sur le projet
- Liens vers la documentation

**Technologies** :
- HTML5 sémantique
- CSS3 avec animations
- JavaScript vanilla pour interactions

####  Interface Voice (`/voice`)

**Fonctionnalités** :
- Interface push-to-talk intuitive
- Visualisation audio en temps réel
- Historique de conversation scrollable
- Lecture automatique des réponses
- Indicateurs de statut (listening, processing, speaking)

**Composants** :
```javascript
// Gestion du bouton push-to-talk
let isRecording = false;
let mediaRecorder;

pushToTalkBtn.addEventListener('mousedown', startRecording);
pushToTalkBtn.addEventListener('mouseup', stopRecording);

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({audio: true});
  mediaRecorder = new MediaRecorder(stream);
  // ...
}
```

####  Panel Admin (`/admin`)

**Sections** :
1. **Dashboard** : Statistiques et KPIs
2. **Documents** : Upload et gestion de PDFs
3. **Sinistres** : Liste et détails des déclarations
4. **Configuration** : Paramètres système

**APIs Utilisées** :
```javascript
// Upload de document
const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/api/documents/upload', {
    method: 'POST',
    body: formData
  });
  
  return response.json();
};

// Récupération des statistiques
const getStats = async () => {
  const response = await fetch('/api/admin/stats');
  return response.json();
};
```

---

##  Documentation API

### Authentication

Actuellement, l'API est ouverte pour le développement. En production, utilisez JWT :

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/api/protected")
async def protected_route(token: str = Depends(security)):
    # Validate token
    pass
```

### Endpoints Complets

#### POST `/api/transcribe`

Transcrit un fichier audio en texte.

**Request** :
```http
POST /api/transcribe
Content-Type: multipart/form-data

audio: <fichier WAV/MP3>
language: fr (optionnel)
```

**Response** :
```json
{
  "text": "Bonjour, je souhaite déclarer un sinistre",
  "language": "fr",
  "duration": 3.2,
  "confidence": 0.95
}
```

#### POST `/api/chat`

Envoie un message et reçoit une réponse.

**Request** :
```http
POST /api/chat
Content-Type: application/json

{
  "message": "Quels sont vos horaires?",
  "session_id": "user-123" (optionnel),
  "context": {} (optionnel)
}
```

**Response** :
```json
{
  "response": "Nous sommes disponibles du lundi au vendredi...",
  "intent": "information_request",
  "confidence": 0.92,
  "sources": ["faq.pdf", "horaires.pdf"],
  "session_id": "user-123"
}
```

#### POST `/api/synthesize`

Convertit du texte en audio.

**Request** :
```http
POST /api/synthesize
Content-Type: application/json

{
  "text": "Bonjour, je suis Julie",
  "voice": "charlotte" (optionnel),
  "speed": 1.0 (optionnel)
}
```

**Response** :
```
Content-Type: audio/mpeg
<binary MP3 data>
```

#### GET `/api/claims`

Liste tous les sinistres.

**Response** :
```json
{
  "claims": [
    {
      "id": "SIN-2024-001",
      "type": "auto",
      "status": "in_progress",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1
}
```

#### GET `/api/claims/{claim_id}`

Récupère un sinistre spécifique.

**Response** :
```json
{
  "id": "SIN-2024-001",
  "type": "auto",
  "status": "in_progress",
  "description": "Collision arrière",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-16T14:20:00Z",
  "documents": ["constat.pdf", "photos.zip"]
}
```

---

##  Déploiement

### Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run
CMD ["python", "run_web.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  julie:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant:/qdrant/storage
    restart: unless-stopped
```

**Démarrage** :
```bash
docker-compose up -d
```

### Production (Cloud)

#### Sur AWS EC2

```bash
# 1. Créer une instance EC2 (Ubuntu 22.04, t3.medium)
# 2. Configurer le security group (ports 8000, 22)
# 3. Se connecter via SSH

ssh -i key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# 4. Installer les dépendances
sudo apt update
sudo apt install -y python3.11 python3.11-venv portaudio19-dev

# 5. Cloner et configurer
git clone https://github.com/votre-repo/julie.git
cd julie
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 6. Configuration
cp .env.example .env
nano .env  # Ajouter les clés

# 7. Lancer avec systemd
sudo nano /etc/systemd/system/julie.service
```

```ini
[Unit]
Description=Julie Voice Assistant
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/julie
Environment="PATH=/home/ubuntu/julie/venv/bin"
ExecStart=/home/ubuntu/julie/venv/bin/python run_web.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Activer et démarrer
sudo systemctl enable julie
sudo systemctl start julie
sudo systemctl status julie
```

---

##  Sécurité

### Best Practices

1. **Ne jamais committer les clés API** : Utilisez `.env` et `.gitignore`
2. **HTTPS en production** : Configurez un certificat SSL
3. **Rate limiting** : Limitez les requêtes API
4. **Validation des entrées** : Sanitize tous les inputs utilisateur
5. **Authentification** : Ajoutez JWT pour l'API en production

### Configuration SSL (Nginx)

```nginx
server {
    listen 443 ssl;
    server_name julie.example.com;

    ssl_certificate /etc/letsencrypt/live/julie.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/julie.example.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

##  Contribution

Les contributions sont les bienvenues ! Voici comment contribuer :

### Process

1. **Fork** le projet
2. **Créer** une branche feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** vos changements (`git commit -m 'Add AmazingFeature'`)
4. **Push** vers la branche (`git push origin feature/AmazingFeature`)
5. **Ouvrir** une Pull Request

### Guidelines

- Suivez PEP 8 pour le code Python
- Ajoutez des tests pour les nouvelles fonctionnalités
- Documentez les fonctions avec des docstrings
- Mettez à jour le README si nécessaire


##  Problèmes Connus

### macOS
- **Problème** : Erreur SSL avec ElevenLabs
- **Solution** : `pip install --upgrade certifi`

### Linux
- **Problème** : Permission denied sur audio
- **Solution** : `sudo usermod -a -G audio $USER` puis redémarrer

### Windows
- **Problème** : WebRTC VAD ne s'installe pas
- **Solution** : Utiliser WSL2 ou installer Visual C++ Build Tools

---

##  Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

```
MIT License

Copyright (c) 2024 Julie Voice Assistant Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


##  Remerciements

Merci aux projets et technologies qui ont rendu Julie possible :

- **Groq** pour l'infrastructure LLM ultra-rapide
- **ElevenLabs** pour les voix naturelles de qualité
- **Anthropic** pour l'inspiration sur les assistants conversationnels
- **La communauté open-source** pour tous les outils utilisés

---


<div align="center">

**Fait avec ❤️ par l'équipe Julie**

[ Donnez une étoile](https://github.com/votre-repo/julie) | [🐛 Signaler un bug](https://github.com/votre-repo/julie/issues) | [ Demander une fonctionnalité](https://github.com/votre-repo/julie/issues/new)

</div>