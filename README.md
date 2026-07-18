# Delivery_Rescue
RTO Project
ScriptedByHer[2.0] | Theme: Building for Bharat with Agentic AI

# Delivery Rescue — Agentic AI for Last-Mile Delivery
### Voice-to-GPS · Real-time Orchestration · Tier 2/3 India

> *"Address is not where you live. It's whether the world can reach you."*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Problem We Solve](#2-the-problem-we-solve)
3. [Our Solution](#3-our-solution)
4. [Architecture Diagram](#4-architecture-diagram)
5. [Agentic AI Design](#5-agentic-ai-design)
6. [LangGraph Flow Diagram](#6-langgraph-flow-diagram)
7. [Request / Response Flow](#7-request--response-flow)
8. [File Structure & What Each File Does](#8-file-structure--what-each-file-does)
9. [Landmark Database Design](#9-landmark-database-design)
10. [Libraries & Open Source Tools](#10-libraries--open-source-tools)
11. [How to Set Up and Run](#11-how-to-set-up-and-run)
12. [API Reference](#12-api-reference)
13. [Demo Scenarios](#13-demo-scenarios)
14. [Edge Cases Handled](#14-edge-cases-handled)

---

## 1. Project Overview

**Delivery Rescue** is a multi-agent AI system that intercepts failing deliveries
in real time — the moment a delivery partner marks an address as unclear — and
resolves the location autonomously in under 90 seconds without any human operator.

| Stat | Value |
|------|-------|
| Target reduction in RTOs | 9 percentage points |
| Time to resolution | < 90 seconds |
| Human intervention required | Zero |
| Languages supported | Hindi, Bhojpuri, Maithili, Awadhi, Marathi, Tamil + more |
| Cities in landmark database | 16 Tier 2/3 cities |
| States covered | Bihar, UP, Rajasthan, MP, Maharashtra |

---

## 2. The Problem Statement

### India's addressing crisis

Over **500 million Indians** have no structured address. They navigate using
landmarks — temples, schools, chowks, bus stands — with directions like:

> *"Hanuman Mandir ke peeche, teesra ghar, neeli deewar"*
> *(Behind the Hanuman Temple, third house, blue wall)*

This causes:
- **20–30%** of COD e-commerce orders to fail and return (RTO)
- **₹80–200** lost per RTO (forward + reverse logistics)
- Sellers losing income, buyers losing trust, delivery partners not getting paid

### Why existing solutions fail

| Solution | When it runs | Why it's insufficient |
|----------|-------------|----------------------|
| Order-time geocoding (e.g. GeoIndia) | At checkout | Makes a static prediction once; can't handle "near the yellow gate" |
| Generic IVR calls | On failure | English/Hindi only, no landmark extraction, no GPS update |
| Manual ops escalation | After all attempts fail | Slow, expensive, doesn't scale |
| **Our system** | **At the moment of failure, at the door** | **Real-time voice → GPS in 90 seconds** |

### The ambiguity problem (why this is hard)

The same landmark name exists **multiple times** in every Tier 2/3 city:

| City | Landmark | Count in city |
|------|----------|---------------|
| Muzaffarpur | "Hanuman Mandir" | 4 different temples |
| Varanasi | "Shiv Mandir" | 18 different temples |
| Gorakhpur | "Panchayat Bhawan" | 22 different offices |
| Jodhpur | "Kirana Store" | 500+ stores |
| Ujjain | "Dharmashala" | 25+ rest houses |

A static geocoder returns the first result. Our system **detects ambiguity**,
**asks a clarifying question** in the customer's own dialect, and resolves it.

---

## 3. Our Solution

A **3-agent agentic AI system** orchestrated by LangGraph:

```
Driver taps "Address Unclear"
        │
        ▼
┌────────────────────────────────────────────────────────┐
│              LangGraph Orchestrator                    │
│                                                        │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────┐   │
│  │ Voice Agent │──▶│ Spatial Agent│──▶ │  Route   │   │
│  │             │    │              │    │  Agent   │   │
│  │ • Calls in  │    │ • Local CSV  │    │ • ≥75%   │   │
│  │   dialect   │    │   DB search  │    │   auto-  │   │
│  │ • Whisper   │    │ • OSM fallbk │    │   push   │   │
│  │   ASR       │    │ • Ambiguity  │    │ • 50-75% │   │
│  │ • Claude    │    │   detection  │    │   flag   │   │
│  │   extract   │    │ • Confidence │    │ • <50%   │   │
│  └─────────────┘    │   scoring    │    │   retry  │   │
│         ▲           └──────────────┘    └──────────┘   │
│         │                                    │         │
│         └────── retry_voice ◀──── low conf ──┘        │
│                                    │                   │
│                              ┌─────▼─────┐             │
│                              │ Escalate  │             │
│                              │ (≥3 fail) │             │
│                              └───────────┘             │
└────────────────────────────────────────────────────────┘
        │
        ▼
Driver's map pin updates. Delivery saved.
```

---

## 4. Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Browser)                            │
│  ┌──────────────┐  ┌───────────────────┐  ┌──────────────────────┐    │
│  │  Order Queue │  │  Agent Flow View  │  │  GPS Map + Metrics   │    │
│  │  (4 scenarios│  │  (LangGraph       │  │  (Confidence meter,  │    │
│  │   hardcoded) │  │   nodes animated) │  │   live pin drop)     │    │
│  └──────┬───────┘  └────────┬──────────┘  └──────────────────────┘    │
│         │                   │                                         │
│         │    WebSocket      │    Server-Sent Events                   │
└─────────┼───────────────────┼─────────────────────────────────────────┘
          │                   │
          ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI, port 8000)                     │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  LangGraph StateGraph                         │  │
│  │                                                               │  │
│  │  voice_agent ──▶ spatial_agent ──▶ route_agent               │  │
│  │       ▲                                 │                     │  │
│  │       └──────────── retry ──────────────┘                     │  │
│  │                         └──── escalate ──▶escalate_node      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │
│  │  /api/      │  │  landmarks   │  │  External Services          │ │
│  │  transcribe │  │  .csv        │  │                             │ │
│  │  (Whisper   │  │  (41 real    │  │  • Claude API (Anthropic)   │ │
│  │   ASR)      │  │   landmarks) │  │  • OSM Nominatim (free)     │ │
│  └─────────────┘  └──────────────┘  │  • Whisper (local, free)    │ │
│                                     └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Agentic AI Design

### Why this is genuinely "agentic" (not just a pipeline)

A pipeline always runs A→B→C in the same order.
An agentic system **makes decisions** based on what's happening.

```
What makes it agentic:

1. SHARED STATE (not API calls between agents)
   ┌──────────────────────────────────────┐
   │ RescueState (TypedDict)              │
   │  order_id, raw_address, pincode      │
   │  audio_transcript    ← voice writes  │
   │  extracted_landmarks ← voice writes  │
   │  geocode_result      ← spatial writes│
   │  confidence_score    ← spatial writes│
   │  final_gps           ← route writes  │
   │  retry_count, status, error_log      │
   └──────────────────────────────────────┘
   Agents never call each other.
   They write to state. Orchestrator decides what runs next.

2. CONDITIONAL ROUTING (the brain)
   routing_decision(state) → one of:
     "end"         if resolved
     "escalate"    if retry_count ≥ 3
     "retry_voice" if confidence < 0.50 and retries remain

3. LOOP-BACK CAPABILITY
   Voice Agent can run multiple times in a single rescue.
   Each time: better transcript, more specific question asked.

4. GRACEFUL DEGRADATION
   LLM fails → keyword fallback (still works)
   OSM unreachable → local CSV only (still works)
   Call unanswered → WhatsApp fallback (still works)
   3 total failures → human escalation (never silent fail)
```

### Confidence threshold design

```
Confidence Score  │  Action
──────────────────┼──────────────────────────────
≥ 0.75            │  AUTO-PUSH   — GPS sent to driver, delivery proceeds
0.50 – 0.74       │  PUSH+FLAG   — GPS sent, ops team notified for review
0.25 – 0.49       │  RETRY       — Loop back to Voice Agent, ask more
< 0.25            │  ESCALATE    — Human ops, create support ticket
```

### Ambiguity detection

When the local landmark CSV contains multiple entries with similar names
in the same city, the scoring algorithm detects this and halves confidence:

```python
close_matches = [s for s, _ in scored if s >= top_score - 0.12]
if len(close_matches) > 1:
    top_score *= 0.55   # ambiguity penalty
```

This forces a retry where the Voice Agent asks a more specific question.

---

## 6. LangGraph Flow Diagram

```
                    START
                      │
                      ▼
              ┌───────────────┐
              │  voice_agent  │◀─────────────────────┐
              │               │                       │
              │ 1. Check call │                       │
              │    answered?  │                       │
              │ 2. Detect     │                       │
              │    noise      │                       │
              │ 3. Clean      │                       │
              │    transcript │                       │
              │ 4. Claude AI  │                       │
              │    extract    │                       │
              │    landmarks  │                       │
              └──────┬────────┘                       │
                     │                                │
                     ▼                                │
           ┌─────────────────┐                        │
           │  spatial_agent  │                        │
           │                 │                        │
           │ 1. Search local │                        │
           │    landmarks.csv│                        │
           │ 2. Detect       │                        │
           │    ambiguity    │                        │
           │ 3. OSM fallback │                        │
           │ 4. Score        │                        │
           │    confidence   │                        │
           └──────┬──────────┘                        │
                  │                                   │
                  ▼                                   │
          ┌──────────────┐                            │
          │  route_agent │                            │
          └──────┬───────┘                            │
                 │                                    │
         routing_decision(state)                      │
                 │                                    │
      ┌──────────┼──────────────┐                     │
      │          │              │                     │
      ▼          ▼              ▼                     │
    "end"    "escalate"   "retry_voice" ──────────────┘
      │          │
      │          ▼
      │   ┌─────────────┐
      │   │  escalate   │
      │   │  node       │
      │   └──────┬──────┘
      │          │
      └──────────┤
                 ▼
                END
```

---

## 7. Request / Response Flow

### Full sequence: from driver tap to GPS delivered

```mermaid
sequenceDiagram
    autonumber
    actor App as Delivery Partner App
    participant BE as Backend
    participant Ext as External Services / LLM

    App->>BE: HTTP POST /api/mock-call<br/>{scenario: "normal"}
    BE->>Ext: Simulate IVR call
    Ext-->>BE: {transcript: "..."}
    BE-->>App: {answered: true, transcript: "..."}

    App->>BE: WS CONNECT /ws/rescue/ORD-001
    App->>BE: WS SEND {raw_address, pincode, transcript, language}
    BE-->>App: WS EVENT "started"

    Note over BE, Ext: LangGraph.astream() initiated
    
    Note over BE: [voice_agent runs]
    BE->>Ext: Claude API (Extract Landmarks)
    Ext-->>BE: {landmarks: [...]}
    BE-->>App: WS EVENT "step"<br/>{node: "voice_agent", state: {landmarks}}

    Note over BE: [spatial_agent runs]
    Note over BE: Local Search: landmarks.csv
    BE->>Ext: OSM Nominatim (Geocode)
    Ext-->>BE: {lat, lng, conf}
    BE-->>App: WS EVENT "step"<br/>{node: "spatial_agent", state: {geocode}}

    Note over BE: [route_agent runs]
    Note over BE: Confidence Gate Check
    BE-->>App: WS EVENT "step"<br/>{node: "route_agent", state: {final_gps}}
    
    BE-->>App: WS EVENT "complete"
    
    Note over App: UI Updates:<br/>Map renders, GPS pin drops,<br/>confidence score shown.
```
Download `Mermaid` to view this diagram.

### Audio transcription flow

```mermaid
sequenceDiagram
    autonumber
    actor Mic as Browser Mic
    participant BE as Backend
    participant Whisper as Whisper (Local)

    Mic->>BE: Record audio (MediaRecorder API)
    Note over BE: Save audio to temp file
    
    BE->>Whisper: model.transcribe(temp_file)
    Whisper-->>BE: {text, language, noise_prob}
    
    BE-->>Mic: {transcript, noise_detected, language}
```
Download `Mermaid` to view this diagram.

---

## 8. File Structure & What Each File Does

```
delivery-rescue/
│
├── backend.py              Main FastAPI app + all LangGraph agents
│                           1019 lines. Self-contained.
│                           Run: uvicorn backend:app --reload --port 8000
│
├── landmarks.csv           Real landmark database — 41 entries
│                           16 Tier 2/3 cities across 5 states
│                           Each row = one real physical landmark
│                           Includes ambiguity metadata
│                           HOW TO EXPAND: add more rows, same format
│
├── requirements.txt        All Python packages with pinned versions
│                           Install: pip install -r requirements.txt
│
├── DeliveryRescue_Meesho.html
│                           Complete frontend — single HTML file
│                           Zero server, zero npm, double-click to run
│                           Uses Tailwind CDN + React CDN + Babel CDN
│                           All 4 demo scenarios hardcoded
│
└── README.md               This file
```

### Inside `backend.py` — section by section

| Lines | Section | What it does |
|-------|---------|--------------|
| 1–50 | Imports + app init | FastAPI setup, CORS, Claude client init |
| 51–110 | Landmark loading | Reads landmarks.csv, builds token index |
| 111–200 | `_local_search()` | Fuzzy-match function — the core search logic |
| 201–240 | `_osm_search()` | Async OSM Nominatim fallback |
| 241–280 | `RescueState` TypedDict | Shared state schema — all agents read/write this |
| 281–410 | `voice_agent()` | Node 1 — call, noise, ASR, Claude extraction |
| 411–510 | `spatial_agent()` | Node 2 — local search, OSM fallback, confidence |
| 511–580 | `route_agent()` | Node 3 — confidence gate, push/flag/retry/escalate |
| 581–600 | `escalate_node()` | Final fallback — human ops notification |
| 601–630 | `routing_decision()` | The conditional edge — orchestrator brain |
| 631–670 | `build_graph()` | Assembles the LangGraph StateGraph |
| 671–740 | `/ws/rescue/` | WebSocket endpoint — streams state updates |
| 741–800 | `/api/transcribe` | Whisper ASR endpoint |
| 801–840 | `/api/mock-call` | Simulated IVR (replaces Exotel for demo) |
| 841–880 | `/api/geocode` | Direct geocode testing endpoint |
| 881–920 | `/api/health` | Health check + system status |
| 921–960 | `/api/landmarks` | List landmarks (for map preview) |

### Inside `landmarks.csv` — column meanings

| Column | Meaning |
|--------|---------|
| `landmark_name` | What the customer says ("Hanuman Mandir") |
| `alias_1`, `alias_2` | Other names same landmark is called |
| `raw_address_example` | Real example of how an address is given |
| `city`, `district`, `state` | Geographic location |
| `pincode` | India Post pincode for this area |
| `lat`, `lng` | Coordinates (verified from government/OSM sources) |
| `landmark_type` | temple / transport / government / education / medical / shop |
| `ambiguity_level` | LOW / MEDIUM / HIGH / VERY HIGH |
| `same_name_count_in_city` | How many identical-name landmarks in same city |
| `ambiguity_note` | Why this is ambiguous — shown to ops team |
| `distinguishing_clue_needed` | What to ask the customer to resolve ambiguity |

---

## 9. Landmark Database Design

### Why CSV and not a database?

For this prototype: CSV is the right choice.
- Zero setup — no Postgres/MongoDB installation needed
- Git-friendly — can be edited in Excel/Google Sheets
- Portable — works anywhere Python runs
- Fast enough — 41 rows searched in microseconds

### How to expand the database

Open `landmarks.csv` in Excel or Google Sheets and add rows. Rules:
1. One row = one specific physical location
2. If same landmark name exists 3 times in a city → add 3 rows, each with different lat/lng
3. Set `ambiguity_level` honestly — this drives confidence scoring
4. Fill `distinguishing_clue_needed` — Voice Agent uses this for retry questions

### How the fuzzy search works

```
Input: ["Hanuman Mandir", "842001"]
         │
         ▼
Tokenize: {"hanuman", "mandir", "842001"}
         │
         ▼
For each row in landmarks.csv:
  - Overlap with landmark_name tokens → +0.35
  - Overlap with alias tokens          → +0.15 each
  - Pincode exact match               → +0.25
  - City/district name match          → +0.15
  - "multiple/ambiguous" in note      → -0.10
         │
         ▼
Sort by score. Top result returned.
If 2+ results within 0.12 of each other → AMBIGUOUS → confidence *= 0.55
```

---

## 10. Libraries & Open Source Tools

### Backend

| Library | Version | Purpose | Why chosen |
|---------|---------|---------|------------|
| **FastAPI** | 0.115 | Web framework + WebSocket | Async-native, auto API docs, fastest Python web framework |
| **LangGraph** | 0.2.28 | Multi-agent orchestration | State machine + conditional routing + loops — only framework that does this properly |
| **langchain-core** | 0.3.8 | LangGraph dependency | Provides base types and utilities |
| **Anthropic** | 0.34.2 | Claude API client | Claude Haiku is fast and cheap for landmark extraction |
| **httpx** | 0.27.2 | Async HTTP client | Used for OSM Nominatim requests |
| **uvicorn** | 0.30.6 | ASGI server | Runs FastAPI, supports WebSocket natively |
| **pydantic** | 2.9.2 | Data validation | FastAPI uses this internally; we use for request models |
| **openai-whisper** | optional | Speech-to-text | Free, runs locally, supports Hindi/Bhojpuri out of box |

### Frontend

| Tool | How used | Why |
|------|---------|-----|
| **React 18** | UI framework (via CDN) | Component-based, fast, works from CDN |
| **Babel Standalone** | JSX transform in browser | Allows JSX without build step |
| **Tailwind CSS** | Utility-class styling (via CDN) | No build needed, fast prototyping |
| **Browser MediaRecorder API** | Voice capture | Built into every browser, no library needed |

### Data & Maps

| Tool | Purpose | Cost |
|------|---------|------|
| **OpenStreetMap Nominatim** | Geocoding fallback | Free, no key needed |
| **landmarks.csv** | Primary geocoding | Free, self-maintained |
| **Census India 2011** | Coordinate verification source | Free public data |

### Why LangGraph specifically?

Most hackathon teams use one of:
- LangChain: good for sequential chains, bad for loops and conditional routing
- CrewAI: good for agent roles, bad for shared typed state
- Custom code: works but no retry/loop/state management built in

LangGraph gives us:
```
✓ TypedDict shared state (type-safe, auditable)
✓ Conditional edges (if/else routing)
✓ Loop-back capability (retry voice → spatial)
✓ Stream each step (for live UI updates)
✓ Checkpointing (can resume if interrupted)
```

---

## 11. How to Set Up and Run

### Prerequisites

| Tool | Where to get | Check if installed |
|------|-------------|-------------------|
| Python 3.10+ | python.org/downloads | `python --version` |
| pip | comes with Python | `pip --version` |
| Claude API key | console.anthropic.com | — |
| (optional) ffmpeg | ffmpeg.org (for Whisper audio) | `ffmpeg -version` |

### Step 1 — Download project files

Save these files to a folder on your computer:
```
delivery-rescue/
  backend.py
  landmarks.csv
  requirements.txt
  DeliveryRescue_Meesho.html
```

### Step 2 — Create isolated Python environment

Open Terminal (Mac/Linux) or Command Prompt (Windows):

```bash
# Navigate to your folder
cd delivery-rescue

# Create virtual environment
python -m venv venv

# Activate it
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# You'll see (venv) at the start of your prompt
```

### Step 3 — Install Python packages

```bash
pip install -r requirements.txt
```

This takes 2–3 minutes the first time. You'll see packages downloading.

### Step 4 — Set your Claude API key

```bash
# Mac/Linux:
export ANTHROPIC_API_KEY=sk-ant-YOUR-ACTUAL-KEY-HERE

# Windows:
set ANTHROPIC_API_KEY=sk-ant-YOUR-ACTUAL-KEY-HERE
```

Get your key at: https://console.anthropic.com → API Keys → Create Key

### Step 5 — Run the backend

```bash
uvicorn backend:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Loaded 41 landmarks from landmarks.csv
INFO:     LangGraph compiled successfully.
```

### Step 6 — Verify it's working

Open your browser and go to: http://localhost:8000/api/health

You should see:
```json
{
  "status": "ok",
  "landmarks_loaded": 41,
  "claude_key_set": true,
  "graph_nodes": ["voice_agent", "spatial_agent", "route_agent", "escalate"]
}
```

### Step 7 — Open the frontend

Double-click `DeliveryRescue_Meesho.html`
OR open it in your browser.

**The simulation works immediately** — no connection to the backend needed.
The HTML file has the full demo simulation built in.

To connect to the real backend: the WebSocket at `ws://localhost:8000/ws/rescue/{id}`
can be called from any frontend. The HTML demo uses simulated data for reliability.

### Optional: Install Whisper (real voice transcription)

```bash
pip install openai-whisper

# Also install ffmpeg (system-level):
# Mac:   brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: download from ffmpeg.org, add to PATH
```

First transcription call downloads the Whisper model (~150MB). Cached after that.

---

## 12. API Reference

All endpoints available at http://localhost:8000
Auto-generated docs at http://localhost:8000/docs

### `GET /api/health`
Returns system status, landmark count, Claude key status.

### `GET /api/landmarks?state=Bihar&city=Muzaffarpur`
Returns all landmarks, optionally filtered.

### `POST /api/mock-call`
Simulates an IVR phone call (replaces Exotel for demo).
```json
Request:  {"scenario": "normal", "language": "Bhojpuri", "order_id": "ORD-001"}
Response: {"answered": true, "transcript": "Panchayat bhawan ke peeche...", "noise_detected": false}
```

### `POST /api/transcribe`
Upload an audio file, get back transcript.
```
Request:  multipart/form-data, field "audio" = audio file (webm/wav/mp3)
Response: {"transcript": "...", "language": "hi", "noise_detected": false}
```

### `POST /api/geocode`
Test geocoding directly without running the full agent graph.
```json
Request:  {"landmarks": ["Hanuman Mandir"], "pincode": "842001", "city": "Muzaffarpur"}
Response: {"source": "local_db", "result": {"lat": 26.1219, "lng": 85.3906, "confidence": 0.72}}
```

### `WS /ws/rescue/{order_id}`
Main WebSocket endpoint. Send initial state, receive streaming events.

Send:
```json
{
  "raw_address": "Panchayat bhawan ke peeche teesra ghar",
  "pincode": "842001",
  "city_hint": "Muzaffarpur",
  "language": "Bhojpuri",
  "call_answered": true,
  "transcript": "Panchayat bhawan ke peeche teesra ghar neeli deewar",
  "retry_count": 0
}
```

Receive (streaming):
```json
{"event": "started", "state": {...}}
{"event": "step", "node": "voice_agent", "state": {...}}
{"event": "step", "node": "spatial_agent", "state": {...}}
{"event": "step", "node": "route_agent", "state": {...}}
{"event": "complete"}
```

---

## 13. Demo Scenarios

### Scenario 1 — Normal (ORD-4821, Sunita Devi, Muzaffarpur)
```
Address: "Panchayat bhawan ke peeche, teesra ghar, neeli deewar"
Language: Bhojpuri
Flow: Call connects → transcript clean → Panchayat Bhawan found → 88% confidence → AUTO-PUSH
Duration: ~8 seconds
```

### Scenario 2 — No Answer (ORD-5503, Ramesh Kumar, Gorakhpur)
```
Address: "Hanuman mandir baaju mein, peela gate, second lane"
Language: Hindi
Flow: Call → no answer → WhatsApp voice note sent → customer replies → Hanuman Mandir found
     → 72% confidence → PUSH+FLAG (multiple mandirs in Gorakhpur)
Duration: ~15 seconds
```

### Scenario 3 — Noisy Audio (ORD-6712, Priya Singh, Darbhanga)
```
Address: "[noise] school ke [inaudible] lal deewar [static] ghar"
Language: Maithili
Flow: Call connects → noise detected → audio cleaned → Primary School found
     → confidence capped at 68% (noise penalty) → PUSH+FLAG
Duration: ~12 seconds
```

### Scenario 4 — Low Confidence / Ambiguous (ORD-7890, Deepak Yadav, Varanasi)
```
Address: "koi mandir ke paas... ek dukaan bhi shayad"
Language: Bhojpuri
Flow: Call → vague transcript → Shiv Mandir matched (18 in Varanasi!) → 42% confidence
     → RETRY → asks: "Kaunsa mandir — ghat ke paas?" → customer clarifies
     → better match → 65% confidence → PUSH+FLAG
Duration: ~25 seconds
```

---

## 14. Edge Cases Handled

| Situation | Detection | Response |
|-----------|-----------|---------|
| Customer doesn't answer | `call_answered = False` | WhatsApp voice note sent in dialect |
| Background noise / market sounds | Noise markers in ASR output (`[noise]`, `[inaudible]`) | Strip markers, clean transcript, reduce confidence by 18% |
| Transcript too short after cleaning | `len(transcript) < 5` | Retry with explicit "please repeat" |
| Same landmark name in multiple locations | Multiple CSV rows with similar scores (within 0.12) | Ambiguity flag → confidence halved → retry with clarifying question |
| No match in local DB | `_local_search()` returns empty | Fall back to OSM Nominatim API |
| OSM returns no result | Empty response | Set confidence to 0, trigger retry |
| LLM API fails | Exception catch | Keyword fallback extraction (temple, mandir, school, etc.) |
| 3 consecutive failures | `retry_count >= 3` | Escalate to human ops with full transcript and error log |
| Confidence 50–75% | Route agent threshold | Push GPS but flag for post-delivery verification |
| Very low confidence < 25% | Route agent threshold | Escalate immediately without retry |

---

## Sources & References

- Census India 2011 — population data for city tier classification
- UDISE+ 2022-23 — primary school counts per district
- India Post Pincode Directory — pincode-to-district mapping
- OpenStreetMap contributors — geographic coordinate verification
- iCarry.in Logistics Blog — RTO rate statistics for Tier 2/3 cities
- Mordor Intelligence India E-commerce Logistics Report 2025-26 — industry cost data
- Delhivery RTO Predictor documentation — baseline RTO reduction benchmarks
- Google Maps Architecture Blog — India address ambiguity research
- SHRUG Dataset (Asher et al., 2021) — high-resolution India geographic data

---

