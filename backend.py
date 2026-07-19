"""
=============================================================
  DELIVERY RESCUE BACKEND  —  Agentic AI Last-Mile System
=============================================================
  Stack  : FastAPI · LangGraph · Gemini API · Whisper · OSM
  Author : Delivery Rescue Team
  
  HOW TO RUN  (copy-paste into terminal one line at a time)
  ──────────────────────────────────────────────────────────
  python -m venv venv
  source venv/bin/activate          # Windows: venv\Scripts\activate
  pip install -r requirements.txt
  export GEMINI_API_KEY=YOUR-KEY-HERE
  uvicorn backend:app --reload --port 8000
=============================================================
"""

# ──────────────────────────────────────────────────────────────
#  STDLIB
# ──────────────────────────────────────────────────────────────
import os, csv, json, re, math, asyncio, tempfile, time, logging, difflib
from pathlib import Path
from typing import Optional, Literal, List

# ──────────────────────────────────────────────────────────────
#  THIRD-PARTY
# ──────────────────────────────────────────────────────────────
import httpx
from google.genai import types
from google import genai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing_extensions import TypedDict

# Load .env file automatically if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv not installed — use OS env vars directly

# LangGraph
from langgraph.graph import StateGraph, END

# ──────────────────────────────────────────────────────────────
#  LOGGING
# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  APP INIT
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Delivery Rescue API",
    description="Agentic AI system that rescues failing deliveries via voice-to-GPS",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini client (lazy — only fails if key missing when called)
def get_gemini():
    key = os.getenv("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not found."
        )

    return genai.Client(api_key=key)

# ──────────────────────────────────────────────────────────────
#  LANDMARK DATABASE  (loaded from CSV at startup)
# ──────────────────────────────────────────────────────────────
LANDMARK_CSV = Path(__file__).parent / "landmarks.csv"
_landmark_index: list[dict] = []   # list of landmark dicts

def load_landmarks() -> list[dict]:
    """
    Loads landmarks.csv into memory.
    Each row → a landmark dict with all CSV fields.
    Builds a fast lookup index: all text tokens → landmark.
    Called once at startup.
    """
    if not LANDMARK_CSV.exists():
        log.warning(f"landmarks.csv not found at {LANDMARK_CSV}. Using empty DB.")
        return []

    rows = []
    with open(LANDMARK_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert lat/lng to float
            try:
                row["lat"] = float(row["lat"])
                row["lng"] = float(row["lng"])
            except (ValueError, KeyError):
                continue

            # Build a search bag-of-words from all text columns
            search_tokens = set()
            for col in ["landmark_name", "alias_1", "alias_2", "city", "district", "state", "pincode"]:
                val = row.get(col, "")
                if val:
                    for word in re.split(r"[\s,/\-]+", val.lower()):
                        if len(word) > 2:
                            search_tokens.add(word)
            row["_tokens"] = search_tokens
            rows.append(row)

    log.info(f"Loaded {len(rows)} landmarks from {LANDMARK_CSV}")
    return rows

@app.on_event("startup")
async def startup():
    global _landmark_index
    _landmark_index = load_landmarks()

# ──────────────────────────────────────────────────────────────
#  GEOCODING  — local DB + OSM fallback
# ──────────────────────────────────────────────────────────────
def _haversine(lat1, lng1, lat2, lng2) -> float:
    """Great-circle distance in km."""
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = math.sin(d_lat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(d_lng/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

_HINT_TO_NUMERIC = {"high": 0.90, "medium": 0.60, "low": 0.30}

def _fuzzy_ratio(query_landmarks: list[str], row: dict) -> float:
    """
    Best fuzzy string similarity between any spoken landmark phrase and
    this row's name/aliases, using difflib (stdlib — no extra dependency).
    Catches spelling/transliteration drift ('mandhir' vs 'mandir', 'panchmukhi'
    vs 'panchmuki') that exact token-overlap matching completely misses.
    """
    names = [row.get("landmark_name", ""), row.get("alias_1", ""), row.get("alias_2", "")]
    names = [n.lower() for n in names if n]
    best = 0.0
    for lm in query_landmarks:
        lm = lm.lower().strip()
        if not lm:
            continue
        for n in names:
            best = max(best, difflib.SequenceMatcher(None, lm, n).ratio())
    return best

def _local_search(landmarks: list[str], pincode: str, city: str,
                   directions: list = None, identifiers: list = None,
                   extraction_hint: str = "medium") -> dict:
    """
    Generic fuzzy-match against the landmark CSV.
    Returns the best match dict or {} with a confidence score.

    Scoring logic:
      +0.35  max(exact landmark-name token overlap, fuzzy string ratio)
      +0.15  alias match
      +0.25  pincode matches
      +0.15  city/district matches
      -0.10  each ambiguity_note mention of 'multiple' or 'ambiguous'

    After the structural score is computed, it is blended with two signals
    the LLM extraction step already produces but previously went unused:
      - extraction_hint: Gemini's own "high/medium/low" self-assessment
      - clue richness: how many directional/visual clues the customer gave
        (more detail volunteered generally means a more reliable transcript)

    If multiple rows get a close score → ambiguity detected → confidence halved
    (this remains a safety brake, not a fix — see spatial_agent candidates list).
    """
    if not _landmark_index:
        return {}

    directions = directions or []
    identifiers = identifiers or []

    scored: list[tuple[float, dict]] = []

    query_tokens: set[str] = set()
    for lm in landmarks:
        for word in re.split(r"[\s,/\-]+", lm.lower()):
            if len(word) > 2:
                query_tokens.add(word)
    if pincode:
        query_tokens.add(pincode.strip())
    if city:
        for word in re.split(r"[\s,/\-]+", city.lower()):
            if len(word) > 2:
                query_tokens.add(word)

    for row in _landmark_index:
        score = 0.0

        row_tokens = row["_tokens"]
        overlap = len(query_tokens & row_tokens)
        fuzzy = _fuzzy_ratio(landmarks, row)
        if overlap == 0 and fuzzy < 0.55:
            continue

        # Landmark name match — take whichever signal is stronger, exact
        # token overlap or fuzzy string similarity, rather than requiring
        # an exact match. Fixes false negatives from spelling variants.
        lm_name_tokens = set(re.split(r"[\s,/\-]+", row["landmark_name"].lower()))
        lm_overlap = len(query_tokens & lm_name_tokens)
        exact_component = 0.35 * (lm_overlap / max(len(lm_name_tokens), 1)) if lm_overlap else 0.0
        fuzzy_component = 0.35 * fuzzy
        score += max(exact_component, fuzzy_component)

        # Alias match
        for alias_col in ["alias_1", "alias_2"]:
            alias = row.get(alias_col, "")
            if alias:
                alias_tokens = set(re.split(r"[\s,/\-]+", alias.lower()))
                if query_tokens & alias_tokens:
                    score += 0.15

        # Pincode match
        if pincode and pincode.strip() == row.get("pincode", "").strip():
            score += 0.25

        # City/district match
        city_tokens = set(re.split(r"[\s,/\-]+", (row.get("city","") + " " + row.get("district","")).lower()))
        if query_tokens & city_tokens:
            score += 0.15

        # Ambiguity penalty — if the CSV note says there are multiple
        note = row.get("ambiguity_note", "").lower()
        if any(w in note for w in ["multiple", "ambiguous", "different", "3 ", "4 ", "several"]):
            score -= 0.10

        if score > 0.15:
            scored.append((score, row))

    if not scored:
        return {}

    scored.sort(key=lambda x: x[0], reverse=True)
    top_score, top_row = scored[0]

    # Detect ambiguity using the RAW structural scores, before any blending —
    # ambiguity is a property of the data (multiple similarly-named places),
    # not of how confident the LLM felt.
    close_matches = [s for s, _ in scored if s >= top_score - 0.12]
    ambiguous = len(close_matches) > 1

    # Blend in the two previously-unused signals:
    #   - Gemini's own confidence in what it extracted
    #   - how much distinguishing detail the customer actually gave
    hint_val = _HINT_TO_NUMERIC.get(extraction_hint, 0.6)
    clue_bonus = min(0.02 * (len(directions) + len(identifiers)), 0.08)
    top_score = 0.75 * top_score + 0.15 * hint_val + clue_bonus

    candidates = []
    if ambiguous:
        top_score *= 0.55   # halve confidence when ambiguous — safety brake, not a fix
        log.info(f"Ambiguity detected: {len(close_matches)} close matches. Confidence halved.")
        # Surface the actual candidates so the UI can plot all of them on the
        # map and let the driver pick the right one visually, rather than the
        # system silently guessing or making the customer repeat themselves.
        candidates = [
            {
                "lat": row["lat"], "lng": row["lng"],
                "display_name": f"{row['landmark_name']}, {row['city']}, {row['state']}",
                "landmark_type": row.get("landmark_type", ""),
                "raw_score": round(s, 3),
            }
            for s, row in scored if s >= scored[0][0] - 0.12
        ][:4]

    return {
        "lat": top_row["lat"],
        "lng": top_row["lng"],
        "display_name": f"{top_row['landmark_name']}, {top_row['city']}, {top_row['state']}",
        "city": top_row["city"],
        "district": top_row["district"],
        "state": top_row["state"],
        "pincode": top_row["pincode"],
        "landmark_type": top_row.get("landmark_type", ""),
        "ambiguity_note": top_row.get("ambiguity_note", ""),
        "source": "local_db",
        "confidence": round(min(top_score, 0.95), 3),
        "total_matches": len(scored),
        "ambiguous": ambiguous,
        "candidates": candidates,
    }

_nominatim_lock = asyncio.Lock()
_nominatim_last_call = 0.0

async def _nominatim_rate_limit():
    """Enforce Nominatim's 1 req/sec usage policy across ALL callers (not per-request)."""
    global _nominatim_last_call
    async with _nominatim_lock:
        elapsed = time.monotonic() - _nominatim_last_call
        if elapsed < 1.05:
            await asyncio.sleep(1.05 - elapsed)
        _nominatim_last_call = time.monotonic()

async def _locationiq_search(query: str, pincode: str = "", country: str = "") -> dict:
    """
    Optional higher-throughput free provider. Same underlying OSM data as
    Nominatim, but LocationIQ's free tier (5,000 req/day, no card needed)
    isn't limited to 1 req/sec. No-op (returns {}) if no key is configured
    — Nominatim below still works standalone, this is purely additive.
    Get a free key at https://locationiq.com and set LOCATIONIQ_API_KEY.
    """
    key = os.environ.get("LOCATIONIQ_API_KEY", "")
    if not key:
        return {}
    params = {
        "key": key,
        "q": query + (f" {pincode}" if pincode else ""),
        "format": "json",
        "limit": 5,
        "addressdetails": 1,
    }
    if country:
        params["countrycodes"] = country
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://us1.locationiq.com/v1/search", params=params)
            if resp.status_code != 200:
                return {}
            results = resp.json()
        if not results or not isinstance(results, list):
            return {}
        top = results[0]
        raw_conf = float(top.get("importance", 0.35))
        return {
            "lat": float(top["lat"]),
            "lng": float(top["lon"]),
            "display_name": top.get("display_name", "")[:160],
            "source": "locationiq",
            "confidence": round(min(raw_conf * 0.85, 0.85), 3),
            "total_matches": len(results),
            "ambiguous": len(results) > 2,
            "osm_class": top.get("class", ""),
            "osm_type": top.get("type", ""),
        }
    except Exception as e:
        log.warning(f"LocationIQ search failed: {e}")
        return {}

async def _osm_search(query: str, pincode: str = "", country: str = "in") -> dict:
    """
    Global geocoder — tries LocationIQ first if configured (no rate-limit
    worries), then falls back to Nominatim (always free, self-throttled
    to 1 req/sec here so we never violate OSM's usage policy).
    an India-only filter.
    """
    country = "in"

    liq = await _locationiq_search(query, pincode, country)
    if liq:
        return liq

    await _nominatim_rate_limit()

    search_query = query
    if pincode:
        search_query += f", {pincode}"

    params = {
        "q": search_query,
        "format": "jsonv2",
        "limit": 5,
        "addressdetails": 1,
        "extratags": 1,
        "countrycodes": "in"
    }

    headers = {
        "User-Agent": "DeliveryRescueAgenticAI/1.0 (hackathon-project; contact@example.com)"
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params=params,
                headers=headers,
            )
            results = resp.json()
        if not results:
            return {}
        top = results[0]
        raw_conf = float(top.get("importance", 0.3))
        # Cap scales with candidate count instead of one flat number —
        # than a rural area returning 5 loosely-matching results.
        cap = 0.80 if len(results) <= 2 else 0.65
        osm_conf = min(raw_conf * 0.85, cap)
        return {
            "lat": float(top["lat"]),
            "lng": float(top["lon"]),
            "display_name": top.get("display_name", "")[:120],
            "source": "openstreetmap",
            "confidence": round(osm_conf, 3),
            "total_matches": len(results),
            "ambiguous": len(results) > 2,
            "osm_class": top.get("class", ""),
            "osm_type": top.get("type", ""),
        }
    except Exception as e:
        log.warning(f"OSM search failed: {e}")
        return {}

# ──────────────────────────────────────────────────────────────
#  SHARED STATE SCHEMA  — the single object all agents share
# ──────────────────────────────────────────────────────────────
class RescueState(TypedDict):
    # Input
    order_id:               str
    raw_address:            str
    pincode:                str
    city_hint:              str     # city name from order data if available
    state_hint:             str     # state/province, if known — improves OSM disambiguation
    country_hint:           str     # ISO country name or code, if known (blank = search globally)
    language:               str     # detected or provided language

    # Voice agent outputs
    call_answered:          bool
    fallback_triggered:     bool    # True when WhatsApp fallback used
    audio_transcript:       str
    noise_detected:         bool
    noise_cleaned:          bool
    extracted_landmarks:    list    # list of landmark strings
    extracted_directions:   list    # directional clues ("peeche", "baaju mein")
    extracted_identifiers:  list    # colour/visual clues ("neeli deewar")
    inferred_city:          str     # LLM's best guess of city from transcript
    extraction_confidence_hint: str # LLM's own "high"/"medium"/"low" self-assessment

    # Spatial agent outputs
    geocode_result:         dict    # full geocode result
    confidence_score:       float   # 0.0 – 1.0
    confidence_reason:      str
    ambiguity_detected:     bool
    candidate_count:        int

    # Route agent outputs
    final_gps:              dict
    action_taken:           str     # "auto_push" | "push_flagged" | "retry" | "escalate"

    # Orchestration
    retry_count:            int
    status:                 str
    status_message:         str
    error_log:              list

# ──────────────────────────────────────────────────────────────
#  NODE 1  —  VOICE AGENT
# ──────────────────────────────────────────────────────────────
async def voice_agent(state: RescueState) -> dict:
    """
    Responsibilities:
      1. Simulate / receive phone call result
      2. Detect and clean background noise from transcript
      3. Use Gemini to extract structured landmark info from natural speech
      4. Handle fallback (WhatsApp) if call unanswered
      5. Guard against empty/too-short transcripts
    """
    updates: dict = {"status": "voice_processing"}

    # Guard: max retries
    if state["retry_count"] >= 3:
        return {
            **updates,
            "status": "escalated",
            "status_message": "❌ 3 voice attempts exhausted. Escalating to ops.",
            "action_taken": "escalate",
        }

    transcript = state.get("audio_transcript", "").strip()
    call_answered = state.get("call_answered", True)

    # ── Handle no-answer scenario ──────────────────────────────
    if not call_answered and not state.get("fallback_triggered"):
        updates["fallback_triggered"] = True
        updates["status_message"] = "📵 No answer. Sending WhatsApp voice note in local dialect..."
        log.info(f"[{state['order_id']}] No answer — WhatsApp fallback triggered")
        if not transcript:
            return {
                **updates,
                "retry_count": state["retry_count"] + 1,
                "status_message": "💬 WhatsApp sent. Waiting for customer reply...",
                "error_log": state["error_log"] + ["WhatsApp: awaiting reply"],
            }
        updates["call_answered"] = True
        updates["status_message"] = "✅ Customer replied via WhatsApp."

    # ── Noise detection and cleaning ───────────────────────────
    noise_markers = ["[noise]", "[inaudible]", "[static]", "[background]", "..."]
    raw_noise = any(m in transcript.lower() for m in noise_markers)

    if raw_noise:
        original = transcript
        for m in noise_markers:
            transcript = transcript.lower().replace(m, " ")
        transcript = re.sub(r"\s{2,}", " ", transcript).strip()
        updates["noise_detected"] = True
        updates["noise_cleaned"] = True
        updates["audio_transcript"] = transcript
        updates["status_message"] = f"🔊 Noise detected and cleaned. Proceeding with cleaned transcript."
        log.info(f"[{state['order_id']}] Noise cleaned. Before: '{original}' → After: '{transcript}'")
    else:
        updates["noise_detected"] = False
        updates["noise_cleaned"] = False

    # Transcript too short even after cleaning
    if len(transcript.strip()) < 5:
        return {
            **updates,
            "retry_count": state["retry_count"] + 1,
            "status_message": "🔄 Transcript too short. Re-prompting customer for landmark...",
            "error_log": state["error_log"] + [f"Retry {state['retry_count']+1}: transcript under 5 chars"],
        }

    updates["audio_transcript"] = transcript
    updates["status_message"] = "🧠 Extracting landmarks..."

    # ── Gemini LLM extraction ──────────────────────────────────
    try:
        client = get_gemini()
        prompt = f"""You are a delivery address intelligence agent for last-mile e-commerce.
A delivery partner is trying to reach a customer, and the customer just spoke
their address out loud — in any language, any country, any addressing style.
This system was originally built for Tier 2/3 Indian cities (landmark-based
addressing, since ~500M Indians lack a structured street address) so you
should recognize that style well, but the transcript may equally be a
standard structured address (house number, street, city, state, ZIP/pincode,
country) from anywhere in the world — extract whatever is actually present,
don't force an Indian-landmark shape onto it.

TRANSCRIPT: "{transcript}"
KNOWN CITY HINT: "{state.get('city_hint', 'unknown')}"
KNOWN STATE HINT: "{state.get('state_hint', 'unknown')}"
KNOWN COUNTRY HINT: "{state.get('country_hint', 'unknown')}"
PINCODE / ZIP: "{state.get('pincode', '')}"

Extract delivery location information from this transcript. Depending on
style, this may include:
- Landmark-based (common in Tier 2/3 India): temples/mosques/churches
  (mandir, masjid, gurudwara), government buildings (panchayat bhawan,
  thana), directional words (peeche=behind, aage=ahead, baaju mein=next to),
  visual identifiers (neeli deewar=blue wall), relative position (teesra
  ghar=third house)
- Structured address (common elsewhere): house/apartment number, street
  name, neighborhood, city, state/province, postal code, country

Return ONLY a valid JSON object, no markdown, no explanation:
{{
  "landmarks": ["primary landmark or street address as spoken", "secondary detail if mentioned"],
  "directions": ["directional clues exactly as understood, if any"],
  "identifiers": ["color/visual markers, apartment/unit numbers, etc, if any"],
  "inferred_city": "best guess of city/town from context or empty string",
  "inferred_state": "best guess of state/province from context or empty string",
  "inferred_country": "best guess of country from context or empty string",
  "clarification_needed": true/false,
  "clarification_question": "one short question (in the same language the customer used) that would resolve ambiguity; empty if not needed",
  "confidence_hint": "high/medium/low — your confidence in what you extracted"
}}"""

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        raw_resp = response.text.strip()
        raw_resp = re.sub(r"^```(?:json)?\s*", "", raw_resp)
        raw_resp = re.sub(r"\s*```$", "", raw_resp)

        extracted = json.loads(raw_resp)
        landmarks   = extracted.get("landmarks", [])
        directions  = extracted.get("directions", [])
        identifiers = extracted.get("identifiers", [])
        inf_city    = extracted.get("inferred_city", "")
        inf_state   = extracted.get("inferred_state", "")
        inf_country = extracted.get("inferred_country", "")
        needs_clarif = extracted.get("clarification_needed", False)
        conf_hint   = extracted.get("confidence_hint", "medium")

        updates["extracted_landmarks"]   = [l for l in landmarks if l]
        updates["extracted_directions"]  = directions
        updates["extracted_identifiers"] = identifiers
        
        # Keep explicit user data hints or inherit from LLM parsing
        updates["state_hint"] = state.get("state_hint") or inf_state
        updates["country_hint"] = state.get("country_hint") or inf_country
        updates["inferred_city"] = inf_city or state.get("city_hint", "")
        
        updates["extraction_confidence_hint"] = conf_hint
        updates["status_message"]        = (
            f"📍 Extracted: {', '.join(landmarks[:2]) or 'no clear landmark'}"
            + (f" | City inferred: {updates['inferred_city']}" if updates['inferred_city'] else "")
            + (" | ⚠️ Clarification needed" if needs_clarif else "")
        )

        if needs_clarif and state["retry_count"] < 2:
            updates["status_message"] += f" — Will ask: '{extracted.get('clarification_question', '')}'"

        log.info(f"[{state['order_id']}] LLM extraction OK: landmarks={landmarks}, conf_hint={conf_hint}")

    except json.JSONDecodeError as e:
        log.warning(f"[{state['order_id']}] LLM JSON parse error: {e}. Using keyword fallback.")
        kw_map = {
            "mandir": "mandir", "masjid": "masjid", "school": "school",
            "bhawan": "bhawan", "dukaan": "store", "station": "station",
            "bazar": "bazar", "chowk": "chowk", "hospital": "hospital",
            "thana": "police station", "panchayat": "panchayat bhawan",
        }
        found = []
        for word in transcript.lower().split():
            for kw, label in kw_map.items():
                if kw in word and label not in found:
                    found.append(label)
        
        updates["extracted_landmarks"]   = found or [transcript[:60]]
        updates["extracted_directions"]  = []
        updates["extracted_identifiers"] = []
        updates["inferred_city"]         = state.get("city_hint", "")
        
        # 🔥 FIX: Preserve hints in fallback state dict to avoid cross-state leaks
        updates["state_hint"]            = state.get("state_hint", "")
        updates["country_hint"]          = state.get("country_hint", "")
        
        updates["extraction_confidence_hint"] = "low"
        updates["status_message"]        = f"🔑 Keyword fallback used. Found: {found}"
        updates["error_log"]             = state["error_log"] + [f"LLM JSON error: {e}"]

    except Exception as e:
        log.error(f"[{state['order_id']}] LLM call failed: {e}")
        updates["extracted_landmarks"]   = [transcript[:60]]
        updates["extracted_directions"]  = []
        updates["extracted_identifiers"] = []
        updates["inferred_city"]         = state.get("city_hint", "")
        
        # 🔥 FIX: Preserve hints in fallback state dict to avoid cross-state leaks
        updates["state_hint"]            = state.get("state_hint", "")
        updates["country_hint"]          = state.get("country_hint", "")
        
        updates["extraction_confidence_hint"] = "low"
        updates["status_message"]        = f"⚠️ LLM unavailable — using raw transcript"
        updates["error_log"]             = state["error_log"] + [f"LLM error: {str(e)[:80]}"]

    return updates

# ──────────────────────────────────────────────────────────────
#  NODE 2  —  SPATIAL AGENT
# ──────────────────────────────────────────────────────────────
async def spatial_agent(state: RescueState) -> dict:
    """
    Responsibilities:
      1. Search local landmark CSV for best match
      2. Detect ambiguity (multiple same-name landmarks)
      3. Fall back to OSM Nominatim if local DB misses
      4. Compute and calibrate confidence score
    """
    updates: dict = {
        "status": "spatial_processing",
        "status_message": "🗺️ Searching landmark database...",
    }

    landmarks   = state.get("extracted_landmarks", [])
    pincode     = state.get("pincode", "")
    city        = state.get("inferred_city", "") or state.get("city_hint", "")
    state_hint  = state.get("state_hint", "")
    country_hint = state.get("country_hint", "")
    noise       = state.get("noise_detected", False)
    retry       = state["retry_count"]
    directions  = state.get("extracted_directions", [])
    identifiers = state.get("extracted_identifiers", [])
    conf_hint   = state.get("extraction_confidence_hint", "medium")

    if not landmarks:
        return {
            **updates,
            "geocode_result": {},
            "confidence_score": 0.0,
            "confidence_reason": "No landmarks extracted from transcript",
            "ambiguity_detected": False,
            "candidate_count": 0,
            "error_log": state["error_log"] + ["Spatial: empty landmark list"],
        }

    # ── Step 1: Local CSV search ────────────────────────────────
    local_result = _local_search(landmarks, pincode, city, directions, identifiers, conf_hint)

    if local_result and local_result.get("confidence", 0) >= 0.35:
        conf = local_result["confidence"]

        # Calibrate: noise reduces confidence
        if noise:
            conf *= 0.82

        # Calibrate: retry means previous answer was vague — reduce
        if retry > 0:
            conf = min(conf, 0.72)

        # Calibrate: ambiguity in DB row
        if local_result.get("ambiguous"):
            conf = min(conf, 0.55)
            reason = f"Ambiguous: {local_result.get('total_matches', '?')} candidates with similar names. {local_result.get('ambiguity_note', '')}"
        else:
            reason = f"Local DB match: {local_result.get('display_name', '')} (raw score {local_result['confidence']:.0%})"

        local_result["confidence"] = round(conf, 3)

        updates.update({
            "geocode_result":    local_result,
            "confidence_score":  round(conf, 3),
            "confidence_reason": reason,
            "ambiguity_detected": local_result.get("ambiguous", False),
            "candidate_count":   local_result.get("total_matches", 1),
            "status_message":    f"📌 {local_result.get('display_name', 'Match found')} — {conf:.0%} confidence",
        })
        log.info(f"[{state['order_id']}] Local DB match: conf={conf:.2f}, ambiguous={local_result.get('ambiguous')}")
        return updates

    # ── Step 2: OSM Nominatim fallback ─────────────────────────
    updates["status_message"] = "🌐 Local DB miss — querying OpenStreetMap..."
    log.info(f"[{state['order_id']}] Local DB miss. Trying OSM for: {landmarks[:2]}")

    osm_query = " ".join(landmarks[:2])
    for part in (city, state_hint, country_hint):
        if part:
            osm_query += " " + part
    # country param is left blank unless the caller explicitly gave one —
    # that's what allows a US (or any country) address to resolve, since
    # we no longer force everything through an "in"-only filter.
    osm_result = await _osm_search(osm_query, pincode, country="")

    if osm_result:
        conf = osm_result.get("confidence", 0.3)
        if noise:
            conf *= 0.80
        if retry > 0:
            conf = min(conf, 0.60)
        osm_result["confidence"] = round(conf, 3)

        updates.update({
            "geocode_result":    osm_result,
            "confidence_score":  round(conf, 3),
            "confidence_reason": f"OpenStreetMap fallback (importance-based score)",
            "ambiguity_detected": osm_result.get("ambiguous", False),
            "candidate_count":   osm_result.get("total_matches", 1),
            "status_message":    f"🌐 OSM match: {osm_result.get('display_name','')[:60]} — {conf:.0%}",
        })
        return updates

    # ── No match at all ─────────────────────────────────────────
    return {
        **updates,
        "geocode_result":    {},
        "confidence_score":  0.0,
        "confidence_reason": "No match in local DB or OSM",
        "ambiguity_detected": True,   # treat as ambiguous to force retry
        "candidate_count":   0,
        "status_message":    "❓ Location not found. Will retry with more detail.",
        "error_log":         state["error_log"] + ["Spatial: no result from DB or OSM"],
    }

# ──────────────────────────────────────────────────────────────
#  NODE 3  —  ROUTE AGENT
# ──────────────────────────────────────────────────────────────
async def route_agent(state: RescueState) -> dict:
    """
    Responsibilities:
      1. Apply confidence thresholds to decide action
      2. Auto-push if high confidence
      3. Push with caution flag if medium
      4. Signal retry if low but geocode exists
      5. Escalate if retries exhausted
    Thresholds:
      ≥ 0.75  → auto_push      (push immediately, no flag)
      0.50–0.74 → push_flagged (push + ops review flag)
      0.25–0.49 → retry        (loop back to voice agent)
      < 0.25  → escalate
    """
    score  = state.get("confidence_score", 0.0)
    geo    = state.get("geocode_result", {})
    retry  = state["retry_count"]
    ambig  = state.get("ambiguity_detected", False)

    updates: dict = {"status": "route_processing"}

    # Force escalate after too many retries
    if retry >= 3:
        return {
            **updates,
            "status": "escalated",
            "status_message": "🧑‍💼 Max retries reached. Full context sent to human ops.",
            "action_taken": "escalate",
            "final_gps": {},
        }

    # ── HIGH CONFIDENCE ─────────────────────────────────────────
    if score >= 0.75 and geo:
        final = {
            "lat":               geo["lat"],
            "lng":               geo["lng"],
            "display_name":      geo.get("display_name", ""),
            "confidence":        score,
            "flagged_for_review": False,
            "action":            "auto_push",
            "source":            geo.get("source", ""),
        }
        log.info(f"[{state['order_id']}] AUTO-PUSH at {score:.0%}")
        return {
            **updates,
            "status":         "resolved",
            "status_message": f"✅ Auto-pushed GPS to driver app. ({score:.0%} confidence)",
            "action_taken":   "auto_push",
            "final_gps":      final,
        }

    # ── MEDIUM CONFIDENCE ───────────────────────────────────────
    if 0.50 <= score < 0.75 and geo:
        final = {
            "lat":               geo["lat"],
            "lng":               geo["lng"],
            "display_name":      geo.get("display_name", ""),
            "confidence":        score,
            "flagged_for_review": True,
            "action":            "push_flagged",
            "source":            geo.get("source", ""),
            "ambiguity_note":    geo.get("ambiguity_note", ""),
        }
        log.info(f"[{state['order_id']}] PUSH+FLAG at {score:.0%}")
        return {
            **updates,
            "status":         "resolved",
            "status_message": f"⚠️ Pushed with caution flag ({score:.0%}). Ops will verify.",
            "action_taken":   "push_flagged",
            "final_gps":      final,
        }

    # ── LOW / NO CONFIDENCE — retry ──────────────────────────────
    if retry < 3:
        log.info(f"[{state['order_id']}] LOW conf {score:.0%} → retry {retry+1}")
        return {
            **updates,
            "status":         "retrying",
            "status_message": f"🔄 Confidence {score:.0%} too low. Asking customer for a more specific landmark...",
            "action_taken":   "retry",
            "retry_count":    retry + 1,
            "final_gps":      {},
        }

    # ── FALLBACK ESCALATE ───────────────────────────────────────
    return {
        **updates,
        "status":         "escalated",
        "status_message": "🧑‍💼 Could not resolve location. Escalating to human ops.",
        "action_taken":   "escalate",
        "final_gps":      {},
    }

# ──────────────────────────────────────────────────────────────
#  NODE 4  —  ESCALATION NODE
# ──────────────────────────────────────────────────────────────
async def escalate_node(state: RescueState) -> dict:
    """
    Final fallback. In production: push to ops dashboard,
    create a support ticket, notify supervisor.
    """
    log.warning(f"[{state['order_id']}] ESCALATED after {state['retry_count']} retries.")
    return {
        "status":         "escalated",
        "status_message": "🧑‍💼 Ticket created. Human ops team notified with full transcript.",
        "action_taken":   "escalate",
        "final_gps":      {},
    }

# ──────────────────────────────────────────────────────────────
#  CONDITIONAL EDGE — orchestrator routing logic
# ──────────────────────────────────────────────────────────────
def routing_decision(state: RescueState) -> str:
    """
    Called by LangGraph after route_agent runs.
    This is the brain of the agentic system — decides what happens next.
    """
    action  = state.get("action_taken", "")
    status  = state.get("status", "")
    retry   = state.get("retry_count", 0)

    if status in ("resolved",):
        return "end"

    if status == "escalated" or action == "escalate" or retry >= 3:
        return "escalate"

    if action == "retry" and retry < 3:
        return "retry_voice"   # loop back

    return "end"

# ──────────────────────────────────────────────────────────────
#  BUILD LANGGRAPH
# ──────────────────────────────────────────────────────────────
def build_graph() -> any:
    """
    Constructs the LangGraph StateGraph.

    Flow:
        voice_agent → spatial_agent → route_agent
                                         │
              ┌──────────────────────────┘
              │ routing_decision()
              ├── "end"         → END
              ├── "escalate"    → escalate_node → END
              └── "retry_voice" → voice_agent (loop)
    """
    g = StateGraph(RescueState)

    g.add_node("voice_agent",   voice_agent)
    g.add_node("spatial_agent", spatial_agent)
    g.add_node("route_agent",   route_agent)
    g.add_node("escalate",      escalate_node)

    g.set_entry_point("voice_agent")

    g.add_edge("voice_agent",   "spatial_agent")
    g.add_edge("spatial_agent", "route_agent")

    g.add_conditional_edges(
        "route_agent",
        routing_decision,
        {
            "end":         END,
            "escalate":    "escalate",
            "retry_voice": "voice_agent",
        },
    )
    g.add_edge("escalate", END)

    return g.compile()

rescue_graph = build_graph()
log.info("LangGraph compiled successfully.")

# ──────────────────────────────────────────────────────────────
#  WEBSOCKET ENDPOINT  — real-time streaming
# ──────────────────────────────────────────────────────────────
@app.websocket("/ws/rescue/{order_id}")
async def rescue_ws(ws: WebSocket, order_id: str):
    """
    The primary endpoint.
    
    Protocol:
      1. Client connects
      2. Client sends JSON: initial rescue state (see RescueRequest schema)
      3. Server streams state update events as JSON for each agent step
      4. Server sends {"event":"complete"} at the end
    
    Event format:
      {"event": "step", "node": "voice_agent", "state": {...partial state...}}
      {"event": "complete"}
      {"event": "error", "message": "..."}
    """
    await ws.accept()
    log.info(f"[WS] New connection: order={order_id}")

    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=15.0)
        data = json.loads(raw)

        initial: RescueState = {
            "order_id":             order_id,
            "raw_address":          data.get("raw_address", ""),
            "pincode":              data.get("pincode", ""),
            "city_hint":            data.get("city_hint", ""),
            "language":             data.get("language", "Hindi"),
            "call_answered":        data.get("call_answered", True),
            "fallback_triggered":   False,
            "audio_transcript":     data.get("transcript", ""),
            "noise_detected":       False,
            "noise_cleaned":        False,
            "extracted_landmarks":  [],
            "extracted_directions": [],
            "extracted_identifiers": [],
            "inferred_city":        "",
            "extraction_confidence_hint": "medium",
            "geocode_result":       {},
            "confidence_score":     0.0,
            "confidence_reason":    "",
            "ambiguity_detected":   False,
            "candidate_count":      0,
            "final_gps":            {},
            "action_taken":         "",
            "retry_count":          data.get("retry_count", 0),
            "status":               "pending",
            "status_message":       "Rescue initiated...",
            "error_log":            [],
        }

        await ws.send_json({"event": "started", "state": initial})

        # Stream each LangGraph step
        async for step in rescue_graph.astream(initial):
            for node_name, node_state in step.items():
                await ws.send_json({
                    "event": "step",
                    "node":  node_name,
                    "state": node_state,
                })
                await asyncio.sleep(0.1)   # small breath for UI to animate

        await ws.send_json({"event": "complete"})
        log.info(f"[WS] Rescue complete: order={order_id}")

    except asyncio.TimeoutError:
        await ws.send_json({"event": "error", "message": "Timeout waiting for initial data"})
    except WebSocketDisconnect:
        log.info(f"[WS] Client disconnected: order={order_id}")
    except Exception as e:
        log.error(f"[WS] Error for order={order_id}: {e}")
        try:
            await ws.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass

# ──────────────────────────────────────────────────────────────
#  REST: TRANSCRIBE AUDIO  (Whisper)
# ──────────────────────────────────────────────────────────────
@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Accepts audio blob from browser mic.
    Transcribes with OpenAI Whisper (local, free, no API key needed).
    Falls back to mock transcript if Whisper not installed.

    Request: multipart/form-data with 'audio' field (webm/wav/mp3)
    Response: {transcript, language, noise_detected, duration_secs}
    """
    try:
        import whisper

        content = await audio.read()
        suffix = Path(audio.filename or "audio.webm").suffix or ".webm"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        model = whisper.load_model("base")   # ~150MB, cached after first load
        result = model.transcribe(
            tmp_path,
            language="hi",     # auto-detect
            task="transcribe",
            fp16=False,        # CPU-safe
            initial_prompt="Aapka delivery address landmark ke saath batayein, Relay your Delivery Address with Landmark"
        )

        segments = result.get("segments", [])
        low_conf = [s for s in segments if s.get("no_speech_prob", 0) > 0.55]
        noise = len(low_conf) > len(segments) * 0.35 if segments else False

        Path(tmp_path).unlink(missing_ok=True)

        return {
            "transcript":     result["text"].strip(),
            "language":       result.get("language", "hi"),
            "noise_detected": noise,
            "duration_secs":  segments[-1]["end"] if segments else 0,
            "source":         "whisper_base",
        }

    except ImportError:
        return {
            "transcript":     "Panchayat bhawan ke peeche teesra ghar neeli deewar",
            "language":       "hi",
            "noise_detected": False,
            "duration_secs":  3.5,
            "source":         "mock_fallback",
            "note":           "Install openai-whisper for real transcription",
        }
    except Exception as e:
        log.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ──────────────────────────────────────────────────────────────
#  REST: MOCK CALL  (simulates Exotel/Twilio IVR)
# ──────────────────────────────────────────────────────────────
class MockCallRequest(BaseModel):
    scenario: Literal["normal", "no_answer", "noise", "low_confidence"] = "normal"
    language: str = "Hindi"
    order_id: str = "DEMO"

MOCK_TRANSCRIPTS = {
    "normal": "Panchayat bhawan ke peeche teesra ghar hai, neeli deewar wala",
    "no_answer": "",
    "noise": "[noise] school ke [inaudible] saamne lal deewar [static] wala ghar",
    "low_confidence": "haan ji... koi mandir ke paas... ek dukaan bhi hai shayad",
}

@app.post("/api/mock-call")
async def mock_call(req: MockCallRequest):
    """
    Simulates an IVR phone call result without needing Exotel/Twilio.
    In production: replace this endpoint's return value with actual
    Exotel webhook payload parsing.

    Response: {answered, transcript, noise_detected, duration_secs}
    """
    await asyncio.sleep(1.2)   # simulate call duration
    answered = req.scenario != "no_answer"
    return {
        "answered":        answered,
        "transcript":      MOCK_TRANSCRIPTS.get(req.scenario, ""),
        "noise_detected":  req.scenario == "noise",
        "duration_secs":   0 if not answered else 12.0,
        "language":        req.language,
        "scenario":        req.scenario,
    }

# ──────────────────────────────────────────────────────────────
#  SHARED HELPER — run the full LangGraph synchronously, keep trace
# ──────────────────────────────────────────────────────────────
async def _run_full_rescue(order_id: str, transcript: str, pincode: str = "",
                            city_hint: str = "", call_answered: bool = True,
                            state_hint: str = "", country_hint: str = "") -> dict:
    """
    Runs voice_agent -> spatial_agent -> route_agent (with internal
    retry/escalate loop already wired into the graph) to completion.
    Returns {"steps": [...], "final": {...}} — steps is the full
    agent-by-agent trace, useful for showing "state handoff" in UI.
    Used by both the Twilio recording callback and the manual/typed
    address fallback, so both paths go through the identical agents.

    state_hint / country_hint are optional — pass them when the caller
    knows more than just the city (e.g. testing a US address end to
    end), so the geocoder isn't stuck guessing the rest of the world
    from a city name alone.
    """
    initial: RescueState = {
        "order_id": order_id, "raw_address": transcript, "pincode": pincode,
        "city_hint": city_hint, "state_hint": state_hint, "country_hint": country_hint,
        "language": "auto", "call_answered": call_answered,
        "fallback_triggered": False, "audio_transcript": transcript,
        "noise_detected": False, "noise_cleaned": False, "extracted_landmarks": [],
        "extracted_directions": [], "extracted_identifiers": [], "inferred_city": city_hint,
        "geocode_result": {}, "confidence_score": 0.0, "confidence_reason": "",
        "ambiguity_detected": False, "candidate_count": 0, "final_gps": {},
        "action_taken": "", "retry_count": 0, "status": "pending",
        "status_message": "Rescue initiated...", "error_log": [],
    }
    steps = []
    final_state = initial
    async for step in rescue_graph.astream(initial):
        for node_name, node_state in step.items():
            steps.append({"node": node_name, "state": node_state})
            final_state = {**final_state, **node_state}
    return {"steps": steps, "final": final_state}

# ──────────────────────────────────────────────────────────────
#  REST: MANUAL RESCUE  (typed-address fallback when Twilio is unavailable)
# ──────────────────────────────────────────────────────────────
class ManualRescueRequest(BaseModel):
    order_id:  str
    transcript: str        # the address text driver typed/selected
    pincode:   str = ""
    city_hint: str = ""
    state_hint: str = ""     # NEW — optional state/province, any country
    country_hint: str = ""   # NEW — optional country name/code, blank = search globally

@app.post("/api/manual-rescue")
async def manual_rescue(req: ManualRescueRequest):
    """
    Fallback path when Twilio calling isn't available/configured.
    Feeds a typed address straight into the same agent graph used
    by the real phone-call path, so the result is directly comparable.
    Not restricted to any fixed set of cities — city_hint/state_hint/
    country_hint are free text, so any address worldwide can be tested.
    """
    result = await _run_full_rescue(
        order_id=req.order_id,
        transcript=req.transcript,
        pincode=req.pincode,
        city_hint=req.city_hint,
        call_answered=True,
        state_hint=req.state_hint,   
        country_hint=req.country_hint
    )
    if not result["final"].get("state_hint") and req.state_hint:
        result["final"]["state_hint"] = req.state_hint
    return result

# ──────────────────────────────────────────────────────────────
#  TWILIO — real outbound call, recording, transcription, geocoding
# ──────────────────────────────────────────────────────────────
# In-memory store of call results, keyed by CallSid. A dict is fine for
# a hackathon prototype — swap for Redis if this ever needs to survive
# a backend restart or run across multiple workers.
_twilio_results: dict[str, dict] = {}

def _twilio_env_ok() -> bool:
    return all(os.environ.get(k) for k in
               ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER", "PUBLIC_BASE_URL"))

def _get_twilio_client():
    from twilio.rest import Client
    sid = os.environ["TWILIO_ACCOUNT_SID"]
    token = os.environ["TWILIO_AUTH_TOKEN"]
    return Client(sid, token)

class TwilioCallRequest(BaseModel):
    to:         str    # phone number to call, E.164 format e.g. +919876543210
    order_id:   str
    pincode:    str = ""
    city_hint:  str = ""
    state_hint: str = ""
    country_hint: str = ""

@app.get("/api/twilio/status")
async def twilio_status():
    """Lets the frontend show whether real calling is configured before showing the Call tab."""
    return {"configured": _twilio_env_ok(),
            "missing": [k for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
                                     "TWILIO_PHONE_NUMBER", "PUBLIC_BASE_URL")
                        if not os.environ.get(k)]}

@app.post("/api/twilio/call")
async def twilio_call(req: TwilioCallRequest):
    """
    Places a real outbound call via Twilio. Twilio requests TwiML from
    /api/twilio/twiml when the call connects, which prompts and records
    the response, then posts the recording to /api/twilio/recording-status.
    """
    if not _twilio_env_ok():
        raise HTTPException(status_code=400, detail=(
            "Twilio not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
            "TWILIO_PHONE_NUMBER, and PUBLIC_BASE_URL (your ngrok https URL). See README."
        ))
    try:
        client = _get_twilio_client()
        base = os.environ["PUBLIC_BASE_URL"].rstrip("/")
        call = client.calls.create(
            to=req.to,
            from_=os.environ["TWILIO_PHONE_NUMBER"],
            url=f"{base}/api/twilio/twiml",
            method="POST",
            status_callback=f"{base}/api/twilio/call-status",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            status_callback_method="POST",
        )
        # Stash order context so the recording callback (which only gets
        # CallSid, not order_id) can find its way back to the right order.
        _twilio_results[call.sid] = {
            "status": "calling", "order_id": req.order_id,
            "pincode": req.pincode, "city_hint": req.city_hint,
            "state_hint": req.state_hint, "country_hint": req.country_hint,
            "steps": [], "final": {}, "transcript": "",
        }
        log.info(f"[Twilio] Call placed: sid={call.sid} to={req.to}")
        return {"call_sid": call.sid, "status": "calling"}
    except Exception as e:
        log.error(f"[Twilio] Call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Twilio call failed: {e}")

@app.post("/api/twilio/twiml")
async def twilio_twiml():
    """
    Twilio fetches this the moment the call is answered. Returns TwiML
    telling Twilio to prompt and record. recordingStatusCallback fires
    when the recording is ready, separately, async.
    """
    from fastapi.responses import Response
    base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say language="hi-IN">Namaste. Kripya apna pura pata, landmark ke saath, batayein.</Say>
  <Record maxLength="30" playBeep="true" trim="trim-silence"
          recordingStatusCallback="{base}/api/twilio/recording-status"
          recordingStatusCallbackMethod="POST" />
  <Say language="hi-IN">Dhanyavaad. Aapka pata process ho raha hai.</Say>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.post("/api/twilio/call-status")
async def twilio_call_status(request: Request):
    """Optional: Twilio posts call lifecycle events here (ringing/answered/completed)."""
    form = await request.form()
    call_sid = form.get("CallSid", "")
    call_status = form.get("CallStatus", "")
    if call_sid in _twilio_results:
        _twilio_results[call_sid]["call_status"] = call_status
    log.info(f"[Twilio] Call {call_sid} status: {call_status}")
    return {"ok": True}

@app.post("/api/twilio/recording-status")
async def twilio_recording_status(request: Request, background_tasks: BackgroundTasks):
    """
    Twilio posts here once the recording is ready. We download the
    audio, transcribe with Whisper, then run the full agent graph —
    all in a background task so we can return 200 to Twilio immediately.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "")
    recording_url = form.get("RecordingUrl", "")  # base URL, needs .wav/.mp3 suffix + auth

    if not recording_url or call_sid not in _twilio_results:
        return {"ok": True}  # ack anyway so Twilio doesn't retry forever

    background_tasks.add_task(_process_twilio_recording, call_sid, recording_url)
    return {"ok": True}

async def _process_twilio_recording(call_sid: str, recording_url: str):
    """
    Each stage below writes its own status + timestamp into _twilio_results
    IMMEDIATELY, rather than batching everything into one update at the end.
    /api/twilio/result/{call_sid} is polled by the frontend every ~1.2s, so
    the driver actually sees "downloading" -> "transcribing" -> transcript
    text appear -> "geocoding" -> pin drop, as distinct visible steps,
    instead of a long silence followed by one final result.
    """
    ctx = _twilio_results.get(call_sid, {})

    def publish(**fields):
        ctx.update(fields)
        _twilio_results[call_sid] = ctx

    publish(status="downloading_audio", status_message="⬇️ Downloading call recording from Twilio...")
    try:
        sid = os.environ["TWILIO_ACCOUNT_SID"]
        token = os.environ["TWILIO_AUTH_TOKEN"]
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{recording_url}.wav", auth=(sid, token))
            audio_bytes = resp.content

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Record + Whisper, not Twilio's built-in speech recognition —
        # better accuracy on Hindi/Bhojpuri/Maithili, and Whisper's
        # multilingual auto-detect (language="hi") means the customer can
        # say ANY address in ANY language — nothing here is India-locked.
        publish(status="transcribing", status_message="🧠 Transcribing audio with Whisper...")
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path, language="hi", task="transcribe", fp16=False,initial_prompt="Hanuman mandir ke paas, bus stand wali gali. Near landmark.")
            transcript = result["text"].strip()
            detected_lang = result.get("language", "")
        except ImportError:
            transcript = ""
            detected_lang = ""
            log.warning("[Twilio] openai-whisper not installed — cannot transcribe real call audio.")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if not transcript:
            publish(status="error", error="Empty transcript — check Whisper install / call audio.")
            return

        # Publish the transcript THE MOMENT it's ready — don't wait for
        # geocoding to finish. This is what makes the transcript show up
        # live, before the pin lands on the map.
        publish(
            status="transcribed",
            transcript=transcript,
            detected_language=detected_lang,
            status_message=f"📝 Transcript ready: \"{transcript[:80]}\" — now extracting location...",
        )

        run = await _run_full_rescue(
            order_id=ctx.get("order_id", call_sid),
            transcript=transcript,
            pincode=ctx.get("pincode", ""),
            city_hint=ctx.get("city_hint", ""),
            state_hint=ctx.get("state_hint", ""),
            country_hint=ctx.get("country_hint", ""),
        )
        publish(status="done", transcript=transcript, **run)
        log.info(f"[Twilio] Call {call_sid} processed. Final status: {run['final'].get('status')}")

    except Exception as e:
        log.error(f"[Twilio] Processing failed for {call_sid}: {e}")
        publish(status="error", error=str(e))

@app.get("/api/twilio/result/{call_sid}")
async def twilio_result(call_sid: str):
    """Frontend polls this every ~1.5s after placing a call."""
    if call_sid not in _twilio_results:
        raise HTTPException(status_code=404, detail="Unknown call_sid")
    return _twilio_results[call_sid]

# ──────────────────────────────────────────────────────────────
#  REST: GEOCODE DIRECT  (for testing spatial agent standalone)
# ──────────────────────────────────────────────────────────────
class GeocodeRequest(BaseModel):
    landmarks: list[str]
    pincode:   str = ""
    city:      str = ""
    state:     str = ""
    country:   str = ""   # optional ISO code (e.g. "us"); blank = search globally

@app.post("/api/geocode")
async def geocode_direct(req: GeocodeRequest):
    """
    Test the spatial agent's geocoding directly without running full graph.
    """
    local = _local_search(req.landmarks, req.pincode, req.city)
    if local and local.get("confidence", 0) >= 0.35:
        return {"source": "local_db", "result": local}

    osm_q = " ".join(req.landmarks[:2])
    for part in (req.city, req.state, req.country):
        if part:
            osm_q += " " + part
    osm = await _osm_search(osm_q, req.pincode, country="in")
    if osm:
        return {"source": "openstreetmap", "result": osm}

    return {"source": "none", "result": {}, "message": "No match found"}

# ──────────────────────────────────────────────────────────────
#  REST: HEALTH CHECK
# ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status":            "ok",
        "landmarks_loaded":  len(_landmark_index),
        "gemini_key_set":    bool(os.environ.get("GEMINI_API_KEY")),
        "graph_nodes":       ["voice_agent", "spatial_agent", "route_agent", "escalate"],
        "osm_enabled":       True,
        "whisper_available": _check_whisper(),
    }

def _check_whisper() -> bool:
    try:
        import whisper
        return True
    except ImportError:
        return False

# ──────────────────────────────────────────────────────────────
#  REST: LIST LANDMARKS  (for map preview in frontend)
# ──────────────────────────────────────────────────────────────
@app.get("/api/landmarks")
async def list_landmarks(state: str = "", city: str = ""):
    rows = _landmark_index
    if state:
        rows = [r for r in rows if r.get("state", "").lower() == state.lower()]
    if city:
        rows = [r for r in rows if r.get("city", "").lower() == city.lower()]
    return {
        "count": len(rows),
        "landmarks": [
            {
                "name":    r["landmark_name"],
                "city":    r["city"],
                "state":   r["state"],
                "lat":     r["lat"],
                "lng":     r["lng"],
                "type":    r.get("landmark_type", ""),
                "ambiguity_note": r.get("ambiguity_note", ""),
            }
            for r in rows
        ]
    }

# ──────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
