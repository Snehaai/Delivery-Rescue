"""
=============================================================
  DELIVERY RESCUE BACKEND  —  Agentic AI Last-Mile System
=============================================================
  Stack  : FastAPI · LangGraph · Groq API (Llama 3) · Whisper · OSM · Twilio
  Author : Delivery Rescue Team
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
from groq import Groq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing_extensions import TypedDict

# Load .env file automatically if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# LangGraph
from langgraph.graph import StateGraph, END

# ──────────────────────────────────────────────────────────────
#  LOGGING & SECRET MASKING
# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def mask_secrets(text: str) -> str:
    """Utility to scrub sensitive environment variables and API keys from log strings."""
    if not isinstance(text, str):
        text = str(text)
    for env_var in ["GROQ_API_KEY", "LOCATIONIQ_API_KEY", "TWILIO_AUTH_TOKEN", "TWILIO_ACCOUNT_SID"]:
        secret_val = os.getenv(env_var, "")
        if secret_val and len(secret_val) > 4:
            text = text.replace(secret_val, f"{secret_val[:4]}...[REDACTED]")
    return text

# ──────────────────────────────────────────────────────────────
#  CIRCUIT BREAKER IMPLEMENTATION FOR GROQ API
# ──────────────────────────────────────────────────────────────
class GroqCircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_time: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN
        self.last_state_change = 0.0

    def can_execute(self) -> bool:
        if self.state == "OPEN":
            if time.time() - self.last_state_change > self.recovery_time:
                log.info("[CircuitBreaker] Entering HALF-OPEN state. Testing Groq API...")
                return True
            return False
        return True

    def record_success(self):
        if self.state != "CLOSED":
            log.info("[CircuitBreaker] Groq API recovered. Closing Circuit Breaker.")
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.last_state_change = time.time()
            log.warning(f"[CircuitBreaker] Groq API failed {self.failure_count} times. Opening breaker for {self.recovery_time}s.")

groq_breaker = GroqCircuitBreaker()

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

def get_groq():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not found.")
    return Groq(api_key=key)

# ──────────────────────────────────────────────────────────────
#  LANDMARK DATABASE
# ──────────────────────────────────────────────────────────────
LANDMARK_CSV = Path(__file__).parent / "landmarks.csv"
_landmark_index: list[dict] = []

def load_landmarks() -> list[dict]:
    if not LANDMARK_CSV.exists():
        log.warning(f"landmarks.csv not found at {LANDMARK_CSV}. Using empty DB.")
        return []

    rows = []
    with open(LANDMARK_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["lat"] = float(row["lat"])
                row["lng"] = float(row["lng"])
            except (ValueError, KeyError):
                continue

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
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = math.sin(d_lat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(d_lng/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

_HINT_TO_NUMERIC = {"high": 0.90, "medium": 0.60, "low": 0.30}

def _fuzzy_ratio(query_landmarks: list[str], row: dict) -> float:
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

        # Ambiguity penalty
        note = row.get("ambiguity_note", "").lower()
        if any(w in note for w in ["multiple", "ambiguous", "different", "3 ", "4 ", "several"]):
            score -= 0.10

        if score > 0.15:
            scored.append((score, row))

    if not scored:
        return {}

    scored.sort(key=lambda x: x[0], reverse=True)
    top_score, top_row = scored[0]

    close_matches = [s for s, _ in scored if s >= top_score - 0.12]
    ambiguous = len(close_matches) > 1

    hint_val = _HINT_TO_NUMERIC.get(extraction_hint, 0.6)
    clue_bonus = min(0.02 * (len(directions) + len(identifiers)), 0.08)
    top_score = 0.75 * top_score + 0.15 * hint_val + clue_bonus

    candidates = []
    if ambiguous:
        top_score *= 0.55
        log.info(f"Ambiguity detected: {len(close_matches)} close matches. Confidence halved.")
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
    global _nominatim_last_call
    async with _nominatim_lock:
        elapsed = time.monotonic() - _nominatim_last_call
        if elapsed < 1.05:
            await asyncio.sleep(1.05 - elapsed)
        _nominatim_last_call = time.monotonic()

async def _locationiq_search(query: str, pincode: str = "", country: str = "") -> dict:
    key = os.environ.get("LOCATIONIQ_API_KEY", "")
    if not key:
        return {}
    params = {
        "key": key,
        "q": query + (f", {pincode}" if pincode else ""),
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
            "confidence": round(max(raw_conf, 0.70), 3),
            "total_matches": len(results),
            "ambiguous": len(results) > 2,
            "osm_class": top.get("class", ""),
            "osm_type": top.get("type", ""),
        }
    except Exception as e:
        log.warning(f"LocationIQ search failed: {mask_secrets(str(e))}")
        return {}

async def _osm_search(query: str, pincode: str = "", country: str = "in") -> dict:
    liq = await _locationiq_search(query, pincode, country="in")
    if liq:
        return liq

    await _nominatim_rate_limit()

    # Build clean structured query with commas for OpenStreetMap
    query_parts = [p.strip() for p in query.split(",") if p and p.strip()]
    if pincode and pincode.strip() not in query_parts:
        query_parts.append(pincode.strip())
    
    search_query = ", ".join(query_parts)

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
        raw_conf = float(top.get("importance", 0.35))
        
        # Give strong default baseline for valid OSM hits
        osm_conf = max(raw_conf * 1.2, 0.75)
        
        return {
            "lat": float(top["lat"]),
            "lng": float(top["lon"]),
            "display_name": top.get("display_name", "")[:160],
            "source": "openstreetmap",
            "confidence": round(min(osm_conf, 0.90), 3),
            "total_matches": len(results),
            "ambiguous": len(results) > 3,
            "osm_class": top.get("class", ""),
            "osm_type": top.get("type", ""),
        }
    except Exception as e:
        log.warning(f"OSM search failed: {mask_secrets(str(e))}")
        return {}

# ──────────────────────────────────────────────────────────────
#  SHARED STATE SCHEMA
# ──────────────────────────────────────────────────────────────
class RescueState(TypedDict):
    order_id:               str
    raw_address:            str
    pincode:                str
    city_hint:              str
    state_hint:             str
    country_hint:           str
    language:               str

    call_answered:          bool
    fallback_triggered:      bool
    audio_transcript:       str
    noise_detected:         bool
    noise_cleaned:          bool
    extracted_landmarks:    list
    extracted_directions:   list
    extracted_identifiers:  list
    inferred_city:          str
    extraction_confidence_hint: str

    geocode_result:         dict
    confidence_score:       float
    confidence_reason:      str
    ambiguity_detected:     bool
    candidate_count:        int

    final_gps:              dict
    action_taken:           str

    retry_count:            int
    status:                 str
    status_message:         str
    error_log:              list

# ──────────────────────────────────────────────────────────────
#  NODE 1  —  VOICE AGENT
# ──────────────────────────────────────────────────────────────
async def voice_agent(state: RescueState) -> dict:
    updates: dict = {"status": "voice_processing"}

    if state["retry_count"] >= 3:
        return {
            **updates,
            "status": "escalated",
            "status_message": "❌ 3 voice attempts exhausted. Escalating to ops.",
            "action_taken": "escalate",
        }

    transcript = state.get("audio_transcript", "").strip()
    call_answered = state.get("call_answered", True)

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

    if len(transcript.strip()) < 3:
        return {
            **updates,
            "retry_count": state["retry_count"] + 1,
            "status_message": "🔄 Transcript too short. Re-prompting customer for landmark...",
            "error_log": state["error_log"] + [f"Retry {state['retry_count']+1}: transcript under 3 chars"],
        }

    updates["audio_transcript"] = transcript
    updates["status_message"] = "🧠 Extracting landmarks..."

    if not groq_breaker.can_execute():
        log.warning(f"[{state['order_id']}] Groq Circuit Breaker is OPEN. Bypassing LLM call.")
        return _apply_keyword_fallback(state, updates, transcript, "Circuit Breaker OPEN")

    try:
        groq_client = get_groq()
        prompt = f"""You are a delivery address intelligence agent for last-mile e-commerce.
Extract delivery location information from this transcript:
TRANSCRIPT: "{transcript}"
KNOWN CITY HINT: "{state.get('city_hint', 'unknown')}"
KNOWN STATE HINT: "{state.get('state_hint', 'unknown')}"
KNOWN COUNTRY HINT: "{state.get('country_hint', 'unknown')}"
PINCODE / ZIP: "{state.get('pincode', '')}"

Return ONLY a valid JSON object, no markdown, no explanation:
{{
  "landmarks": ["primary landmark or street address as spoken", "secondary detail if mentioned"],
  "directions": ["directional clues if any"],
  "identifiers": ["color/visual markers, apartment/unit numbers, etc if any"],
  "inferred_city": "best guess of city/town from context or empty string",
  "inferred_state": "best guess of state/province from context or empty string",
  "inferred_country": "best guess of country or empty string",
  "clarification_needed": false,
  "clarification_question": "one short question that would resolve ambiguity; empty if not needed",
  "confidence_hint": "high/medium/low"
}}"""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a precise JSON-only delivery location extractor. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        raw_resp = response.choices[0].message.content.strip()
        extracted = json.loads(raw_resp)

        groq_breaker.record_success()

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

        updates["state_hint"] = state.get("state_hint") or inf_state
        updates["country_hint"] = state.get("country_hint") or inf_country
        updates["inferred_city"] = inf_city or state.get("city_hint", "")

        updates["extraction_confidence_hint"] = conf_hint
        updates["status_message"]         = (
            f"📍 Extracted: {', '.join(landmarks[:2]) or transcript}"
            + (f" | City inferred: {updates['inferred_city']}" if updates['inferred_city'] else "")
            + (" | ⚠️ Clarification needed" if needs_clarif else "")
        )

        log.info(f"[{state['order_id']}] Groq LLM extraction OK: landmarks={landmarks}, conf_hint={conf_hint}")

    except Exception as e:
        groq_breaker.record_failure()
        safe_error = mask_secrets(str(e))
        log.error(f"[{state['order_id']}] Groq LLM call failed: {safe_error}. Using keyword fallback.")
        return _apply_keyword_fallback(state, updates, transcript, safe_error)

    return updates

def _apply_keyword_fallback(state: RescueState, updates: dict, transcript: str, err_msg: str) -> dict:
    kw_map = {
        "mandir": "mandir", "masjid": "masjid", "school": "school",
        "bhawan": "bhawan", "dukaan": "store", "station": "station",
        "bazar": "bazar", "chowk": "chowk", "hospital": "hospital",
        "thana": "police station", "panchayat": "panchayat bhawan",
        "valley": "valley", "villa": "villa", "society": "society",
    }
    found = []
    for word in transcript.lower().split():
        for kw, label in kw_map.items():
            if kw in word and label not in found:
                found.append(label)

    updates["extracted_landmarks"]   = [transcript.strip()] if not found else found
    updates["extracted_directions"]  = []
    updates["extracted_identifiers"] = []
    updates["inferred_city"]         = state.get("city_hint", "")
    updates["state_hint"]            = state.get("state_hint", "")
    updates["country_hint"]          = state.get("country_hint", "")
    updates["extraction_confidence_hint"] = "medium"
    updates["status_message"]         = f"🔑 Keyword fallback used. Extracted: {updates['extracted_landmarks']}"
    updates["error_log"]              = state["error_log"] + [f"LLM error: {err_msg[:80]}"]
    return updates

# ──────────────────────────────────────────────────────────────
#  NODE 2  —  SPATIAL AGENT
# ──────────────────────────────────────────────────────────────
async def spatial_agent(state: RescueState) -> dict:
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
        landmarks = [state.get("audio_transcript", "")]

    # 1. Local Search First
    local_result = _local_search(landmarks, pincode, city, directions, identifiers, conf_hint)

    if local_result and local_result.get("confidence", 0) >= 0.35:
        conf = local_result["confidence"]
        if noise:
            conf *= 0.90
        if local_result.get("ambiguous"):
            conf = min(conf, 0.55)
            reason = f"Ambiguous: {local_result.get('total_matches', '?')} candidates with similar names."
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
        return updates

    # 2. OSM Search with Structured Comma-Separated Format
    query_parts = []
    if landmarks:
        query_parts.append(", ".join(landmarks))
    if city:
        query_parts.append(city)
    if state_hint:
        query_parts.append(state_hint)
    if country_hint:
        query_parts.append(country_hint)

    osm_query = ", ".join([p for p in query_parts if p.strip()])

    osm_result = await _osm_search(osm_query, pincode, country="in")

    if osm_result:
        conf = osm_result.get("confidence", 0.70)
        
        # SMART BOOST: Check if primary landmark tokens exist in display_name
        osm_display = osm_result.get("display_name", "").lower()
        matched_landmarks = [lm for lm in landmarks if lm.lower().strip() in osm_display]
        
        if matched_landmarks:
            conf = max(conf, 0.85) # High confidence when primary landmark exists in output

        if noise:
            conf *= 0.90

        osm_result["confidence"] = round(conf, 3)

        updates.update({
            "geocode_result":    osm_result,
            "confidence_score":  round(conf, 3),
            "confidence_reason": f"OpenStreetMap match for {osm_query}",
            "ambiguity_detected": osm_result.get("ambiguous", False),
            "candidate_count":   osm_result.get("total_matches", 1),
            "status_message":    f"🌐 OSM match: {osm_result.get('display_name','')[:60]} — {conf:.0%}",
        })
        return updates

    return {
        **updates,
        "geocode_result":    {},
        "confidence_score":  0.0,
        "confidence_reason": "No match in local DB or OSM",
        "ambiguity_detected": True,
        "candidate_count":   0,
        "status_message":    "❓ Location not found.",
        "error_log":          state["error_log"] + ["Spatial: no result"],
    }

# ──────────────────────────────────────────────────────────────
#  NODE 3  —  ROUTE AGENT
# ──────────────────────────────────────────────────────────────
async def route_agent(state: RescueState) -> dict:
    score  = state.get("confidence_score", 0.0)
    geo    = state.get("geocode_result", {})
    retry  = state["retry_count"]

    updates: dict = {"status": "route_processing"}

    if retry >= 3:
        return {
            **updates,
            "status": "escalated",
            "status_message": "🧑‍💼 Max retries reached. Context sent to human ops.",
            "action_taken": "escalate",
            "final_gps": {},
        }

    if score >= 0.75 and geo:
        final = {
            "lat": geo["lat"], "lng": geo["lng"],
            "display_name": geo.get("display_name", ""),
            "confidence": score, "flagged_for_review": False,
            "action": "auto_push", "source": geo.get("source", ""),
        }
        return {
            **updates,
            "status": "resolved",
            "status_message": f"✅ Auto-pushed GPS to driver app. ({score:.0%})",
            "action_taken": "auto_push",
            "final_gps": final,
        }

    if 0.50 <= score < 0.75 and geo:
        final = {
            "lat": geo["lat"], "lng": geo["lng"],
            "display_name": geo.get("display_name", ""),
            "confidence": score, "flagged_for_review": True,
            "action": "push_flagged", "source": geo.get("source", ""),
        }
        return {
            **updates,
            "status": "resolved",
            "status_message": f"⚠️ Pushed with caution flag ({score:.0%}).",
            "action_taken": "push_flagged",
            "final_gps": final,
        }

    if retry < 3:
        return {
            **updates,
            "status": "retrying",
            "status_message": f"🔄 Confidence {score:.0%} low. Retrying...",
            "action_taken": "retry",
            "retry_count": retry + 1,
            "final_gps": {},
        }

    return {
        **updates,
        "status": "escalated",
        "status_message": "🧑‍💼 Escalating to human ops.",
        "action_taken": "escalate",
        "final_gps": {},
    }

# ──────────────────────────────────────────────────────────────
#  NODE 4  —  ESCALATION NODE
# ──────────────────────────────────────────────────────────────
async def escalate_node(state: RescueState) -> dict:
    return {
        "status": "escalated",
        "status_message": "🧑‍💼 Ticket created. Human ops team notified.",
        "action_taken": "escalate",
        "final_gps": {},
    }

# ──────────────────────────────────────────────────────────────
#  CONDITIONAL EDGE — orchestrator routing logic
# ──────────────────────────────────────────────────────────────
def routing_decision(state: RescueState) -> str:
    action  = state.get("action_taken", "")
    status  = state.get("status", "")
    retry   = state.get("retry_count", 0)

    if status in ("resolved",):
        return "end"
    if status == "escalated" or action == "escalate" or retry >= 3:
        return "escalate"
    if action == "retry" and retry < 3:
        return "retry_voice"
    return "end"

# ──────────────────────────────────────────────────────────────
#  BUILD LANGGRAPH
# ──────────────────────────────────────────────────────────────
def build_graph() -> any:
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

# ──────────────────────────────────────────────────────────────
#  SHARED HELPER WITH REQUEST TIMEOUT ENFORCEMENT
# ──────────────────────────────────────────────────────────────
async def _run_full_rescue(order_id: str, transcript: str, pincode: str = "",
                            city_hint: str = "", call_answered: bool = True,
                            state_hint: str = "", country_hint: str = "") -> dict:
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

    async def _execute():
        nonlocal final_state
        async for step in rescue_graph.astream(initial):
            for node_name, node_state in step.items():
                steps.append({"node": node_name, "state": node_state})
                final_state = {**final_state, **node_state}

    try:
        await asyncio.wait_for(_execute(), timeout=12.0)
    except asyncio.TimeoutError:
        log.error(f"[{order_id}] Rescue execution timed out after 12s!")
        final_state.update({
            "status": "escalated",
            "status_message": "⏱️ Request timed out. Escalating to human ops.",
            "action_taken": "escalate"
        })

    return {"steps": steps, "final": final_state}

# ──────────────────────────────────────────────────────────────
#  WEBSOCKET ENDPOINT
# ──────────────────────────────────────────────────────────────
@app.websocket("/ws/rescue/{order_id}")
async def rescue_ws(ws: WebSocket, order_id: str):
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

        async for step in rescue_graph.astream(initial):
            for node_name, node_state in step.items():
                await ws.send_json({
                    "event": "step",
                    "node":  node_name,
                    "state": node_state,
                })
                await asyncio.sleep(0.1)

        await ws.send_json({"event": "complete"})
        log.info(f"[WS] Rescue complete: order={order_id}")

    except asyncio.TimeoutError:
        await ws.send_json({"event": "error", "message": "Timeout waiting for initial data"})
    except WebSocketDisconnect:
        log.info(f"[WS] Client disconnected: order={order_id}")
    except Exception as e:
        log.error(f"[WS] Error for order={order_id}: {mask_secrets(str(e))}")
        try:
            await ws.send_json({"event": "error", "message": mask_secrets(str(e))})
        except Exception:
            pass

# ──────────────────────────────────────────────────────────────
#  REST: TRANSCRIBE AUDIO (Whisper)
# ──────────────────────────────────────────────────────────────
@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        import whisper

        content = await audio.read()
        suffix = Path(audio.filename or "audio.webm").suffix or ".webm"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        model = whisper.load_model("base")
        result = model.transcribe(
            tmp_path,
            language="hi",
            task="transcribe",
            fp16=False,
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
        log.error(f"Transcription error: {mask_secrets(str(e))}")
        raise HTTPException(status_code=500, detail=mask_secrets(str(e)))

# ──────────────────────────────────────────────────────────────
#  REST: MOCK CALL
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
    await asyncio.sleep(1.2)
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
#  REST: MANUAL RESCUE
# ──────────────────────────────────────────────────────────────
class ManualRescueRequest(BaseModel):
    order_id:   str
    transcript: str
    pincode:    str = ""
    city_hint:  str = ""
    state_hint: str = ""
    country_hint: str = ""

@app.post("/api/manual-rescue")
async def manual_rescue(req: ManualRescueRequest):
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
#  TWILIO ENDPOINTS
# ──────────────────────────────────────────────────────────────
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
    to:         str
    order_id:   str
    pincode:    str = ""
    city_hint:  str = ""
    state_hint: str = ""
    country_hint: str = ""

@app.get("/api/twilio/status")
async def twilio_status():
    return {"configured": _twilio_env_ok(),
            "missing": [k for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
                                    "TWILIO_PHONE_NUMBER", "PUBLIC_BASE_URL")
                        if not os.environ.get(k)]}

@app.post("/api/twilio/call")
async def twilio_call(req: TwilioCallRequest):
    if not _twilio_env_ok():
        raise HTTPException(status_code=400, detail=(
            "Twilio not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
            "TWILIO_PHONE_NUMBER, and PUBLIC_BASE_URL (your ngrok https URL)."
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
        _twilio_results[call.sid] = {
            "status": "calling", "order_id": req.order_id,
            "pincode": req.pincode, "city_hint": req.city_hint,
            "state_hint": req.state_hint, "country_hint": req.country_hint,
            "steps": [], "final": {}, "transcript": "",
        }
        log.info(f"[Twilio] Call placed: sid={call.sid} to={req.to}")
        return {"call_sid": call.sid, "status": "calling"}
    except Exception as e:
        log.error(f"[Twilio] Call failed: {mask_secrets(str(e))}")
        raise HTTPException(status_code=500, detail=f"Twilio call failed: {mask_secrets(str(e))}")

@app.post("/api/twilio/twiml")
async def twilio_twiml():
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
    form = await request.form()
    call_sid = form.get("CallSid", "")
    call_status = form.get("CallStatus", "")
    if call_sid in _twilio_results:
        _twilio_results[call_sid]["call_status"] = call_status
    log.info(f"[Twilio] Call {call_sid} status: {call_status}")
    return {"ok": True}

@app.post("/api/twilio/recording-status")
async def twilio_recording_status(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    call_sid = form.get("CallSid", "")
    recording_url = form.get("RecordingUrl", "")

    if not recording_url or call_sid not in _twilio_results:
        return {"ok": True}

    background_tasks.add_task(_process_twilio_recording, call_sid, recording_url)
    return {"ok": True}

async def _process_twilio_recording(call_sid: str, recording_url: str):
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

        publish(status="transcribing", status_message="🧠 Transcribing audio with Whisper...")
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path, language="hi", task="transcribe", fp16=False, initial_prompt="Hanuman mandir ke paas, bus stand wali gali. Near landmark.")
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
        log.error(f"[Twilio] Processing failed for {call_sid}: {mask_secrets(str(e))}")
        publish(status="error", error=mask_secrets(str(e)))

@app.get("/api/twilio/result/{call_sid}")
async def twilio_result(call_sid: str):
    if call_sid not in _twilio_results:
        raise HTTPException(status_code=404, detail="Unknown call_sid")
    return _twilio_results[call_sid]

# ──────────────────────────────────────────────────────────────
#  REST: GEOCODE DIRECT
# ──────────────────────────────────────────────────────────────
class GeocodeRequest(BaseModel):
    landmarks: list[str]
    pincode:   str = ""
    city:      str = ""
    state:     str = ""
    country:   str = ""

@app.post("/api/geocode")
async def geocode_direct(req: GeocodeRequest):
    local = _local_search(req.landmarks, req.pincode, req.city)
    if local and local.get("confidence", 0) >= 0.35:
        return {"source": "local_db", "result": local}

    query_parts = [", ".join(req.landmarks)]
    for part in (req.city, req.state, req.country):
        if part:
            query_parts.append(part)
    osm_q = ", ".join(query_parts)

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
        "groq_key_set":      bool(os.environ.get("GROQ_API_KEY")),
        "circuit_breaker":   groq_breaker.state,
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
#  REST: LIST LANDMARKS
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
                "name":     r["landmark_name"],
                "city":     r["city"],
                "state":    r["state"],
                "lat":      r["lat"],
                "lng":      r["lng"],
                "type":     r.get("landmark_type", ""),
                "ambiguity_note": r.get("ambiguity_note", ""),
            }
            for r in rows
        ]
    }

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>Delivery Rescue API Backend Running</h1>", status_code=200)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# ──────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=False, log_level="info")
