"""
=============================================================
  DELIVERY RESCUE BACKEND  —  Agentic AI Last-Mile System
=============================================================
  Stack  : FastAPI · LangGraph · Claude API · Whisper · OSM
  Author : Delivery Rescue Team
  
  HOW TO RUN  (copy-paste into terminal one line at a time)
  ──────────────────────────────────────────────────────────
  python -m venv venv
  source venv/bin/activate          # Windows: venv\Scripts\activate
  pip install -r requirements.txt
  export ANTHROPIC_API_KEY=sk-ant-YOUR-KEY-HERE
  uvicorn backend:app --reload --port 8000
=============================================================
"""

# ──────────────────────────────────────────────────────────────
#  STDLIB
# ──────────────────────────────────────────────────────────────
import os, csv, json, re, math, asyncio, tempfile, time, logging
from pathlib import Path
from typing import Optional, Literal, List

# ──────────────────────────────────────────────────────────────
#  THIRD-PARTY
# ──────────────────────────────────────────────────────────────
import httpx
import anthropic
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing_extensions import TypedDict

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

# Claude client (lazy — only fails if key missing when called)
def get_claude():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY env var not set. See README.")
    return anthropic.Anthropic(api_key=key)

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

def _local_search(landmarks: list[str], pincode: str, city: str) -> dict:
    """
    Generic fuzzy-match against the landmark CSV.
    Returns the best match dict or {} with a confidence score.

    Scoring logic:
      +0.35  exact landmark name token match
      +0.20  alias match
      +0.25  pincode matches
      +0.15  city/district matches
      -0.10  each ambiguity_note mention of 'multiple' or 'ambiguous'
    
    If multiple rows get the same score → ambiguity detected → confidence halved.
    """
    if not _landmark_index:
        return {}

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
        if overlap == 0:
            continue

        # Landmark name match
        lm_name_tokens = set(re.split(r"[\s,/\-]+", row["landmark_name"].lower()))
        lm_overlap = len(query_tokens & lm_name_tokens)
        if lm_overlap > 0:
            score += 0.35 * (lm_overlap / max(len(lm_name_tokens), 1))

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

    # Detect ambiguity: are there multiple rows with close scores?
    close_matches = [s for s, _ in scored if s >= top_score - 0.12]
    ambiguous = len(close_matches) > 1

    if ambiguous:
        top_score *= 0.55   # halve confidence when ambiguous
        log.info(f"Ambiguity detected: {len(close_matches)} close matches. Confidence halved.")

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
    }

async def _osm_search(query: str, pincode: str = "", country: str = "in") -> dict:
    """
    Fallback: query OpenStreetMap Nominatim.
    Returns best result or {} on failure.
    Rate limit: 1 req/sec per OSM policy.
    """
    params = {
        "q": query + (" " + pincode if pincode else "") + " India",
        "format": "jsonv2",
        "limit": 5,
        "countrycodes": country,
        "addressdetails": 1,
        "extratags": 1,
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
        # OSM importance score typically 0.0–1.0 but caps at ~0.7 for rural India
        raw_conf = float(top.get("importance", 0.3))
        osm_conf = min(raw_conf * 0.75, 0.68)  # cap — OSM rural India coverage is patchy
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
      3. Use Claude to extract structured landmark info from natural speech
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
        # In real system: Meta WhatsApp Business API call here
        # For demo: we rely on transcript being provided after fallback
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
    updates["status_message"] = "🧠 Extracting landmarks with Claude AI..."

    # ── Claude LLM extraction ──────────────────────────────────
    try:
        claude = get_claude()
        prompt = f"""You are a delivery address intelligence agent for Indian e-commerce.
A delivery partner is trying to reach a customer in a Tier 2/3 Indian city.
The customer spoke in their local dialect (Hindi, Bhojpuri, Maithili, Awadhi, Marathi, Tamil, etc.)

TRANSCRIPT: "{transcript}"
KNOWN CITY HINT: "{state.get('city_hint', 'unknown')}"
PINCODE: "{state.get('pincode', '')}"

Extract delivery location information from this transcript.
Indian addresses commonly use:
- Temples (mandir, masjid, church, gurudwara)
- Government buildings (panchayat bhawan, block office, thana)
- Schools, hospitals, shops
- Directional words (peeche=behind, aage=ahead, baaju mein=next to, samne=in front)
- Visual identifiers (neeli deewar=blue wall, peela gate=yellow gate, lal makaan=red house)
- Relative position (teesra ghar=third house, doosri gali=second lane)

Return ONLY a valid JSON object, no markdown, no explanation:
{{
  "landmarks": ["primary landmark as spoken", "secondary landmark if mentioned"],
  "directions": ["directional clues exactly as understood"],
  "identifiers": ["color or visual markers"],
  "inferred_city": "best guess of city/town from context or empty string",
  "clarification_needed": true/false,
  "clarification_question": "in simple Hindi, what single question would resolve ambiguity? empty if not needed",
  "confidence_hint": "high/medium/low — your confidence in landmark extraction"
}}"""

        msg = claude.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_resp = msg.content[0].text.strip()

        # Strip markdown fences if present
        raw_resp = re.sub(r"^```(?:json)?\s*", "", raw_resp)
        raw_resp = re.sub(r"\s*```$", "", raw_resp)

        extracted = json.loads(raw_resp)
        landmarks   = extracted.get("landmarks", [])
        directions  = extracted.get("directions", [])
        identifiers = extracted.get("identifiers", [])
        inf_city    = extracted.get("inferred_city", "")
        needs_clarif = extracted.get("clarification_needed", False)

        updates["extracted_landmarks"]   = [l for l in landmarks if l]
        updates["extracted_directions"]  = directions
        updates["extracted_identifiers"] = identifiers
        updates["inferred_city"]         = inf_city or state.get("city_hint", "")
        updates["status_message"]        = (
            f"📍 Extracted: {', '.join(landmarks[:2]) or 'no clear landmark'}"
            + (f" | City inferred: {inf_city}" if inf_city else "")
            + (" | ⚠️ Clarification needed" if needs_clarif else "")
        )

        if needs_clarif and state["retry_count"] < 2:
            # Flag for re-ask — orchestrator will loop back
            updates["status_message"] += f" — Will ask: '{extracted.get('clarification_question', '')}'"

        log.info(f"[{state['order_id']}] LLM extraction OK: landmarks={landmarks}")

    except json.JSONDecodeError as e:
        log.warning(f"[{state['order_id']}] LLM JSON parse error: {e}. Using keyword fallback.")
        # Keyword fallback — works without LLM
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
        updates["status_message"]        = f"🔑 Keyword fallback used. Found: {found}"
        updates["error_log"]             = state["error_log"] + [f"LLM JSON error: {e}"]

    except Exception as e:
        log.error(f"[{state['order_id']}] LLM call failed: {e}")
        updates["extracted_landmarks"]   = [transcript[:60]]
        updates["extracted_directions"]  = []
        updates["extracted_identifiers"] = []
        updates["inferred_city"]         = state.get("city_hint", "")
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
    noise       = state.get("noise_detected", False)
    retry       = state["retry_count"]

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
    local_result = _local_search(landmarks, pincode, city)

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

    osm_query = " ".join(landmarks[:2]) + (" " + city if city else "")
    osm_result = await _osm_search(osm_query, pincode)

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
            language=None,     # auto-detect
            task="transcribe",
            fp16=False,        # CPU-safe
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
#  REST: GEOCODE DIRECT  (for testing spatial agent standalone)
# ──────────────────────────────────────────────────────────────
class GeocodeRequest(BaseModel):
    landmarks: list[str]
    pincode:   str = ""
    city:      str = ""

@app.post("/api/geocode")
async def geocode_direct(req: GeocodeRequest):
    """
    Test the spatial agent's geocoding directly without running full graph.
    Useful for testing with new addresses.
    """
    local = _local_search(req.landmarks, req.pincode, req.city)
    if local and local.get("confidence", 0) >= 0.35:
        return {"source": "local_db", "result": local}

    osm_q = " ".join(req.landmarks[:2]) + (" " + req.city if req.city else "")
    osm = await _osm_search(osm_q, req.pincode)
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
        "claude_key_set":    bool(os.environ.get("ANTHROPIC_API_KEY")),
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
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )