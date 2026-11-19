from __future__ import annotations

import math
import os
from typing import Any

import cv2

from detector import FootballDetector


CLASS_NAMES = {0: "ball", 1: "player", 2: "referee"}


def _distance(a, b):
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _track_detections(prev_tracks, detections, next_id):
    tracks = []
    used = set()
    max_dist = float(os.getenv("TACTICAI_TRACK_MAX_PX", "90"))

    for det in detections:
        best = None
        best_dist = max_dist
        for tr in prev_tracks:
            if tr["id"] in used or tr["class_id"] != det["class_id"]:
                continue
            d = _distance(tr, det)
            if d < best_dist:
                best_dist = d
                best = tr

        if best is not None:
            tid = best["id"]
            used.add(tid)
        else:
            tid = next_id
            next_id += 1

        tracks.append({
            **det,
            "id": tid,
            "movement_px": round(best_dist if best is not None else 0.0, 2),
        })

    return tracks, next_id


def _zone(x, y, width, height):
    nx = max(0.0, min(0.999, x / max(width, 1)))
    ny = max(0.0, min(0.999, y / max(height, 1)))
    horizontal = "left" if ny < 1 / 3 else "central" if ny < 2 / 3 else "right"
    vertical = "defensive" if nx < 1 / 3 else "middle" if nx < 2 / 3 else "attacking"
    return f"{vertical}_{horizontal}"


def build_analysis(path: str, sample_fps: float, team: str, opp: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else 0.0
    step = max(1, int(round(fps / max(sample_fps, 0.1))))

    detector = FootballDetector()
    detections = []
    tracks = []
    prev_tracks = []
    next_id = 1
    sampled = 0
    first_size = [0, 0]
    frame_index = 0

    while frame_index < total_frames and frame_index / fps <= float(os.getenv("TACTICAI_MAX_SECONDS", "7200")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        first_size = [w, h]
        ds = detector.detect(frame) if detector.available else []
        ts, next_id = _track_detections(prev_tracks, ds, next_id)
        prev_tracks = ts

        players = [d for d in ts if d["class_id"] == 1]
        balls = [d for d in ts if d["class_id"] == 0]
        referees = [d for d in ts if d["class_id"] == 2]

        detections.append({
            "timestamp": round(frame_index / fps, 3),
            "frame": frame_index,
            "players": len(players),
            "balls": len(balls),
            "referees": len(referees),
            "detections": ts,
        })

        tracks.extend([
            {
                "timestamp": round(frame_index / fps, 3),
                "id": d["id"],
                "class": CLASS_NAMES.get(d["class_id"], str(d["class_id"])),
                "confidence": round(d["confidence"], 3),
                "x_px": round(d["x"], 1),
                "y_px": round(d["y"], 1),
                "movement_px": d["movement_px"],
                "zone": _zone(d["x"], d["y"], w, h),
            }
            for d in ts
        ])

        sampled += 1
        frame_index += step

    cap.release()

    events = []
    possession_candidates = []
    visual_facts = []

    for i, cur in enumerate(detections):
        t = cur["timestamp"]

        if cur["players"] > 0:
            visual_facts.append({
                "timestamp": t,
                "type": "players_detected",
                "status": "candidate",
                "evidence_level": "video_detection",
                "count": cur["players"],
            })

        if cur["players"] >= 2 and cur["balls"]:
            possession_candidates.append({
                "timestamp": t,
                "status": "candidate",
                "evidence_level": "ball_player_proximity",
                "reason": "Balón y jugadores detectados en el mismo frame; no implica posesión confirmada.",
            })

        if i:
            prev = detections[i - 1]
            dt = max(0.001, t - prev["timestamp"])
            player_delta = cur["players"] - prev["players"]

            if player_delta != 0:
                events.append({
                    "timestamp": t,
                    "start": max(0.0, t - 1.0),
                    "end": min(duration, t + 2.0),
                    "type": "player_count_change",
                    "status": "candidate",
                    "confidence": 0.35,
                    "evidence_level": "video_detection",
                    "reason": f"El detector pasó de {prev['players']} a {cur['players']} jugadores detectados.",
                })

            moving = [d for d in cur["detections"] if d["class_id"] == 1 and d["movement_px"] > 20]
            if moving:
                events.append({
                    "timestamp": t,
                    "start": max(0.0, t - 1.0),
                    "end": min(duration, t + 2.0),
                    "type": "player_movement_candidate",
                    "status": "candidate",
                    "confidence": 0.25,
                    "evidence_level": "video_tracking",
                    "reason": f"{len(moving)} jugadores muestran desplazamiento entre muestras ({dt:.2f}s).",
                })

    # Keep the evidence useful but bounded.
    events = events[:500]
    possession_candidates = possession_candidates[:500]
    visual_facts = visual_facts[:500]

    clips = [
        {
            "id": f"clip-{i+1}",
            "title": e["type"],
            "type": e["type"],
            "timestamp": e["timestamp"],
            "start": e["start"],
            "end": e["end"],
            "status": e["status"],
            "reason": e["reason"],
        }
        for i, e in enumerate(events[:100])
    ]

    return {
        "match": {"team": team, "opponent": opp},
        "duration": duration,
        "metrics": {
            "frames_sampled": sampled,
            "fps": fps,
            "source_frames": total_frames,
            "detections": sum(len(x["detections"]) for x in detections),
            "players_detected_frames": sum(x["players"] > 0 for x in detections),
            "ball_detected_frames": sum(x["balls"] > 0 for x in detections),
            "referee_detected_frames": sum(x["referees"] > 0 for x in detections),
        },
        "capability": {
            "video_decoded": True,
            "football_detector_loaded": detector.available,
            "football_events_confirmed": False,
            "notes": (
                "Detección y tracking generan evidencia candidata. "
                "No se consideran posesiones, pérdidas o acciones tácticas confirmadas sin validación temporal adicional."
            ),
        },
        "events": events,
        "clips": clips,
        "evidence_ledger": clips,
        "sequence": {
            "windows": detections[:500],
            "tracks": tracks[:5000],
            "possession_candidates": possession_candidates,
            "visual_facts": visual_facts,
        },
    }


def build_sequence_features(samples):
    return {"frames": len(samples), "tracks": [], "windows": samples}


def event_candidates(sequence):
    return []


def derive_football_sequences(samples, tracks):
    return []


def build_football_event_summary(events):
    return {"candidate_counts": {}, "confirmed_counts": {}}
