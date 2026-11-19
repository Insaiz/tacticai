from __future__ import annotations

import math
import os
from typing import Any

import cv2
import numpy as np

from detector import FootballDetector


CLASS_NAMES = {0: "ball", 1: "player", 2: "referee"}


def _distance(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _team_color(frame, det: dict):
    x, y, w, h = det["x"], det["y"], det["w"], det["h"]
    x1 = max(0, int(x - w * 0.32)); x2 = min(frame.shape[1], int(x + w * 0.32))
    y1 = max(0, int(y - h * 0.42)); y2 = min(frame.shape[0], int(y - h * 0.02))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 45) & (hsv[:, :, 2] < 245)
    if int(mask.sum()) < 20:
        return None
    pixels = hsv[mask]
    ang = pixels[:, 0].astype(float) * math.pi / 90.0
    hue = math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean())) * 90.0 / math.pi
    if hue < 0:
        hue += 180.0
    return [hue / 180.0, float(pixels[:, 1].mean()) / 255.0, float(pixels[:, 2].mean()) / 255.0]

def _assign_teams(players: list[dict], frame) -> list[dict]:
    vectors, valid = [], []
    for p in players:
        v = _team_color(frame, p)
        if v is not None:
            vectors.append(v); valid.append(p)
    if len(vectors) < 2:
        return [{**p, "team": "unknown"} for p in players]
    best_i, best_j, best_d = 0, 1, -1.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            d = float(np.linalg.norm(np.asarray(vectors[i]) - np.asarray(vectors[j])))
            if d > best_d:
                best_i, best_j, best_d = i, j, d
    a, b = np.asarray(vectors[best_i]), np.asarray(vectors[best_j])
    labels = [0] * len(vectors)
    for _ in range(5):
        labels = [0 if np.linalg.norm(np.asarray(v)-a) <= np.linalg.norm(np.asarray(v)-b) else 1 for v in vectors]
        for k in (0, 1):
            pts = [np.asarray(vectors[i]) for i, lab in enumerate(labels) if lab == k]
            if pts:
                if k == 0: a = np.mean(pts, axis=0)
                else: b = np.mean(pts, axis=0)
    by_id = {p["id"]: ("team_a" if labels[i] == 0 else "team_b") for i, p in enumerate(valid)}
    return [{**p, "team": by_id.get(p["id"], "unknown")} for p in players]

def _zone(x: float, y: float, width: int, height: int) -> str:
    nx = max(0.0, min(0.999, x / max(width, 1)))
    ny = max(0.0, min(0.999, y / max(height, 1)))
    lane = "left" if ny < 1 / 3 else "central" if ny < 2 / 3 else "right"
    depth = "defensive" if nx < 1 / 3 else "middle" if nx < 2 / 3 else "attacking"
    return f"{depth}_{lane}"


def _nearest_player(ball: dict | None, players: list[dict], max_distance: float = 180.0):
    if not ball or not players:
        return None, None
    ranked = sorted((_distance(ball, p), p) for p in players)
    d, p = ranked[0]
    return (p, d) if d <= max_distance else (None, None)


def _track_detections(prev_tracks: list[dict], detections: list[dict], next_id: int):
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
                best_dist, best = d, tr

        tid = best["id"] if best is not None else next_id
        if best is not None:
            used.add(tid)
        else:
            next_id += 1

        tracks.append({
            **det,
            "id": tid,
            "movement_px": round(best_dist if best is not None else 0.0, 2),
        })
    return tracks, next_id


def _event(ts: float, duration: float, typ: str, reason: str, confidence: float,
           evidence_level: str, extra: dict | None = None) -> dict:
    e = {
        "timestamp": round(ts, 3),
        "start": round(max(0.0, ts - 2.0), 3),
        "end": round(min(duration, ts + 3.0), 3),
        "type": typ,
        "status": "candidate",
        "confidence": confidence,
        "evidence_level": evidence_level,
        "reason": reason,
    }
    if extra:
        e.update(extra)
    return e


def build_analysis(path: str, sample_fps: float, team: str, opp: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else 0.0
    max_seconds = float(os.getenv("TACTICAI_MAX_SECONDS", "7200"))
    step = max(1, int(round(fps / max(sample_fps, 0.1))))

    detector = FootballDetector()
    detections: list[dict] = []
    tracks_flat: list[dict] = []
    prev_tracks: list[dict] = []
    next_id = 1
    sampled = 0
    width = height = 0
    frame_index = 0

    while frame_index < total_frames and frame_index / fps <= max_seconds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break

        height, width = frame.shape[:2]
        raw = detector.detect(frame) if detector.available else []
        current, next_id = _track_detections(prev_tracks, raw, next_id)
        prev_tracks = current

        players = [d for d in current if d["class_id"] == 1]
        players = _assign_teams(players, frame)
        for p in players:
            for d in current:
                if d["id"] == p["id"]:
                    d["team"] = p.get("team", "unknown")
                    break
        balls = [d for d in current if d["class_id"] == 0]
        referees = [d for d in current if d["class_id"] == 2]
        ball = max(balls, key=lambda x: x["confidence"], default=None)
        nearest, ball_distance = _nearest_player(ball, players)

        frame_data = {
            "timestamp": round(frame_index / fps, 3),
            "frame": frame_index,
            "players": len(players),
            "balls": len(balls),
            "referees": len(referees),
            "ball": ball,
            "nearest_player": (
                {"id": nearest["id"], "distance_px": round(ball_distance, 1)}
                if nearest is not None else None
            ),
            "detections": current,
        }
        detections.append(frame_data)

        for d in current:
            tracks_flat.append({
                "timestamp": frame_data["timestamp"],
                "id": d["id"],
                "class": CLASS_NAMES.get(d["class_id"], str(d["class_id"])),
                "confidence": round(d["confidence"], 3),
                "x_px": round(d["x"], 1),
                "y_px": round(d["y"], 1),
                "movement_px": d["movement_px"],
                "zone": _zone(d["x"], d["y"], width, height),
                "team": d.get("team", "unknown"),
            })

        sampled += 1
        frame_index += step

    cap.release()

    events: list[dict] = []
    possession_candidates: list[dict] = []
    visual_facts: list[dict] = []

    previous_owner = None
    previous_ball_ts = None

    for i, cur in enumerate(detections):
        ts = cur["timestamp"]
        players = [d for d in cur["detections"] if d["class_id"] == 1]
        ball = cur["ball"]
        owner = cur["nearest_player"]["id"] if cur["nearest_player"] else None

        if cur["players"] > 0:
            visual_facts.append({
                "timestamp": ts,
                "type": "players_detected",
                "status": "candidate",
                "evidence_level": "video_detection",
                "count": cur["players"],
            })

        if ball and owner is not None:
            p = next((x for x in players if x["id"] == owner), None)
            possession_candidates.append({
                "timestamp": ts,
                "player_track_id": owner,
                "distance_px": cur["nearest_player"]["distance_px"],
                "zone": _zone(p["x"], p["y"], width, height) if p else None,
                "team": p.get("team", "unknown") if p else "unknown",
                "status": "candidate",
                "evidence_level": "ball_player_proximity",
                "reason": "El balón está próximo al jugador más cercano; no implica posesión confirmada ni equipo identificado.",
            })

            if previous_owner is not None and owner != previous_owner:
                events.append(_event(
                    ts, duration, "possession_change_candidate",
                    f"El jugador más cercano al balón cambia del track {previous_owner} al {owner}.",
                    0.35, "ball_player_proximity",
                    {"from_player_track_id": previous_owner, "to_player_track_id": owner}
                ))

            previous_owner = owner
            previous_ball_ts = ts

        if i:
            prev = detections[i - 1]
            dt = max(0.001, ts - prev["timestamp"])
            delta = cur["players"] - prev["players"]

            if delta:
                events.append(_event(
                    ts, duration, "player_count_change",
                    f"El detector pasa de {prev['players']} a {cur['players']} jugadores.",
                    0.35, "video_detection"
                ))

            moving = [d for d in players if d["movement_px"] > float(os.getenv("TACTICAI_MOTION_THRESHOLD_PX", "20"))]
            if moving:
                events.append(_event(
                    ts, duration, "player_movement_candidate",
                    f"{len(moving)} jugadores muestran desplazamiento entre muestras ({dt:.2f}s).",
                    0.25, "video_tracking",
                    {"moving_player_track_ids": [d["id"] for d in moving[:20]]}
                ))

            # Dense local player cluster around the ball: useful as a pressure/contest
            # candidate, but never labelled as a confirmed pressing action.
            if ball:
                nearby = [d for d in players if _distance(ball, d) <= float(os.getenv("TACTICAI_PRESSURE_RADIUS_PX", "260"))]
                if len(nearby) >= 3:
                    events.append(_event(
                        ts, duration, "ball_area_density_candidate",
                        f"Se detectan {len(nearby)} jugadores en un radio cercano al balón.",
                        0.30, "player_ball_geometry",
                        {"nearby_player_count": len(nearby)}
                    ))

    # Build conservative possession/turnover/transition episodes from consecutive
    # ball-player proximity and heuristic team labels. These remain candidates:
    # camera cuts, occlusion and jersey clustering can invalidate them.
    episodes = []
    active = None
    min_distance = float(os.getenv("TACTICAI_POSSESSION_RADIUS_PX", "140"))
    max_gap = float(os.getenv("TACTICAI_POSSESSION_GAP_SECONDS", "1.5"))

    for cur in detections:
        ts = cur["timestamp"]
        nearest = cur["nearest_player"]
        players = [d for d in cur["detections"] if d["class_id"] == 1]
        if nearest:
            p = next((x for x in players if x["id"] == nearest["id"]), None)
            team = p.get("team", "unknown") if p else "unknown"
            distance = float(nearest["distance_px"])
            if distance <= min_distance:
                if active is None:
                    active = {
                        "start": ts, "last": ts, "team": team,
                        "player_track_id": nearest["id"],
                        "samples": 1, "min_distance_px": distance,
                        "start_zone": _zone(p["x"], p["y"], width, height) if p else None,
                    }
                elif ts - active["last"] <= max_gap and (
                    active["player_track_id"] == nearest["id"] or active["team"] == team
                ):
                    active["last"] = ts
                    active["samples"] += 1
                    active["min_distance_px"] = min(active["min_distance_px"], distance)
                else:
                    episodes.append(active)
                    active = {
                        "start": ts, "last": ts, "team": team,
                        "player_track_id": nearest["id"], "samples": 1,
                        "min_distance_px": distance,
                        "start_zone": _zone(p["x"], p["y"], width, height) if p else None,
                    }
            elif active is not None and ts - active["last"] > max_gap:
                episodes.append(active); active = None
        elif active is not None and ts - active["last"] > max_gap:
            episodes.append(active); active = None

    if active is not None:
        episodes.append(active)

    possession_episodes = []
    for ep in episodes:
        if ep["samples"] >= 2 and ep["last"] - ep["start"] >= 0.5:
            possession_episodes.append({
                "start": round(ep["start"], 3),
                "end": round(ep["last"], 3),
                "duration": round(ep["last"] - ep["start"], 3),
                "team": ep["team"],
                "player_track_id": ep["player_track_id"],
                "zone": ep["start_zone"],
                "min_distance_px": round(ep["min_distance_px"], 1),
                "confidence": round(min(0.75, 0.35 + ep["samples"] * 0.04), 2),
                "status": "candidate",
                "evidence_level": "temporal_ball_player_proximity",
                "reason": "Proximidad balón-jugador mantenida durante varias muestras; no constituye posesión confirmada.",
            })

    turnover_candidates = []
    for a, b in zip(possession_episodes, possession_episodes[1:]):
        gap = b["start"] - a["end"]
        if 0 <= gap <= 3.0 and a["team"] != "unknown" and b["team"] != "unknown" and a["team"] != b["team"]:
            turnover_candidates.append({
                "timestamp": round(b["start"], 3),
                "start": round(max(0, a["end"] - 1.0), 3),
                "end": round(min(duration, b["start"] + 3.0), 3),
                "type": "turnover_candidate",
                "status": "candidate",
                "confidence": round(min(a["confidence"], b["confidence"]) * 0.85, 2),
                "evidence_level": "team_change_plus_ball_proximity",
                "from_team": a["team"], "to_team": b["team"],
                "from_player_track_id": a["player_track_id"],
                "to_player_track_id": b["player_track_id"],
                "from_zone": a["zone"], "to_zone": b["zone"],
                "reason": "Se encadenan dos episodios de proximidad al balón de equipos distintos. Requiere validación temporal para confirmar una pérdida/recuperación real.",
            })

    # Add a small, conservative post-turnover snapshot. This is spatial evidence,
    # not a classification of the tactical action.
    for ev in turnover_candidates:
        t0=float(ev["timestamp"])
        own=[]; opps=[]; ball_samples=[]
        for cur in detections:
            ts=float(cur["timestamp"])
            if t0 <= ts <= t0+3.0 and cur.get("ball"):
                players=[d for d in cur["detections"] if d["class_id"]==1]
                b=cur["ball"]
                own_d=[_distance(b,p) for p in players if p.get("team")==ev.get("from_team")]
                opp_d=[_distance(b,p) for p in players if p.get("team")==ev.get("to_team")]
                own.extend(own_d); opps.extend(opp_d); ball_samples.append(cur)
        ev["post_turnover_geometry"]={
            "window_seconds":3.0,
            "samples":len(ball_samples),
            "nearest_losing_team_px":round(min(own),1) if own else None,
            "nearest_recovering_team_px":round(min(opps),1) if opps else None,
            "losing_team_near_ball_180px":round(sum(d<=180 for d in own)/len(own),2) if own else None,
            "recovering_team_near_ball_180px":round(sum(d<=180 for d in opps)/len(opps),2) if opps else None,
            "evidence_level":"post_turnover_spatial_geometry",
        }

    transition_candidates = []
    for ev in turnover_candidates:
        transition_candidates.append({
            **ev,
            "type": "defensive_transition_candidate",
            "reason": "Cambio de equipo asociado al balón; ventana posterior preparada para analizar la transición defensiva.",
        })

    events.extend(turnover_candidates)
    events.extend(transition_candidates)

    events.sort(key=lambda x: x["timestamp"])
    events = events[:1000]
    possession_candidates = possession_candidates[:1000]
    visual_facts = visual_facts[:1000]

    clips = []
    for i, e in enumerate(events[:150]):
        clip_id=f"clip-{i+1}"
        e["clip_id"]=clip_id
        clips.append({
            "id": clip_id,
            "title": e["type"],
            "type": e["type"],
            "timestamp": e["timestamp"],
            "start": e["start"],
            "end": e["end"],
            "status": e["status"],
            "confidence": e.get("confidence", 0),
            "reason": e["reason"],
            "source_video": True,
            "playable": False,
            "note": "Referencia temporal; el render físico del clip se genera en la siguiente capa.",
        })

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
            "unique_track_ids": len({t["id"] for t in tracks_flat if t["class"] == "player"}),
        },
        "capability": {
            "video_decoded": True,
            "football_detector_loaded": detector.available,
            "tracking": True,
            "team_assignment": True,
            "football_events_confirmed": False,
            "notes": "La asignación de equipos es una heurística visual basada en color de camiseta y debe validarse; posesiones, pérdidas y acciones tácticas siguen siendo candidatas hasta incorporar validación temporal e identidad persistente.",
        },
        "events": events,
        "clips": clips,
        "evidence_ledger": clips,
        "sequence": {
            "windows": detections[:1000],
            "tracks": tracks_flat[:10000],
            "possession_candidates": possession_candidates,
            "possession_episodes": possession_episodes,
            "turnover_candidates": turnover_candidates,
            "transition_candidates": transition_candidates,
            "visual_facts": visual_facts,
        },
    }


def build_sequence_features(samples):
    return {"frames": len(samples), "tracks": [], "windows": samples}


def event_candidates(sequence):
    return sequence.get("events", []) if isinstance(sequence, dict) else []


def derive_football_sequences(samples, tracks):
    return {"windows": samples, "tracks": tracks, "sequence_count": 0}


def build_football_event_summary(events):
    candidate_counts: dict[str, int] = {}
    for e in events or []:
        typ = str(e.get("type", "unknown"))
        candidate_counts[typ] = candidate_counts.get(typ, 0) + 1
    return {"candidate_counts": candidate_counts, "confirmed_counts": {}}
