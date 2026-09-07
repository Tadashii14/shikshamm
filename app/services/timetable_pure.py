"""Pure-Python Pomodoro timetable generator — no pandas/plotly, so it runs on Vercel's slim runtime.

Mirrors the output shape of app/services/timetable_repository.generate_timetable_from_repository..
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

DIFFICULTY_WEIGHTS = {"easy": 1, "medium": 2, "hard": 3}

DAY_START = 8.0      # 8:00 AM
DAY_END = 22.0        # 10:00 PM
BLOCK_HOURS = 1.0       # 1 hour study block
BREAK_HOURS = 0.25    # 15 minute break


def _format_time(hour_float: float) -> str:
    h = int(hour_float) % 24
    m = int(round((hour_float - h) * 60)) % 60
    return f"{h:02d}:{m:02d}"


def _parse_busy_hours(busy_hours: Optional[Dict[str, List]]) -> Dict[str, List[List[float]]]:
    """busy_hours: {day: [\"09:00-10:30\", ...]} -> {day: [[start, end] floats]}"""
    parsed: Dict[str, List[List[float]]] = {day: [] for day in DAYS}
    if not busy_hours:
        return parsed
    for day, ranges in busy_hours.items():
        if day not in parsed or not ranges:
            continue
        for r in ranges:
            if not r:
                continue
            m = re.match(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$", str(r).strip())
            if not m:
                continue
            s_start = int(m.group(1)) + int(m.group(2)) / 60.0
            s_end = int(m.group(3)) + int(m.group(4)) / 60.0
            if s_end <= s_start:
                continue
            parsed[day].append([s_start, s_end])
    return parsed


def _available_slots(busy: List[List[float]], max_blocks: int) -> List[float]:
    """Return the study-start times for the available (non-conflicting) blocks of the day."""
    slots = []
    cursor = DAY_START
    while cursor + BLOCK_HOURS <= DAY_END and len(slots) < max_blocks:
        block_end = cursor + BLOCK_HOURS
        overlap = any(not (cursor >= e or block_end <= s) for s, e in busy)
        if not overlap:
            slots.append(cursor)
        cursor += BLOCK_HOURS + BREAK_HOURS
    return slots


def _flatten_weighted_subjects(subjects: List[str], weights: List[int]) -> List[str]:
    """Create an interleaved pool: each subject appears `weight` times, spread across the day."""
    entries = sorted(range(len(subjects)), key=lambda i: (-weights[i], i))  # heaviest first
    max_w = max(weights) if weights else 1
    pool: List[str] = []
    for round_i in range(max_w):
        for i in entries:
            if weights[i] > round_i:
                pool.append(subjects[i])
    return pool


def generate_timetable_pure(
    subjects: List[str],
    difficulty_levels: Optional[List[str]] = None,
    study_hours_per_day: int = 4,
    busy_hours: Optional[Dict[str, List]] = None,
    max_blocks_per_day: Optional[int] = None,
) -> Dict:
    """Generate a Pomodoro study timetable using pure Python (stdlib only)."""

    subjects = [s.strip() for s in subjects if s and s.strip()] or ["General Study"]
    difficulty_levels = difficulty_levels or []
    weights = [
        DIFFICULTY_WEIGHTS.get(str(d).strip().lower(), 2)
        for d in difficulty_levels
    ]
    while len(weights) < len(subjects):
        weights.append(2)
    weights = weights[:len(subjects)]
# Total capacity
    def_cap = max(1, math.floor(study_hours_per_day / (BLOCK_HOURS + BREAK_HOURS)))
    if max_blocks_per_day is None or int(max_blocks_per_day) < 1:
        cap_per_day = def_cap
    else:
        cap_per_day = int(max_blocks_per_day)
    busy = _parse_busy_hours(busy_hours)
    slots_per_day = {day: _available_slots(busy[day], cap_per_day) for day in DAYS}
    total_capacity = (sum(len(slots) for slots in slots_per_day.values()))

    if total_capacity <= 0:
        # Degenerate case: everything busy — just use the default window anyway
        busy = {day: [] for day in DAYS}
        slots_per_day = {day: _available_slots(busy[day], cap_per_day) for day in DAYS}
        total_capacity = (sum(len(slots) for slots in slots_per_day.values()))
    if total_capacity <= 0:
        slots_per_day = {day: [DAY_START] for day in DAYS}
        total_capacity = 7

    # Weighted round-robin subject order
    pool = _flatten_weighted_subjects(subjects, weights)
    if not pool:
        pool = subjects

    timetable: Dict[str, List[Dict]] = {day: [] for day in DAYS}
    cursor = 0
    for day in DAYS:
        for start_h in slots_per_day[day]:
            subject = pool[cursor % len(pool)]
            cursor +=  1
            diff = weights[subjects.index(subject)]
            timetable[day].append({
                "subject": subject,
                "difficulty": ("hard" if diff >= 3 else "easy" if diff <= 1 else "medium"),
                "study_start": start_h,
                "study_end": start_h + BLOCK_HOURS,
                "break_end": start_h + BLOCK_HOURS + BREAK_HOURS,
            })

    # Format the flat schedule (same as repository)
    formatted_timetable = []
    for day, blocks in timetable.items():
        for block in blocks:
            formatted_timetable.append({
                "day": day,
                "subject": block["subject"],
                "difficulty": block["difficulty"],
                "start_time": _format_time(block["study_start"]),
                "end_time": _format_time(block["study_end"]),
                "break_end_time": _format_time(block["break_end"]),
            })

    # subject allocation counts
    subject_allocation = {}
    for block in formatted_timetable:
        subject_allocation[block["subject"]] = subject_allocation.get(block["subject"], 0) + 1

    # Daily summary (same shape as repository)
    daily_summary = []
    for day in DAYS:
        day_blocks = timetable[day]
        if day_blocks:
            study_subjects = [b["subject"] for b in day_blocks]
            start_time = min(b["study_start"] for b in day_blocks)
            end_time = max(b["break_end"] for b in day_blocks)
            daily_summary.append({
                "Day": day,
                "Study Hours": len(day_blocks),
                "Subjects": ", ".join(study_subjects),
                "Start Time": _format_time(start_time),
                "End Time": _format_time(end_time),
            })
        else:
            daily_summary.append({
                "Day": day,
                "Study Hours": 0,
                "Subjects":"Rest Day",
                "Start Time": "-",
                "End Time": "-",
            })

    total_blocks = sum(len(blocks) for blocks in timetable.values())
    total_study_hours = total_blocks * BLOCK_HOURS
    total_time_with_breaks = total_blocks * (BLOCK_HOURS + BREAK_HOURS)
    daily_average = round(total_time_with_breaks / 7, 2)

    return {
        "success": True,
        "schedule": formatted_timetable,
        "timetable": formatted_timetable,
        "subject_allocation": subject_allocation,
        "daily_summary": daily_summary,
        "metrics": {
            "total_blocks": total_blocks,
            "total_study_hours": total_study_hours,
            "total_time_with_breaks": total_time_with_breaks,
            "daily_average_hours_with_breaks": daily_average,
            "daily_average": daily_average,
        },
    }