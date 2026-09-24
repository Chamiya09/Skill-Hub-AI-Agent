"""
scheduler.py
────────────────────────────────────────────────────────────────────────────
Deterministic constraint-satisfaction engine for multi-track interview slot
generation with automatic forward-search extension.
"""

from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Tuple
import logging

from interview_scheduler_agent.schemas import (
    ProposedSlot,
    UnscheduledCandidate,
    ScheduleSummary,
    ScheduleProposalResponse,
)

logger = logging.getLogger(__name__)

TRACK_ROOM_NAMES = [
    "Room A (Panel 1)",
    "Room B (Panel 2)",
    "Room C (Panel 3)",
    "Room D (Executive Panel)",
    "Room E (Technical Lab)",
    "Room F (Virtual Room)",
]


def time_to_minutes(time_str: str) -> int:
    """Parses 'HH:mm' string to total minutes since midnight."""
    clean = time_str.strip()
    if " " in clean:
        clean = clean.split(" ")[0]
    parts = clean.split(":")
    hours = int(parts[0])
    minutes = int(parts[1]) if len(parts) > 1 else 0
    return hours * 60 + minutes


def minutes_to_time(minutes: int) -> str:
    """Converts total minutes since midnight to 24-hour 'HH:mm' string."""
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def is_overlapping(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Returns True if two time intervals overlap."""
    return max(start1, start2) < min(end1, end2)


def get_track_name(track_number: int) -> str:
    """Provides professional conference room / panel label for each track."""
    idx = (track_number - 1) % len(TRACK_ROOM_NAMES)
    return f"Track {track_number}: {TRACK_ROOM_NAMES[idx]}"


def generate_available_slots_for_date(
    target_date: date,
    parallel_tracks: int,
    duration_min: int,
    buffer_min: int,
    work_start_min: int,
    work_end_min: int,
    blocked_slots_for_date: List[Dict[str, Any]],
    is_extended: bool = False
) -> List[Dict[str, Any]]:
    """
    Generates non-overlapping slots across parallel tracks for a single day,
    filtering out any clashes with existing events or holidays.
    """
    # Exclude weekends (Saturday = 5, Sunday = 6)
    if target_date.weekday() >= 5:
        return []

    # Check for full-day holiday or full-day block
    for b in blocked_slots_for_date:
        b_start = time_to_minutes(b.get("startTime", "00:00"))
        b_end = time_to_minutes(b.get("endTime", "23:59"))
        if b_start <= work_start_min and b_end >= work_end_min:
            # Entire working day blocked
            return []

    date_str = target_date.strftime("%Y-%m-%d")
    available_slots: List[Dict[str, Any]] = []

    # Parse blocked intervals for the day
    blocked_intervals: List[Tuple[int, int]] = []
    for b in blocked_slots_for_date:
        b_start = time_to_minutes(b.get("startTime", "00:00"))
        b_end = time_to_minutes(b.get("endTime", "23:59"))
        blocked_intervals.append((b_start, b_end))

    # Generate slots across each track
    curr_min = work_start_min
    while curr_min + duration_min <= work_end_min:
        slot_start = curr_min
        slot_end = curr_min + duration_min

        # Check collision against existing company events
        clashing_blocks = [
            (bs, be) for bs, be in blocked_intervals
            if is_overlapping(slot_start, slot_end, bs, be)
        ]

        if clashing_blocks:
            max_clash_end = max(be for bs, be in clashing_blocks)
            curr_min = max(curr_min + 1, max_clash_end)
            continue

        # Create a slot for each parallel track
        for track_num in range(1, parallel_tracks + 1):
            available_slots.append({
                "date": date_str,
                "start_min": slot_start,
                "end_min": slot_end,
                "start_time": minutes_to_time(slot_start),
                "end_time": minutes_to_time(slot_end),
                "track_number": track_num,
                "track_name": get_track_name(track_num),
                "is_extended": is_extended,
            })

        curr_min = slot_end + buffer_min

    # Sort chronologically by start time then track number
    available_slots.sort(key=lambda s: (s["start_min"], s["track_number"]))
    return available_slots


def compute_interview_schedule(
    job_vacancy_id: str,
    job_title: str,
    candidates: List[Dict[str, Any]],
    blocked_slots: List[Dict[str, Any]],
    start_date_str: str,
    end_date_str: str,
    duration_min: int = 30,
    parallel_tracks: int = 2,
    work_start_str: str = "09:00",
    work_end_str: str = "17:00",
    buffer_min: int = 10,
    max_forward_search_days: int = 14
) -> ScheduleProposalResponse:
    """
    Core scheduling engine:
    1. Generates clash-free slots within the initial date window.
    2. Assigns shortlisted candidates in order.
    3. If overflow occurs, automatically continues forward-search up to max_forward_search_days.
    4. Reports any unassigned candidates who exceeded the search limit.
    """
    start_d = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_d = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    if end_d < start_d:
        end_d = start_d

    work_start_min = time_to_minutes(work_start_str or "09:00")
    work_end_min = time_to_minutes(work_end_str or "17:00")
    if work_end_min <= work_start_min:
        work_end_min = work_start_min + (8 * 60)

    # Group blocked slots by date "YYYY-MM-DD"
    blocked_by_date: Dict[str, List[Dict[str, Any]]] = {}
    for b in blocked_slots:
        d_str = b.get("date", "")
        if d_str:
            blocked_by_date.setdefault(d_str, []).append(b)

    proposed_slots: List[ProposedSlot] = []
    unscheduled_candidates: List[UnscheduledCandidate] = []
    remaining_candidates = list(candidates)

    # Track distinct dates utilized
    dates_scheduled: set = set()
    forward_days_extended = 0

    # Phase 1: Schedule within original date range
    curr_d = start_d
    while curr_d <= end_d and remaining_candidates:
        d_str = curr_d.strftime("%Y-%m-%d")
        day_blocked = blocked_by_date.get(d_str, [])
        slots = generate_available_slots_for_date(
            target_date=curr_d,
            parallel_tracks=parallel_tracks,
            duration_min=duration_min,
            buffer_min=buffer_min,
            work_start_min=work_start_min,
            work_end_min=work_end_min,
            blocked_slots_for_date=day_blocked,
            is_extended=False
        )

        for slot in slots:
            if not remaining_candidates:
                break
            cand = remaining_candidates.pop(0)
            slot_id = f"slot_{d_str}_{slot['track_number']}_{slot['start_time'].replace(':', '')}"
            proposed_slots.append(
                ProposedSlot(
                    slot_id=slot_id,
                    candidate_id=str(cand.get("candidateId", cand.get("candidate_id", ""))),
                    candidate_name=cand.get("fullName", cand.get("full_name", "Candidate")),
                    candidate_email=cand.get("email", ""),
                    date=slot["date"],
                    start_time=slot["start_time"],
                    end_time=slot["end_time"],
                    track_number=slot["track_number"],
                    track_name=slot["track_name"],
                    is_extended_search=False
                )
            )
            dates_scheduled.add(slot["date"])

        curr_d += timedelta(days=1)

    # Phase 2: Forward Search Extension if overflow candidates remain
    search_limit_d = end_d + timedelta(days=max_forward_search_days)
    curr_forward_d = end_d + timedelta(days=1)

    if remaining_candidates:
        logger.info(
            "[Scheduler] Overflow: %d candidates need forward-search scheduling past %s (Limit: %s)",
            len(remaining_candidates),
            end_date_str,
            search_limit_d.strftime("%Y-%m-%d")
        )

    while curr_forward_d <= search_limit_d and remaining_candidates:
        d_str = curr_forward_d.strftime("%Y-%m-%d")
        day_blocked = blocked_by_date.get(d_str, [])
        slots = generate_available_slots_for_date(
            target_date=curr_forward_d,
            parallel_tracks=parallel_tracks,
            duration_min=duration_min,
            buffer_min=buffer_min,
            work_start_min=work_start_min,
            work_end_min=work_end_min,
            blocked_slots_for_date=day_blocked,
            is_extended=True
        )

        if slots:
            forward_days_extended = (curr_forward_d - end_d).days

        for slot in slots:
            if not remaining_candidates:
                break
            cand = remaining_candidates.pop(0)
            slot_id = f"slot_{d_str}_{slot['track_number']}_{slot['start_time'].replace(':', '')}"
            proposed_slots.append(
                ProposedSlot(
                    slot_id=slot_id,
                    candidate_id=str(cand.get("candidateId", cand.get("candidate_id", ""))),
                    candidate_name=cand.get("fullName", cand.get("full_name", "Candidate")),
                    candidate_email=cand.get("email", ""),
                    date=slot["date"],
                    start_time=slot["start_time"],
                    end_time=slot["end_time"],
                    track_number=slot["track_number"],
                    track_name=slot["track_name"],
                    is_extended_search=True
                )
            )
            dates_scheduled.add(slot["date"])

        curr_forward_d += timedelta(days=1)

    # Phase 3: Unscheduled Candidates Handling
    for cand in remaining_candidates:
        unscheduled_candidates.append(
            UnscheduledCandidate(
                candidate_id=str(cand.get("candidateId", cand.get("candidate_id", ""))),
                candidate_name=cand.get("fullName", cand.get("full_name", "Candidate")),
                candidate_email=cand.get("email", ""),
                reason=(
                    f"Could not fit within initial range ({start_date_str} to {end_date_str}) "
                    f"nor within the +{max_forward_search_days} day forward search limit. "
                    "All available working hour slots were exhausted."
                )
            )
        )

    # Summary preparation
    effective_start = start_date_str
    sorted_dates = sorted(list(dates_scheduled))
    effective_end = sorted_dates[-1] if sorted_dates else end_date_str

    assumptions = [
        f"Working hours configured as {work_start_str} to {work_end_str} with {buffer_min}-minute transition buffer.",
        "Weekends (Saturday and Sunday) are strictly excluded from interview scheduling.",
        f"Utilized {parallel_tracks} concurrent parallel track(s) per time slot.",
        f"Interview duration set to {duration_min} minutes per candidate."
    ]
    if forward_days_extended > 0:
        assumptions.append(
            f"Forward search automatically extended the schedule window by {forward_days_extended} day(s) "
            f"to accommodate all candidates without schedule collisions."
        )

    ai_notes = [
        f"Generated {len(proposed_slots)} clash-free interview sessions across {parallel_tracks} track(s).",
        "Checked against existing company calendar events and national holidays to prevent double-booking.",
    ]
    if unscheduled_candidates:
        ai_notes.append(
            f"ALERT: {len(unscheduled_candidates)} candidate(s) could not be scheduled within the search limit. "
            "Please add more parallel tracks or expand the interview date window."
        )

    summary = ScheduleSummary(
        total_candidates=len(candidates),
        scheduled_count=len(proposed_slots),
        unscheduled_count=len(unscheduled_candidates),
        original_date_range=f"{start_date_str} to {end_date_str}",
        effective_date_range=f"{effective_start} to {effective_end}",
        forward_days_extended=forward_days_extended,
        tracks_utilized=parallel_tracks,
        assumptions_made=assumptions,
        ai_validation_notes=ai_notes
    )

    return ScheduleProposalResponse(
        job_vacancy_id=job_vacancy_id,
        job_title=job_title,
        proposed_slots=proposed_slots,
        unscheduled_candidates=unscheduled_candidates,
        summary=summary,
        is_draft=True
    )
