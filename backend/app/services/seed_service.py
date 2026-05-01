from __future__ import annotations

import random
from calendar import monthrange
from datetime import datetime
from typing import Any, Dict, Iterable

from sqlalchemy.orm import Session

from ..db import Appointment


def seed_appointments_2026(
    db: Session,
    *,
    min_per_month: int = 20,
    months: Iterable[int] = range(1, 13),
    medic_min: int = 1,
    medic_max: int = 10,
    patient_max: int = 18000,
    hour_min: int = 7,
    hour_max: int = 18,
    appointment_type: int = 2,
    random_seed: int = 2026,
) -> Dict[str, Any]:
    """Ensure there are at least `min_per_month` appointments for each month of 2026.

    This function is idempotent: it only creates missing appointments per month.
    """

    if min_per_month <= 0:
        return {"created": 0, "by_month": {}}

    rng = random.Random(random_seed)

    created_total = 0
    created_by_month: Dict[int, int] = {}

    for month in months:
        month = int(month)
        if month < 1 or month > 12:
            continue

        existing = db.query(Appointment).filter(Appointment.month == month).count()
        missing = max(0, int(min_per_month) - int(existing))
        if missing == 0:
            created_by_month[month] = 0
            continue

        days_in_month = monthrange(2026, month)[1]
        hours = list(range(int(hour_min), int(hour_max) + 1))

        batch = []
        used = set()
        attempts = 0
        max_attempts = missing * 30

        while len(batch) < missing and attempts < max_attempts:
            attempts += 1
            day = rng.randint(1, days_in_month)
            hour = rng.choice(hours)
            medic_id = str(rng.randint(int(medic_min), int(medic_max)))
            patient_id = str(rng.randint(1, int(patient_max)))

            key = (month, day, hour, medic_id, patient_id)
            if key in used:
                continue
            used.add(key)

            batch.append(
                Appointment(
                    medic_id=medic_id,
                    patient_id=patient_id,
                    hour=int(hour),
                    day=int(day),
                    month=int(month),
                    appointment_type=int(appointment_type),
                    created_at=datetime.utcnow(),
                )
            )

        if batch:
            db.add_all(batch)
            db.commit()
            created_total += len(batch)
            created_by_month[month] = len(batch)
        else:
            created_by_month[month] = 0

    return {"created": created_total, "by_month": created_by_month}
