from datetime import date
from typing import List, Optional

from sqlalchemy.orm import Session

from ..db import Appointment
from ..utils.appointment_dates import filter_by_forward_window, filter_by_operational_window


def create_appointment(db: Session, medic_id: str, patient_id: str, hour: int, day: int, month: int, appointment_type: int = 2) -> Appointment:
    appt = Appointment(
        medic_id=str(medic_id),
        patient_id=str(patient_id),
        hour=int(hour),
        day=int(day),
        month=int(month),
        appointment_type=int(appointment_type),
    )
    db.add(appt)
    db.commit()
    db.refresh(appt)
    return appt


def list_appointments(
    db: Session,
    days: Optional[int] = None,
    *,
    include_past: bool = True,
    reference: date | None = None,
) -> List[Appointment]:
    appts = db.query(Appointment).order_by(Appointment.created_at.desc()).all()
    if include_past:
        return filter_by_operational_window(appts, days, reference=reference)
    return filter_by_forward_window(appts, days, reference=reference)


def list_appointments_by_medic(
    db: Session,
    medic_id: str,
    days: Optional[int] = None,
    *,
    include_past: bool = True,
    reference: date | None = None,
) -> List[Appointment]:
    appts = (
        db.query(Appointment)
        .filter(Appointment.medic_id == medic_id)
        .order_by(Appointment.created_at.desc())
        .all()
    )
    if include_past:
        return filter_by_operational_window(appts, days, reference=reference)
    return filter_by_forward_window(appts, days, reference=reference)

def get_appointment_info(db: Session, appointment_id: int) -> Appointment:
    return db.query(Appointment).filter(Appointment.id == appointment_id).first()

def update_appointment(db: Session, appointment_id: int, medic_id: str, patient_id: str, hour: int, day: int, month: int, appointment_type: int) -> Appointment:
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if appt:
        appt.medic_id = str(medic_id)
        appt.patient_id = str(patient_id)
        appt.hour = int(hour)
        appt.day = int(day)
        appt.month = int(month)
        appt.appointment_type = int(appointment_type)
        db.commit()
        db.refresh(appt)
    return appt

def update_appointment_type(db: Session, appointment_id: int, appointment_type: int) -> Appointment:
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if appt:
        appt.appointment_type = int(appointment_type)
        db.commit()
        db.refresh(appt)
    return appt

def delete_appointment(db: Session, appointment_id: int) -> None:
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if appt:
        db.delete(appt)
        db.commit()
    return
