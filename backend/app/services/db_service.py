from typing import List

from sqlalchemy.orm import Session

from ..db import Appointment


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


def list_appointments(db: Session) -> List[Appointment]:
    return db.query(Appointment).order_by(Appointment.created_at.desc()).all()


def list_appointments_by_medic(db: Session, medic_id: str) -> List[Appointment]:
    return db.query(Appointment).filter(Appointment.medic_id == medic_id).order_by(Appointment.created_at.desc()).all()

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