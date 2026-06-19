"""
Servicio para actualizar datos de entrenamiento y reentrenar el modelo.

Cuando se actualiza el estado de una cita (a Asistida o No asistió),
este servicio:
1. Añade un nuevo registro al Excel de entrenamiento
2. Incrementa los contadores de asistencias/no-asistencias
3. Ejecuta el reentrenamiento del modelo
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import json
import logging
import shutil
from datetime import datetime
from sqlalchemy.orm import Session
from ..db import Appointment

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "database_non-shows.xlsx"
PROCESSED_DATA_PATH = REPO_ROOT / "data" / "processed" / "df_limpio.csv"
MODEL_SCRIPT = REPO_ROOT / "backend" / "models" / "train_pipeline.py"


def get_latest_patient_record(patient_id: str) -> dict | None:
    """
    Obtiene el registro más reciente de un paciente desde el Excel de entrenamiento.
    Si no lo encuentra, intenta buscar en df_limpio.csv.
    
    Args:
        patient_id: ID del paciente
        
    Returns:
        Dict con los datos del paciente o None si no existe
    """
    try:
        # 1. Buscar en Excel de entrenamiento
        if RAW_DATA_PATH.exists():
            df = pd.read_excel(RAW_DATA_PATH)
            
            # Buscar registros del paciente (con manejo flexible del nombre de columna)
            if 'patient_id' in df.columns:
                patient_records = df[df['patient_id'].astype(str) == str(patient_id)]
            else:
                # Fallback: buscar en la primera columna si no existe 'patient_id'
                logger.warning(f"Columna 'patient_id' no encontrada en Excel")
                patient_records = df.iloc[0:0]  # DataFrame vacío
            
            if not patient_records.empty:
                # Retornar el último registro
                latest = patient_records.iloc[-1].to_dict()
                logger.info(f"Encontrado registro previo para patient_id={patient_id}")
                return latest
        
        # 2. Si no hay en Excel, buscar en df_limpio.csv
        if PROCESSED_DATA_PATH.exists():
            df_clean = pd.read_csv(PROCESSED_DATA_PATH)
            
            if 'patient_id' in df_clean.columns:
                patient_data = df_clean[df_clean['patient_id'].astype(str) == str(patient_id)]
                
                if not patient_data.empty:
                    # Retornar datos del paciente desde df_limpio
                    record = patient_data.iloc[-1].to_dict()
                    logger.info(f"Encontrado dato de paciente en df_limpio para patient_id={patient_id}")
                    return record
        
        logger.info(f"No se encontraron registros previos para patient_id: {patient_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error al obtener registro del paciente: {e}")
        return None


def append_training_record(
    patient_id: str,
    appointment: Appointment,
    attended: bool,
) -> bool:
    """
    Añade un nuevo registro al Excel de entrenamiento basado en una cita actualizada.
    
    IMPORTANTE: Solo cambia los contadores. TODO lo demás permanece idéntico.
    
    Args:
        patient_id: ID del paciente
        appointment: Objeto de cita con los datos actuales
        attended: True si la cita fue "Asistida" (0), False si fue "No asistió" (1)
        
    Returns:
        True si se guardó exitosamente, False en caso contrario
    """
    import time
    
    max_retries = 3
    retry_delay = 1  # segundo
    
    try:
        # Cargar Excel actual o crear uno nuevo
        if RAW_DATA_PATH.exists():
            df = pd.read_excel(RAW_DATA_PATH)
        else:
            logger.warning(f"Excel no existe en {RAW_DATA_PATH}, creando nuevo...")
            df = pd.DataFrame()
        
        # Obtener último registro del paciente
        latest_record = get_latest_patient_record(patient_id)
        
        if not latest_record:
            # Si NO hay registro previo, no podemos crear uno sin datos clínicos
            logger.warning(
                f"No hay datos previos para patient_id={patient_id}. "
                f"Se requiere al menos un registro previo o datos en df_limpio.csv"
            )
            return False
        
        # PASO CRÍTICO: Copiar exactamente el registro anterior (incluyendo NaN)
        new_record = latest_record.copy()
        
        # SOLO actualizar estos campos específicos:
        # 1. Contadores de asistencia
        if attended:
            new_record['Appointment Type'] = 0  # Asistida
            # Incrementar contador de asistencias
            prev_attendance = new_record.get('Number of Previous Attendance', 0)
            try:
                prev_attendance = int(prev_attendance)
            except (ValueError, TypeError):
                prev_attendance = 0
            new_record['Number of Previous Attendance'] = prev_attendance + 1
            
            logger.debug(
                f"Registro de asistencia: patient_id={patient_id}, "
                f"Previous Attendance: {prev_attendance} -> {prev_attendance + 1}"
            )
        else:
            new_record['Appointment Type'] = 1  # No-show
            # Incrementar contador de no-asistencias
            prev_non_attendance = new_record.get('Number of Previous Non-Attendance', 0)
            try:
                prev_non_attendance = int(prev_non_attendance)
            except (ValueError, TypeError):
                prev_non_attendance = 0
            new_record['Number of Previous Non-Attendance'] = prev_non_attendance + 1
            
            logger.debug(
                f"Registro de no-asistencia: patient_id={patient_id}, "
                f"Previous Non-Attendance: {prev_non_attendance} -> {prev_non_attendance + 1}"
            )
        
        # 2. Datos de la cita actual
        new_record['Hour'] = appointment.hour
        new_record['Day'] = appointment.day
        new_record['Month'] = appointment.month
        new_record['medic_id'] = str(appointment.medic_id)
        new_record['patient_id'] = str(patient_id)
        
        # TODO LO DEMÁS permanece igual (Age, Sex, Insurance Type, Diseases, etc.)
        
        # Convertir a DataFrame y concatenar
        new_row = pd.DataFrame([new_record])
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Intentar guardar con reintentos y manejo de permisos
        for attempt in range(max_retries):
            try:
                # Usar archivo temporal para evitar corrupción
                temp_path = RAW_DATA_PATH.with_stem(RAW_DATA_PATH.stem + "_temp")
                
                # Guardar primero en temporal
                with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Sheet1')
                
                # Copiar temporal a destino (evita bloqueos parciales)
                shutil.move(str(temp_path), str(RAW_DATA_PATH))
                
                logger.info(
                    f"✓ Registro guardado exitosamente: patient_id={patient_id}, "
                    f"medic_id={appointment.medic_id}, attended={attended}, "
                    f"total_records={len(df)}"
                )
                return True
                
            except PermissionError as pe:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Intento {attempt + 1}/{max_retries} falló por permisos. "
                        f"Reintentando en {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Backoff exponencial
                else:
                    logger.error(f"Falló después de {max_retries} intentos: {pe}")
                    raise
        
        return False
        
    except Exception as e:
        logger.error(f"Error al guardar registro de entrenamiento: {e}")
        return False


def retrain_model() -> bool:
    """
    Ejecuta el pipeline de reentrenamiento del modelo.
    
    Corre el script train_pipeline.py que:
    1. Carga el Excel actualizado
    2. Limpia y normaliza los datos
    3. Entrena el modelo
    4. Guarda el modelo y las métricas
    
    Returns:
        True si el reentrenamiento fue exitoso, False en caso contrario
    """
    try:
        logger.info("Iniciando reentrenamiento del modelo...")
        
        # Ejecutar el script de entrenamiento
        result = subprocess.run(
            [sys.executable, str(MODEL_SCRIPT)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutos máximo
        )
        
        if result.returncode == 0:
            logger.info("Reentrenamiento completado exitosamente")
            logger.info(f"Output: {result.stdout}")
            return True
        else:
            logger.error(f"Reentrenamiento falló con código {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Reentrenamiento excedió el tiempo máximo (5 minutos)")
        return False
    except Exception as e:
        logger.error(f"Error al ejecutar reentrenamiento: {e}")
        return False


def handle_appointment_status_change(
    appointment: Appointment,
    new_status: int,
) -> dict:
    """
    Maneja el cambio de estado de una cita y dispara el proceso de
    actualización de datos de entrenamiento y reentrenamiento.
    
    Args:
        appointment: Objeto de cita
        new_status: Nuevo estado (0 = Asistida, 1 = No asistió, 2 = Pendiente)
        
    Returns:
        Dict con resultado de la operación
    """
    result = {
        "success": False,
        "message": "",
        "record_saved": False,
        "model_retrained": False,
    }
    
    try:
        # Solo procesar si es Asistida (0) o No asistió (1)
        if new_status not in (0, 1):
            result["message"] = f"Estado {new_status} no requiere actualización de entrenamiento"
            result["success"] = True
            return result
        
        attended = (new_status == 0)
        
        # Paso 1: Guardar nuevo registro en Excel
        record_saved = append_training_record(
            patient_id=appointment.patient_id,
            appointment=appointment,
            attended=attended,
        )
        result["record_saved"] = record_saved
        
        if not record_saved:
            result["message"] = "Error al guardar el registro de entrenamiento"
            return result
        
        # Paso 2: Reentrenar el modelo
        model_retrained = retrain_model()
        result["model_retrained"] = model_retrained
        
        if model_retrained:
            result["success"] = True
            result["message"] = (
                f"Cita actualizada: paciente {appointment.patient_id}, "
                f"estado {'Asistida' if attended else 'No asistió'}. "
                f"Registro guardado y modelo reentrenado."
            )
        else:
            result["success"] = True  # Registro se guardó aunque el reentrenamiento falló
            result["message"] = (
                f"Cita actualizada y registro guardado, pero el reentrenamiento falló. "
                f"El modelo anterior seguirá en uso."
            )
        
        logger.info(result["message"])
        return result
        
    except Exception as e:
        result["message"] = f"Error inesperado: {str(e)}"
        logger.error(result["message"])
        return result
