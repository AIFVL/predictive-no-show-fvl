"""
Servicio de scheduling para reentrenamiento automático del modelo.

Ejecuta el train_pipeline.py periódicamente con datos actualizados del Excel.
"""

import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.executors.pool import ThreadPoolExecutor
except ModuleNotFoundError as exc:
    BackgroundScheduler = None
    IntervalTrigger = None
    ThreadPoolExecutor = None
    APSCHEDULER_IMPORT_ERROR = exc
else:
    APSCHEDULER_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_SCRIPT = REPO_ROOT / "backend" / "models" / "train_pipeline.py"


class ModelRetrainingScheduler:
    """Gestor del scheduling de reentrenamiento del modelo."""
    
    def __init__(self):
        self.scheduler = None
        self.is_running = False
        self.last_training_time = None
        self.last_training_status = "No iniciado"
        self.training_interval_minutes = 5  # Por defecto: cada hora
    
    def initialize(self, interval_seconds: int = 300):
        """Inicializa el scheduler con ejecutores configurados.
        
        Args:
            interval_seconds: Intervalo en SEGUNDOS entre reentrenamientos (default 3600 = 1 hora)
        """
        try:
            if APSCHEDULER_IMPORT_ERROR is not None:
                logger.warning(
                    "Scheduler deshabilitado: falta instalar APScheduler (%s)",
                    APSCHEDULER_IMPORT_ERROR,
                )
                self.last_training_status = (
                    "Scheduler deshabilitado: instala APScheduler para reentrenamiento automatico"
                )
                return False

            self.training_interval_minutes = interval_seconds / 60
            
            # Configurar scheduler con ejecutor de threads
            executors = {
                'default': ThreadPoolExecutor(max_workers=2)
            }
            
            job_defaults = {
                'coalesce': True,
                'max_instances': 1
            }
            
            self.scheduler = BackgroundScheduler(
                executors=executors,
                job_defaults=job_defaults,
                timezone='UTC'
            )
            
            # Agregar job de reentrenamiento (usando SEGUNDOS)
            job = self.scheduler.add_job(
                func=self._retrain_job,
                trigger=IntervalTrigger(seconds=interval_seconds),
                id="model_retraining_job",
                name="Automatic Model Retraining",
                replace_existing=True,
            )
            
            logger.info(
                f"✓ Scheduler inicializado: reentrenamiento cada {interval_seconds} segundos"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error al inicializar scheduler: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def start(self) -> bool:
        """Inicia el scheduler."""
        try:
            if self.scheduler and not self.scheduler.running:
                self.scheduler.start()
                self.is_running = True
                logger.info("✓ Scheduler de reentrenamiento INICIADO")
                logger.info(f"Jobs en scheduler: {len(self.scheduler.get_jobs())}")
                for job in self.scheduler.get_jobs():
                    logger.info(f"  - Job: {job.id}, Próxima ejecución: {job.next_run_time}")
                return True
            elif self.scheduler and self.scheduler.running:
                logger.info("ℹScheduler ya está corriendo")
                return True
            else:
                logger.error("Scheduler no inicializado")
                return False
                
        except Exception as e:
            logger.error(f"Error al iniciar scheduler: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def stop(self) -> bool:
        """
        Detiene el scheduler de reentrenamiento.
        
        Returns:
            True si se detuvo correctamente, False en caso contrario
        """
        try:
            if self.scheduler and self.scheduler.running:
                self.scheduler.shutdown(wait=False)
                self.is_running = False
                logger.info("✓ Scheduler de reentrenamiento DETENIDO")
                return True
            else:
                logger.warning("Scheduler no está corriendo")
                return False
                
        except Exception as e:
            logger.error(f"Error al detener scheduler: {e}")
            return False
    
    def update_interval(self, minutes: int) -> bool:
        """Actualiza el intervalo de reentrenamiento."""
        try:
            if minutes < 1:
                logger.error("El intervalo debe ser >= 1 minuto")
                return False
            
            if self.scheduler:
                self.scheduler.remove_job("model_retraining_job")
                
                job = self.scheduler.add_job(
                    func=self._retrain_job,
                    trigger=IntervalTrigger(minutes=minutes),
                    id="model_retraining_job",
                    name="Automatic Model Retraining",
                    replace_existing=True,
                )
                
                self.training_interval_minutes = minutes
                logger.info(f"✓ Intervalo actualizado a {minutes} minutos")
                if self.scheduler.running and job.next_run_time:
                    logger.info(f"Próxima ejecución: {job.next_run_time}")
                return True
            else:
                logger.error("Scheduler no inicializado")
                return False
                
        except Exception as e:
            logger.error(f"Error al actualizar intervalo: {e}")
            return False
    
    def _retrain_job(self):
        """Job de reentrenamiento que se ejecuta periódicamente."""
        try:
            msg = "=" * 70
            logger.info(msg)
            logger.info("INICIANDO REENTRENAMIENTO AUTOMÁTICO DEL MODELO")
            logger.info(f"   Timestamp: {datetime.now().isoformat()}")
            logger.info(msg)
            
            # Imprimir a stdout para debugging en Uvicorn
            print(msg, flush=True)
            print("INICIANDO REENTRENAMIENTO AUTOMÁTICO", flush=True)
            print(f"   Timestamp: {datetime.now().isoformat()}", flush=True)
            print(msg, flush=True)
            
            self.last_training_time = datetime.now()
            
            # Ejecutar script de entrenamiento
            try:
                result = subprocess.run(
                    [sys.executable, str(MODEL_SCRIPT)],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=900,  # 15 minutos max
                )
                
                # Imprimir output del script
                if result.stdout:
                    print(f"[TRAIN OUTPUT]\n{result.stdout}", flush=True)
                if result.stderr:
                    print(f"[TRAIN ERROR]\n{result.stderr}", flush=True)
                
                if result.returncode == 0:
                    self.last_training_status = "✓ Exitoso"
                    msg = f"✓ REENTRENAMIENTO EXITOSO (proceso completado en {(datetime.now() - self.last_training_time).total_seconds():.1f}s)"
                    print(msg, flush=True)
                    logger.info(msg)
                    logger.info("=" * 70)
                    return True
                else:
                    self.last_training_status = f"✗ Error (código {result.returncode})"
                    msg = f"✗ REENTRENAMIENTO FALLÓ (código {result.returncode})"
                    print(msg, flush=True)
                    logger.error(msg)
                    logger.info("=" * 70)
                    return False
                    
            except subprocess.TimeoutExpired:
                self.last_training_status = "✗ Timeout (>15min)"
                msg = "✗ SCRIPT EXCEDIÓ 15 MINUTOS"
                print(msg, flush=True)
                logger.error(msg)
                logger.info("=" * 70)
                return False
                
        except Exception as e:
            self.last_training_status = f"✗ Error: {str(e)}"
            logger.error(f"✗ ERROR: {e}")
            print(f"✗ ERROR EN JOB: {e}", flush=True)
            import traceback
            logger.error(traceback.format_exc())
            logger.info("=" * 70)
            return False
    

    
    def get_status(self) -> dict:
        """Obtiene el estado actual del scheduler."""
        next_run = None
        if self.scheduler and self.scheduler.running:
            job = self.scheduler.get_job("model_retraining_job")
            if job:
                next_run = job.next_run_time.isoformat() if job.next_run_time else None
        
        return {
            "available": APSCHEDULER_IMPORT_ERROR is None,
            "is_running": self.is_running,
            "scheduler_running": self.scheduler.running if self.scheduler else False,
            "interval_minutes": self.training_interval_minutes,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "last_training_status": self.last_training_status,
            "next_job_time": next_run,
        }
    
    def manual_retrain(self) -> bool:
        """
        Ejecuta manualmente el reentrenamiento fuera del schedule.
        
        Returns:
            True si se ejecutó correctamente, False en caso contrario
        """
        return self._retrain_job()


# Instancia global del scheduler
model_scheduler = ModelRetrainingScheduler()
