"""
Rutas para controlar el scheduler de reentrenamiento automático.
"""

from fastapi import APIRouter, HTTPException
from ..services.scheduler_service import model_scheduler

scheduler_routes = APIRouter(prefix="/scheduler", tags=["scheduler"])


@scheduler_routes.post("/start")
def start_scheduler():
    """Inicia el scheduler de reentrenamiento automático."""
    if model_scheduler.start():
        return {
            "message": "Scheduler iniciado correctamente",
            "status": model_scheduler.get_status()
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Error al iniciar el scheduler"
        )


@scheduler_routes.post("/stop")
def stop_scheduler():
    """Detiene el scheduler de reentrenamiento automático."""
    if model_scheduler.stop():
        return {
            "message": "Scheduler detenido correctamente",
            "status": model_scheduler.get_status()
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Error al detener el scheduler"
        )


@scheduler_routes.post("/manual-retrain")
def manual_retrain():
    """Ejecuta manualmente el reentrenamiento del modelo."""
    if model_scheduler.manual_retrain():
        return {
            "message": "Reentrenamiento ejecutado correctamente",
            "status": model_scheduler.get_status()
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Error durante el reentrenamiento manual"
        )


@scheduler_routes.put("/interval")
def update_interval(minutes: int):
    """
    Actualiza el intervalo de reentrenamiento automático.
    
    Args:
        minutes: Nuevo intervalo en minutos (mínimo 1)
    """
    if minutes < 1:
        raise HTTPException(
            status_code=400,
            detail="El intervalo debe ser al menos 1 minuto"
        )
    
    if model_scheduler.update_interval(minutes):
        return {
            "message": f"Intervalo actualizado a {minutes} minutos",
            "status": model_scheduler.get_status()
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Error al actualizar el intervalo"
        )


@scheduler_routes.get("/status")
def get_scheduler_status():
    """Obtiene el estado actual del scheduler."""
    return model_scheduler.get_status()


@scheduler_routes.post("/reload-model")
def reload_model_endpoint():
    """Endpoint manual para recargar el modelo desde disco (debug/testing).
    
    Se usa después de un reentrenamiento para que el API use el nuevo modelo
    sin necesidad de reiniciar el servidor.
    """
    try:
        from ..services.prediction import reload_model
        reload_info = reload_model()
        return {
            "status": "success",
            "message": "Model reloaded from disk",
            "reload_info": reload_info,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )
