from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import logging
import uvicorn

from .matcher_improved import ImprovedPhraseMatcher as PhraseMatcher
from .groups import get_all_phrases
from .preprocess import spell_out_text

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Buscador de Frase Similar",
    description="API para encontrar frases similares usando embeddings y similitud coseno",
    version="1.0.0"
)

# Instancia global del matcher
matcher = None


class QueryRequest(BaseModel):
    """Modelo para la solicitud de búsqueda."""
    texto: str


class QueryResponse(BaseModel):
    """Modelo para la respuesta de búsqueda."""
    query: str
    grupo: str
    frase_similar: str
    similitud: float


class StatusResponse(BaseModel):
    """Modelo para la respuesta de estado."""
    status: str
    grupos_disponibles: List[str]
    total_frases: int


class SpellOutRequest(BaseModel):
    """Modelo para la solicitud de deletreo."""
    texto: str
    incluir_espacios: bool = True


class SpellOutResponse(BaseModel):
    """Modelo para la respuesta de deletreo."""
    texto_original: str
    deletreo: List[str]
    total_caracteres: int


@app.on_event("startup")
async def startup_event():
    """Inicializa el matcher mejorado al arrancar la aplicación."""
    global matcher
    try:
        logger.info("Inicializando la aplicación con matcher mejorado...")
        # Usar modelo balanceado optimizado para español con todas las mejoras
        matcher = PhraseMatcher(
            model_type="multilingual_balanced",  # Mejor modelo para español
            use_reranking=True,  # Re-ranking en dos fases
            use_synonym_expansion=True  # Expansión de sinónimos
        )
        matcher.initialize()
        logger.info("Aplicación inicializada correctamente con matcher mejorado")
    except Exception as e:
        logger.error(f"Error al inicializar la aplicación: {e}")
        raise


@app.get("/", response_model=StatusResponse)
async def root():
    """
    Endpoint raíz que muestra el estado del sistema.
    """
    try:
        grupos = get_all_phrases()
        total_frases = sum(len(frases) for frases in grupos.values())

        return StatusResponse(
            status="OK",
            grupos_disponibles=list(grupos.keys()),
            total_frases=total_frases
        )
    except Exception as e:
        logger.error(f"Error en endpoint root: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.post("/buscar", response_model=QueryResponse)
async def buscar_frase_similar(request: QueryRequest):
    """
    Busca la frase más similar al texto proporcionado.

    Args:
        request: Solicitud con el texto a buscar

    Returns:
        Respuesta con la frase más similar encontrada
    """
    if matcher is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible: matcher no inicializado")

    if not request.texto or not request.texto.strip():
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

    try:
        logger.info(f"Buscando similitud para: {request.texto}")
        resultado = matcher.search_similar_phrase(request.texto)

        response = QueryResponse(
            query=resultado["query"],
            grupo=resultado["grupo"],
            frase_similar=resultado["frase_similar"],
            similitud=resultado["similitud"]
        )

        logger.info(f"Resultado: {response.grupo} - {response.similitud}")
        return response

    except Exception as e:
        logger.error(f"Error al buscar frase similar: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.get("/grupos")
async def obtener_grupos():
    """
    Obtiene todos los grupos y sus frases.

    Returns:
        Diccionario con todos los grupos y frases
    """
    try:
        grupos = get_all_phrases()
        return {"grupos": grupos}
    except Exception as e:
        logger.error(f"Error al obtener grupos: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.get("/grupos/{grupo}")
async def obtener_frases_grupo(grupo: str):
    """
    Obtiene las frases de un grupo específico.

    Args:
        grupo: Nombre del grupo (A, B, C)

    Returns:
        Lista de frases del grupo
    """
    try:
        grupos = get_all_phrases()
        if grupo not in grupos:
            raise HTTPException(status_code=404, detail=f"Grupo '{grupo}' no encontrado")

        return {
            "grupo": grupo,
            "frases": grupos[grupo]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener frases del grupo {grupo}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.post("/deletreo", response_model=SpellOutResponse)
async def deletrear_texto(request: SpellOutRequest):
    """
    Deletrea el texto proporcionado carácter por carácter.

    Args:
        request: Solicitud con el texto a deletrear

    Returns:
        Respuesta con el deletreo del texto
    """
    if not request.texto:
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

    try:
        logger.info(f"Deletreando texto: {request.texto}")
        deletreo = spell_out_text(request.texto, request.incluir_espacios)

        response = SpellOutResponse(
            texto_original=request.texto,
            deletreo=deletreo,
            total_caracteres=len(deletreo)
        )

        logger.info(f"Deletreo completado: {len(deletreo)} caracteres")
        return response

    except Exception as e:
        logger.error(f"Error al deletrear texto: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.get("/health")
async def health_check():
    """
    Endpoint de verificación de salud del servicio.

    Returns:
        Estado del servicio
    """
    try:
        # Verificar que el matcher esté inicializado
        if matcher is None:
            return {"status": "unhealthy", "reason": "Matcher no inicializado"}

        # Verificar que los grupos se puedan cargar
        grupos = get_all_phrases()
        if not grupos:
            return {"status": "unhealthy", "reason": "No se pudieron cargar los grupos"}

        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return {"status": "unhealthy", "reason": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)