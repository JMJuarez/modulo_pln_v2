import re
import unicodedata
from typing import List
from rapidfuzz import fuzz


def normalize_text(text: str) -> str:
    """
    Normaliza el texto removiendo acentos, convirtiendo a minúsculas,
    y limpiando caracteres especiales.

    Args:
        text: Texto a normalizar

    Returns:
        Texto normalizado
    """
    # Convertir a minúsculas
    text = text.lower()

    # Remover acentos
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')

    # Limpiar caracteres especiales, mantener solo letras, números y espacios
    text = re.sub(r'[^\w\s]', ' ', text)

    # Normalizar espacios múltiples a uno solo
    text = re.sub(r'\s+', ' ', text)

    # Remover espacios al inicio y final
    text = text.strip()

    return text


def light_spelling_correction(query: str, reference_phrases: List[str], threshold: float = 80.0) -> str:
    """
    Aplica corrección ligera de ortografía usando similitud difusa.
    Si encuentra una frase con alta similitud, sugiere una corrección.

    Args:
        query: Texto de consulta
        reference_phrases: Lista de frases de referencia
        threshold: Umbral de similitud para sugerir corrección

    Returns:
        Texto corregido o el original si no se encuentra corrección
    """
    query_normalized = normalize_text(query)
    best_match = ""
    best_score = 0.0

    for phrase in reference_phrases:
        phrase_normalized = normalize_text(phrase)
        score = fuzz.ratio(query_normalized, phrase_normalized)

        if score > best_score and score >= threshold:
            best_score = score
            best_match = phrase

    # Si encontramos una buena coincidencia y es suficientemente diferente,
    # sugerimos la corrección
    if best_match and best_score >= threshold:
        return best_match

    return query


def preprocess_query(query: str, reference_phrases: List[str] = None) -> str:
    """
    Preprocesa la consulta aplicando normalización y corrección opcional.

    Args:
        query: Texto de consulta
        reference_phrases: Lista opcional de frases de referencia para corrección

    Returns:
        Consulta preprocesada
    """
    # Aplicar corrección ligera si se proporcionan frases de referencia
    if reference_phrases:
        query = light_spelling_correction(query, reference_phrases)

    # Normalizar texto
    query = normalize_text(query)

    return query


def preprocess_phrases(phrases: List[str]) -> List[str]:
    """
    Preprocesa una lista de frases aplicando normalización.

    Args:
        phrases: Lista de frases a preprocesar

    Returns:
        Lista de frases preprocesadas
    """
    return [normalize_text(phrase) for phrase in phrases]