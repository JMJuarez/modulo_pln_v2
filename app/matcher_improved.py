import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging

from .groups import get_all_phrases
from .preprocess import preprocess_query, preprocess_phrases


class ImprovedPhraseMatcher:
    """
    Versión mejorada del matcher con:
    1. Modelo optimizado para español
    2. Re-ranking en dos fases
    3. Expansión de sinónimos
    4. Ajuste de threshold por grupo
    """

    # Modelos disponibles ordenados por calidad para español
    MODELS = {
        "spanish_optimized": "hiiamsid/sentence_similarity_spanish_es",  # Mejor para español
        "multilingual_advanced": "paraphrase-multilingual-mpnet-base-v2",  # Más potente
        "multilingual_balanced": "paraphrase-multilingual-MiniLM-L12-v2",  # Balanceado
        "current": "all-MiniLM-L6-v2"  # Actual
    }

    # Threshold adaptativo por grupo
    GROUP_THRESHOLDS = {
        "A": 0.70,  # Saludos: más flexible
        "B": 0.65,  # Solicitudes: muy flexible
        "C": 0.75   # Problemas: más estricto
    }

    # Sinónimos para expansión de query
    SYNONYMS = {
        "ayuda": ["asistencia", "soporte", "apoyo"],
        "problema": ["error", "fallo", "inconveniente", "issue"],
        "quiero": ["deseo", "necesito", "requiero"],
        "cambiar": ["modificar", "actualizar", "editar"],
        "cancelar": ["eliminar", "borrar", "anular"],
        "hola": ["saludos", "buenos días", "buenas"],
        "gracias": ["agradecimiento", "muchas gracias", "te agradezco"],
    }

    def __init__(
        self,
        model_type: str = "multilingual_balanced",  # Mejor que el actual
        cache_path: str = "data/embeddings_improved.npz",
        use_reranking: bool = True,
        use_synonym_expansion: bool = True
    ):
        """
        Inicializa el matcher mejorado.

        Args:
            model_type: Tipo de modelo a usar (ver MODELS)
            cache_path: Ruta para cachear los embeddings
            use_reranking: Activar re-ranking en dos fases
            use_synonym_expansion: Expandir query con sinónimos
        """
        self.model_name = self.MODELS.get(model_type, self.MODELS["current"])
        self.cache_path = cache_path
        self.use_reranking = use_reranking
        self.use_synonym_expansion = use_synonym_expansion
        self.model = None
        self.grupos_embeddings = {}
        self.grupos_frases = {}
        self.grupos_centroids = {}
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Matcher mejorado usando modelo: {self.model_name}")

    def _load_model(self):
        """Carga el modelo de embeddings si no está cargado."""
        if self.model is None:
            self.logger.info(f"Cargando modelo mejorado: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def _expand_with_synonyms(self, query: str) -> List[str]:
        """
        Expande la query con sinónimos relevantes.

        Args:
            query: Query original

        Returns:
            Lista de queries expandidas
        """
        if not self.use_synonym_expansion:
            return [query]

        queries = [query]
        words = query.lower().split()

        for word in words:
            if word in self.SYNONYMS:
                for synonym in self.SYNONYMS[word]:
                    expanded = query.lower().replace(word, synonym)
                    queries.append(expanded)

        return queries[:5]  # Limitar a 5 variaciones

    def _load_or_compute_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Carga embeddings desde cache o los computa si no existen.

        Returns:
            Diccionario con embeddings por grupo
        """
        cache_file = Path(self.cache_path)

        # Intentar cargar desde cache
        if cache_file.exists():
            try:
                self.logger.info("Cargando embeddings mejorados desde cache")
                data = np.load(cache_file, allow_pickle=True)
                embeddings_dict = {}
                for key in data.files:
                    embeddings_dict[key] = data[key]
                return embeddings_dict
            except Exception as e:
                self.logger.warning(f"Error al cargar cache: {e}. Recomputando embeddings.")

        # Computar embeddings
        return self._compute_embeddings()

    def _compute_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Computa los embeddings para todas las frases.

        Returns:
            Diccionario con embeddings por grupo
        """
        self.logger.info("Computando embeddings mejorados para todas las frases")
        self._load_model()

        grupos = get_all_phrases()
        embeddings_dict = {}

        for grupo, frases in grupos.items():
            frases_procesadas = preprocess_phrases(frases)
            embeddings = self.model.encode(
                frases_procesadas,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalizar para mejor similitud
            )
            embeddings_dict[grupo] = embeddings

        # Guardar en cache
        self._save_embeddings_cache(embeddings_dict)

        return embeddings_dict

    def _save_embeddings_cache(self, embeddings_dict: Dict[str, np.ndarray]):
        """
        Guarda los embeddings en cache.

        Args:
            embeddings_dict: Diccionario con embeddings por grupo
        """
        try:
            cache_file = Path(self.cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, **embeddings_dict)
            self.logger.info(f"Embeddings mejorados guardados en cache: {cache_file}")
        except Exception as e:
            self.logger.error(f"Error al guardar cache: {e}")

    def _compute_centroids(self):
        """Computa los centroides para cada grupo."""
        self.grupos_centroids = {}
        for grupo, embeddings in self.grupos_embeddings.items():
            centroid = np.mean(embeddings, axis=0)
            # Normalizar centroide
            centroid = centroid / np.linalg.norm(centroid)
            self.grupos_centroids[grupo] = centroid

    def initialize(self):
        """Inicializa el matcher cargando embeddings y computando centroides."""
        self.logger.info("Inicializando PhraseMatcher mejorado")

        # Cargar frases
        self.grupos_frases = get_all_phrases()

        # Cargar o computar embeddings
        self.grupos_embeddings = self._load_or_compute_embeddings()

        # Computar centroides
        self._compute_centroids()

        self.logger.info("PhraseMatcher mejorado inicializado correctamente")

    def find_best_groups(self, query: str, top_k: int = 2) -> List[Tuple[str, float]]:
        """
        Encuentra los top-k grupos más similares usando centroides.

        Args:
            query: Consulta de entrada
            top_k: Número de grupos a retornar

        Returns:
            Lista de tuplas (grupo, similitud)
        """
        if not self.grupos_centroids:
            raise ValueError("Matcher no inicializado. Llama a initialize() primero.")

        self._load_model()

        # Obtener todas las frases para corrección
        all_phrases = []
        for frases in self.grupos_frases.values():
            all_phrases.extend(frases)

        # Preprocesar query
        query_processed = preprocess_query(query, all_phrases)

        # Expandir con sinónimos
        queries = self._expand_with_synonyms(query_processed)

        # Obtener embeddings de todas las variaciones
        query_embeddings = self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Promediar embeddings de variaciones
        query_embedding = np.mean(query_embeddings, axis=0)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Calcular similitud con cada centroide
        group_scores = []
        for grupo, centroid in self.grupos_centroids.items():
            similarity = cosine_similarity([query_embedding], [centroid])[0][0]
            group_scores.append((grupo, similarity))

        # Ordenar por similitud descendente
        group_scores.sort(key=lambda x: x[1], reverse=True)

        return group_scores[:top_k]

    def find_most_similar_phrase_reranked(self, query: str) -> Tuple[str, str, float]:
        """
        Encuentra la frase más similar usando re-ranking en dos fases.

        Args:
            query: Consulta de entrada

        Returns:
            Tupla con (grupo, frase_más_similar, score_similitud)
        """
        # Fase 1: Encontrar top-2 grupos candidatos
        top_groups = self.find_best_groups(query, top_k=2)

        self._load_model()

        # Obtener todas las frases para corrección
        all_phrases = []
        for frases in self.grupos_frases.values():
            all_phrases.extend(frases)

        # Preprocesar query
        query_processed = preprocess_query(query, all_phrases)

        # Obtener embedding del query
        query_embedding = self.model.encode(
            [query_processed],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        best_group = None
        best_phrase = None
        best_similarity = -1.0

        # Fase 2: Búsqueda fina en grupos candidatos
        for grupo, group_score in top_groups:
            embeddings = self.grupos_embeddings[grupo]
            frases = self.grupos_frases[grupo]

            # Calcular similitud con todas las frases del grupo
            similarities = cosine_similarity([query_embedding], embeddings)[0]

            # Encontrar la mejor similitud en este grupo
            max_idx = np.argmax(similarities)
            max_similarity = similarities[max_idx]

            # Aplicar threshold adaptativo
            threshold = self.GROUP_THRESHOLDS.get(grupo, 0.70)

            # Bonus por ser el grupo más probable
            if grupo == top_groups[0][0]:
                max_similarity += 0.05  # Boost al grupo más probable

            if max_similarity > best_similarity and max_similarity >= threshold:
                best_similarity = max_similarity
                best_group = grupo
                best_phrase = frases[max_idx]

        # Si no se encontró nada por threshold, retornar el mejor absoluto
        if best_group is None:
            grupo = top_groups[0][0]
            embeddings = self.grupos_embeddings[grupo]
            frases = self.grupos_frases[grupo]
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            max_idx = np.argmax(similarities)
            best_similarity = similarities[max_idx]
            best_group = grupo
            best_phrase = frases[max_idx]

        return best_group, best_phrase, best_similarity

    def search_similar_phrase(self, query: str) -> Dict:
        """
        Busca la frase más similar usando estrategia mejorada.

        Args:
            query: Consulta de entrada

        Returns:
            Diccionario con resultado de la búsqueda
        """
        if self.use_reranking:
            grupo, frase, similarity = self.find_most_similar_phrase_reranked(query)
        else:
            # Fallback a método básico
            best_group = self.find_best_groups(query, top_k=1)[0][0]
            grupo, frase, similarity = self.find_most_similar_phrase(query, best_group)

        return {
            "query": query,
            "grupo": grupo,
            "frase_similar": frase,
            "similitud": round(similarity, 4)
        }

    def find_most_similar_phrase(self, query: str, group: Optional[str] = None) -> Tuple[str, str, float]:
        """
        Encuentra la frase más similar (método básico para compatibilidad).

        Args:
            query: Consulta de entrada
            group: Grupo específico donde buscar (opcional)

        Returns:
            Tupla con (grupo, frase_más_similar, score_similitud)
        """
        if not self.grupos_embeddings:
            raise ValueError("Matcher no inicializado. Llama a initialize() primero.")

        self._load_model()

        # Obtener todas las frases para corrección
        all_phrases = []
        for frases in self.grupos_frases.values():
            all_phrases.extend(frases)

        # Preprocesar query
        query_processed = preprocess_query(query, all_phrases)

        # Obtener embedding del query
        query_embedding = self.model.encode(
            [query_processed],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        best_group = None
        best_phrase = None
        best_similarity = -1.0

        # Determinar grupos donde buscar
        groups_to_search = [group] if group else list(self.grupos_frases.keys())

        for grupo in groups_to_search:
            if grupo not in self.grupos_embeddings:
                continue

            embeddings = self.grupos_embeddings[grupo]
            frases = self.grupos_frases[grupo]

            # Calcular similitud con todas las frases del grupo
            similarities = cosine_similarity([query_embedding], embeddings)[0]

            # Encontrar la mejor similitud en este grupo
            max_idx = np.argmax(similarities)
            max_similarity = similarities[max_idx]

            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_group = grupo
                best_phrase = frases[max_idx]

        return best_group, best_phrase, best_similarity
