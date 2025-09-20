import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging

from .groups import get_all_phrases
from .preprocess import preprocess_query, preprocess_phrases


class PhraseMatcher:
    """
    Clase para encontrar frases similares usando embeddings y similitud coseno.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_path: str = "data/embeddings.npz"):
        """
        Inicializa el matcher con el modelo de embeddings especificado.

        Args:
            model_name: Nombre del modelo de Sentence-Transformers
            cache_path: Ruta para cachear los embeddings
        """
        self.model_name = model_name
        self.cache_path = cache_path
        self.model = None
        self.grupos_embeddings = {}
        self.grupos_frases = {}
        self.grupos_centroids = {}
        self.logger = logging.getLogger(__name__)

    def _load_model(self):
        """Carga el modelo de embeddings si no está cargado."""
        if self.model is None:
            self.logger.info(f"Cargando modelo: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

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
                self.logger.info("Cargando embeddings desde cache")
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
        self.logger.info("Computando embeddings para todas las frases")
        self._load_model()

        grupos = get_all_phrases()
        embeddings_dict = {}

        for grupo, frases in grupos.items():
            frases_procesadas = preprocess_phrases(frases)
            embeddings = self.model.encode(frases_procesadas, convert_to_numpy=True)
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
            self.logger.info(f"Embeddings guardados en cache: {cache_file}")
        except Exception as e:
            self.logger.error(f"Error al guardar cache: {e}")

    def _compute_centroids(self):
        """Computa los centroides para cada grupo."""
        self.grupos_centroids = {}
        for grupo, embeddings in self.grupos_embeddings.items():
            centroid = np.mean(embeddings, axis=0)
            self.grupos_centroids[grupo] = centroid

    def initialize(self):
        """Inicializa el matcher cargando embeddings y computando centroides."""
        self.logger.info("Inicializando PhraseMatcher")

        # Cargar frases
        self.grupos_frases = get_all_phrases()

        # Cargar o computar embeddings
        self.grupos_embeddings = self._load_or_compute_embeddings()

        # Computar centroides
        self._compute_centroids()

        self.logger.info("PhraseMatcher inicializado correctamente")

    def find_best_group(self, query: str) -> str:
        """
        Encuentra el grupo más similar usando centroides.

        Args:
            query: Consulta de entrada

        Returns:
            Nombre del grupo más similar
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

        # Obtener embedding del query
        query_embedding = self.model.encode([query_processed], convert_to_numpy=True)[0]

        # Calcular similitud con cada centroide
        best_group = None
        best_similarity = -1.0

        for grupo, centroid in self.grupos_centroids.items():
            similarity = cosine_similarity([query_embedding], [centroid])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_group = grupo

        return best_group

    def find_most_similar_phrase(self, query: str, group: Optional[str] = None) -> Tuple[str, str, float]:
        """
        Encuentra la frase más similar dentro de un grupo específico o en todos los grupos.

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
        query_embedding = self.model.encode([query_processed], convert_to_numpy=True)[0]

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

    def search_similar_phrase(self, query: str) -> Dict:
        """
        Busca la frase más similar siguiendo la estrategia de centroides.

        Args:
            query: Consulta de entrada

        Returns:
            Diccionario con resultado de la búsqueda
        """
        # Primero encontrar el mejor grupo usando centroides
        best_group = self.find_best_group(query)

        # Luego encontrar la mejor frase dentro de ese grupo
        grupo, frase, similarity = self.find_most_similar_phrase(query, best_group)

        return {
            "query": query,
            "grupo": grupo,
            "frase_similar": frase,
            "similitud": round(similarity, 4)
        }