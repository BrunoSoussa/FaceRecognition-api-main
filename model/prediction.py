import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
from numpy import asarray, expand_dims
from model.architecture import InceptionResNetV1
import chromadb
from chromadb.utils import embedding_functions

class FaceRecognition:
    def __init__(self, model_path: str, collection_name: str, chroma_path: str = "chroma_data"):
        self._model = InceptionResNetV1(weights_path=model_path)
        self._chroma_client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._initialize_collection(collection_name)

    def _initialize_collection(self, collection_name: str):
        try:
            return self._chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except chromadb.db.base.UniqueConstraintError:
            return self._chroma_client.get_collection(name=collection_name)

    @staticmethod
    def _load_image(filename: str) -> np.ndarray:
        image = Image.open(filename)
        image = image.convert("RGB")
        return asarray(image)

    @staticmethod
    def _extract_face(image: np.ndarray, required_size: Tuple[int, int] = (160, 160)) -> Optional[np.ndarray]:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        face_image = Image.fromarray(face)
        face_image = face_image.resize(required_size)
        return asarray(face_image)

    def _get_embedding(self, face_pixels: np.ndarray) -> np.ndarray:
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        yhat = self._model.predict(samples)
        return yhat[0]

    def _process_image(self, filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        image = self._load_image(filepath)
        face = self._extract_face(image)
        if face is None:
            print("Nenhum rosto detectado.")
            return None, None
        embedding = self._get_embedding(face)
        return face, embedding

    def add_to_db(self, image_path: str, name: str) -> None:
        face, embedding = self._process_image(image_path)
        if face is not None and embedding is not None:
            self._collection.add(
                embeddings=[embedding.tolist()],
                ids=[name],
                metadatas=[{"name": name}]
            )
            print(f"Adicionado ao banco de dados: {name}")
        else:
            print("Falha ao processar a imagem para adicionar ao banco de dados.")

    def find_in_db(self, image_path: str, threshold: float = 0.5) -> Optional[str]:
        face, embedding = self._process_image(image_path)
        if face is None or embedding is None:
            print("Falha ao processar a imagem para comparação.")
            return None

        results = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=1
        )

        if results["distances"] and results["distances"][0][0] <= threshold:
            best_match = results["metadatas"][0][0]["name"]
            print(f"Melhor correspondência: {best_match} com similaridade {1 - results['distances'][0][0]}")
            return best_match
        else:
            print("Nenhuma correspondência encontrada.")
            return None
