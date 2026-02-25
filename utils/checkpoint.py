import json
import os
from datetime import datetime
from typing import Optional

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

# Nombre del archivo de metadata que acompaña a cada .h5
_METADATA_SUFFIX = "_metadata.json"


class CheckpointManager:
    """
    Gestiona el guardado y carga de pesos de modelos Keras/TensorFlow.

    Cada checkpoint consiste en dos archivos guardados juntos:
        - {name}.h5           → pesos del modelo (formato HDF5)
        - {name}_metadata.json → información adicional: timestamp, métricas,
                                  nombre del modelo, número de episodio.

    La metadata permite auditar el historial de entrenamiento y saber
    exactamente cuándo y en qué estado se guardó cada checkpoint.

    Uso típico para las 9 configuraciones de Main-DQN:
        >>> for alpha, omega in hp.get_all_alpha_omega_combinations():
        ...     name = f"main_dqn_alpha{int(alpha*100)}_omega{omega}"
        ...     ckpt.save(model=main_agent.model.get_main_network(), name=name)
    """

    def __init__(self, checkpoint_dir: str = "utils/checkpoints"):

        self._checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: "tf.keras.Model",
        name: str,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Guarda los pesos de un modelo Keras en formato .h5.

        Solo guarda los PESOS (weights), no la arquitectura completa.
        Esto requiere que el modelo ya esté construido (compilado o con
        al menos un forward pass) antes de llamar a save.

        Guarda también un archivo JSON de metadata junto al .h5 con:
            - timestamp: fecha y hora del guardado
            - model_name: nombre pasado como argumento
            - metrics: diccionario opcional con ROI, SR, episodio, etc.

        Returns:
            Ruta completa al archivo .h5 guardado.

        Raises:
            RuntimeError: Si TensorFlow no está disponible.
            ValueError: Si el modelo no tiene pesos definidos (no construido).

        Ejemplo:
            >>> ckpt.save(
            ...     model=trade_agent.model.get_main_network(),
            ...     name="trade_dqn",
            ...     metadata={"episode": 50, "roi": 9.2, "sr": 2.737}
            ... )
            'utils/checkpoints/trade_dqn.h5'
        """
        self._require_tensorflow()

        weights_path = self._get_weights_path(name)
        metadata_path = self._get_metadata_path(name)

        # Guardar pesos en formato HDF5.
        # save_weights solo guarda los tensores de pesos, no la arquitectura.
        # Para cargar correctamente, el modelo receptor debe tener exactamente
        # la misma arquitectura (mismo número de capas y neuronas).
        model.save_weights(weights_path)

        # Guardar metadata como JSON legible.
        full_metadata = {
            "model_name": name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "weights_path": weights_path,
            "metrics": metadata or {},
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)

        return weights_path

    def load(
        self,
        model: "tf.keras.Model",
        name: str
    ) -> bool:
        """
        Carga pesos desde un checkpoint .h5 en un modelo Keras existente.

        El modelo receptor DEBE tener exactamente la misma arquitectura
        que el modelo que generó el checkpoint. No se valida esto
        automáticamente — una arquitectura incompatible producirá un error
        de TensorFlow al intentar asignar los pesos.

        Returns:
            True si los pesos se cargaron exitosamente.
            False si el archivo .h5 no existe (checkpoint no encontrado).

        Raises:
            RuntimeError: Si TensorFlow no está disponible.
            Exception: Si los pesos son incompatibles con la arquitectura del modelo.

        Ejemplo:
            >>> loaded = ckpt.load(
            ...     model=main_agent.model.get_main_network(),
            ...     name="main_dqn_alpha55_omega16"
            ... )
            >>> if loaded:
            ...     print("Pesos cargados, continuando entrenamiento")
            ... else:
            ...     print("No hay checkpoint, entrenando desde cero")
        """
        self._require_tensorflow()

        weights_path = self._get_weights_path(name)

        if not os.path.exists(weights_path):
            return False

        model.load_weights(weights_path)
        return True

    def exists(self, name: str) -> bool:
        """
        Verifica si existe un checkpoint con el nombre dado.

        No requiere TensorFlow (solo verifica existencia de archivo).
        Útil para decidir al inicio de training si reanudar o empezar
        desde cero, sin necesidad de instanciar el modelo primero.

        Returns:
            True si el archivo .h5 existe en el directorio de checkpoints.
            False si no existe.

        Ejemplo:
            >>> if ckpt.exists("trade_dqn"):
            ...     print("Checkpoint encontrado, cargando pesos")
            ... else:
            ...     print("Sin checkpoint previo, entrenando desde cero")
        """
        return os.path.exists(self._get_weights_path(name))

    def get_metadata(self, name: str) -> Optional[dict]:
        """
        Lee y retorna el archivo de metadata de un checkpoint.

        Permite consultar cuándo se guardó el checkpoint y con qué métricas,
        sin necesidad de cargar los pesos en un modelo.

        Returns:
            Diccionario con metadata si el archivo JSON existe, None si no.
            Estructura del dict:
                {
                    "model_name": str,
                    "timestamp": str,       # "YYYY-MM-DD HH:MM:SS"
                    "weights_path": str,
                    "metrics": {            # lo que se pasó en save()
                        "episode": int,
                        "roi": float,
                        "sr": float,
                        ...
                    }
                }
        """
        metadata_path = self._get_metadata_path(name)
        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_checkpoints(self) -> list:
        """
        Lista todos los checkpoints disponibles en el directorio.

        Retorna solo los nombres base (sin extensión) de los archivos .h5
        encontrados. Útil para verificar qué modelos están entrenados antes
        de ejecutar evaluación.

        Returns:
            Lista de nombres base de checkpoints disponibles, ordenados
            alfabéticamente.
            Ejemplo: ["main_dqn_alpha30_omega8", "main_dqn_alpha55_omega16",
                      "main_dqn_best", "predictive_dqn", "trade_dqn"]
        """
        if not os.path.exists(self._checkpoint_dir):
            return []

        checkpoints = []
        for fname in os.listdir(self._checkpoint_dir):
            if fname.endswith(".weights.h5"):
                name = fname[:-11]  # Eliminar extensión .weights.h5
                checkpoints.append(name)

        return sorted(checkpoints)

    def delete(self, name: str) -> bool:
        """
        Elimina un checkpoint y su metadata asociada.

        Útil para liberar espacio después de consolidar los mejores pesos
        o al final de una corrida experimental.


        Returns:
            True si al menos el archivo .h5 fue eliminado exitosamente.
            False si el archivo .h5 no existía.

        Nota:
            Intenta eliminar el .h5 y el _metadata.json. Si el .json no
            existe, no lanza error (puede que no se haya creado).
        """
        weights_path = self._get_weights_path(name)
        metadata_path = self._get_metadata_path(name)

        deleted = False

        if os.path.exists(weights_path):
            os.remove(weights_path)
            deleted = True

        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        return deleted

    def save_best(
        self,
        model: "tf.keras.Model",
        current_sr: float,
        name: str = "main_dqn_best",
        metadata: Optional[dict] = None
    ) -> tuple:
        """
        Guarda el modelo solo si su SR supera al mejor SR registrado.

        Diseñado para usarse en el loop de entrenamiento del Main-DQN,
        donde se quiere conservar la configuración con mejor riesgo ajustado
        (el paper reporta SR=2.74 como mejor resultado).

        Returns:
            Tupla (saved: bool, best_sr: float) donde:
                - saved=True si se guardó (SR mejoró)
                - saved=False si no se guardó (SR no mejoró)
                - best_sr es el mejor SR conocido después de la llamada.

        Ejemplo:
            >>> saved, best_sr = ckpt.save_best(
            ...     model=agent.model.get_main_network(),
            ...     current_sr=2.394,
            ...     metadata={"alpha": 0.55, "omega": 16, "roi": 29.93}
            ... )
            >>> if saved:
            ...     logger.info(f"Nuevo mejor SR: {best_sr:.4f}")
        """
        # Leer el mejor SR registrado en metadata, si existe.
        existing_meta = self.get_metadata(name)
        best_sr_so_far = -float("inf")

        if existing_meta is not None:
            best_sr_so_far = existing_meta.get("metrics", {}).get("sr", -float("inf"))

        if current_sr > best_sr_so_far:
            full_metadata = metadata or {}
            full_metadata["sr"] = current_sr
            self.save(model=model, name=name, metadata=full_metadata)
            return True, current_sr

        return False, best_sr_so_far

    # =========================================================================
    # MÉTODOS PRIVADOS
    # =========================================================================

    def _get_weights_path(self, name: str) -> str:
        return os.path.join(self._checkpoint_dir, f"{name}.weights.h5")

    def _get_metadata_path(self, name: str) -> str:
        return os.path.join(self._checkpoint_dir, f"{name}{_METADATA_SUFFIX}")

    def _require_tensorflow(self) -> None:
        if not _TF_AVAILABLE:
            raise RuntimeError(
                "TensorFlow no está disponible. "
                "Instalar con: pip install tensorflow"
            )

    @property
    def checkpoint_dir(self) -> str:
        return self._checkpoint_dir
