import logging
import os
import sys
from datetime import datetime
from typing import Optional


# Formato compartido por todos los loggers del proyecto.
# Ejemplo de línea: [2024-01-15 14:32:05] [TradeDQN] [INFO] Episodio 10 completado
_LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class Logger:
    """
    Logger unificado para el proyecto M-DQN.

    Crea un logger con dos handlers:
        1. StreamHandler: salida a consola (stdout).
        2. FileHandler (opcional): salida a archivo .log.

    Ambos usan el mismo formato y nivel de log.

    Uso básico:
        >>> logger = Logger(name="TradeDQN")
        >>> logger.info("Iniciando entrenamiento")
        [2024-01-15 14:32:05] [TradeDQN] [INFO] Iniciando entrenamiento

    Uso con archivo:
        >>> logger = Logger(name="MainDQN", log_file="logs/training.log")
        >>> logger.error("Fallo al cargar pesos")

    Uso para métricas de entrenamiento:
        >>> logger.log_episode(episode=10, reward=1250.5, epsilon=0.85, roi=5.2)
        [2024-01-15 14:32:05] [TradeDQN] [INFO] [EPISODE 10] reward=1250.50 | epsilon=0.8500 | roi=5.20%

    Uso para métricas de evaluación:
        >>> logger.log_evaluation(alpha=0.55, omega=16, roi=29.93, sr=2.74, n_trades=372)
        [2024-01-15 14:32:05] [TradeDQN] [INFO] [EVAL] alpha=0.55 | omega=16 | ROI=29.93% | SR=2.7400 | trades=372
    """

    def __init__(self, name: str, log_file: Optional[str] = None, level: int = logging.INFO):
        """
        Inicializa el logger con el nombre del módulo que lo instancia.

        Args:
            name: Nombre del logger, idealmente el nombre del módulo o clase
                  que lo usa (ej. "TradeDQN", "DataIntegrator", "MainPipeline").
                  Aparece en cada línea de log entre corchetes.
            log_file: Ruta opcional al archivo .log donde guardar los mensajes.
                      Si es None, solo se loggea a consola.
                      Si el directorio no existe, se crea automáticamente.
            level: Nivel mínimo de log (default: logging.INFO).
                   Opciones: logging.DEBUG, logging.INFO, logging.WARNING,
                   logging.ERROR, logging.CRITICAL.
        """
        self._name = name
        self._log_file = log_file
        self._level = level

        # Crear el logger interno de Python con el nombre recibido.
        # Si ya existe un logger con ese nombre (reutilización), se reusan
        # sus handlers para evitar duplicación de mensajes.
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        # Solo agregar handlers si el logger no tiene ninguno aún.
        # Evita duplicar handlers si se instancia Logger("X") varias veces.
        if not self._logger.handlers:
            self._setup_handlers(log_file, level)

    def _setup_handlers(self, log_file: Optional[str], level: int) -> None:
        """
        Configura los handlers de consola y archivo.

        Args:
            log_file: Ruta al archivo de log (None = solo consola).
            level: Nivel mínimo de log para ambos handlers.
        """
        formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

        # Handler 1: Consola (stdout).
        # Usar stdout en lugar de stderr para que los logs sean capturables
        # por pipes y redirecciones en scripts de entrenamiento.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # Handler 2: Archivo (opcional).
        if log_file is not None:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            # mode="a" = append, no sobreescribe logs anteriores.
            # encoding="utf-8" para compatibilidad multiplataforma.
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    # =========================================================================
    # MÉTODOS DE LOG BÁSICOS
    # Wrappers directos sobre los niveles estándar de Python logging.
    # =========================================================================

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    # =========================================================================
    # MÉTODOS DE LOG ESPECIALIZADOS PARA MÉTRICAS DEL PAPER
    # Estos métodos formatean los valores numéricos con la precisión adecuada
    # para que los logs sean legibles y comparables con las tablas del paper.
    # =========================================================================

    def log_episode(
        self,
        episode: int,
        reward: float,
        epsilon: float,
        roi: float,
        extra: Optional[dict] = None
    ) -> None:
        """
        Loggea métricas de un episodio de entrenamiento completado.

        Formato de salida:
            [EPISODE 10] reward=1250.50 | epsilon=0.8500 | roi=5.20%

        Uso:
            >>> logger.log_episode(episode=50, reward=3400.0, epsilon=0.77, roi=12.5)
            [EPISODE 50] reward=3400.00 | epsilon=0.7700 | roi=12.50%
        """
        msg = (
            f"[EPISODE {episode:>5}] "
            f"reward={reward:>10.2f} | "
            f"epsilon={epsilon:.4f} | "
            f"roi={roi:>7.2f}%"
        )
        if extra:
            extra_str = " | ".join(f"{k}={v}" for k, v in extra.items())
            msg += f" | {extra_str}"
        self._logger.info(msg)

    def log_evaluation(
        self,
        alpha: float,
        omega: int,
        roi: float,
        sr: float,
        n_trades: int,
        method: str = "proposed",
        initial_inv: float = 1_000_000.0,
        final_inv: Optional[float] = None
    ) -> None:
        """
        Loggea resultados de evaluación de una configuración (alpha, omega).
        Reproduce el formato de las Tables 4, 5 y 6 del paper.

        Formato de salida:
            [EVAL] method=proposed | alpha=0.55 | omega=16 | trades=372 |
            initial=$1,000,000 | final=$1,299,381 | ROI=29.93% | SR=2.3942

        Uso:
            >>> logger.log_evaluation(
            ...     alpha=0.55, omega=16, roi=29.93, sr=2.394,
            ...     n_trades=372, method="proposed", final_inv=1_299_381
            ... )
        """
        final_str = f"${final_inv:>12,.0f}" if final_inv is not None else "N/A"
        msg = (
            f"[EVAL] "
            f"method={method:<8} | "
            f"alpha={alpha:.2f} | "
            f"omega={omega:>2} | "
            f"trades={n_trades:>4} | "
            f"initial=${initial_inv:>12,.0f} | "
            f"final={final_str} | "
            f"ROI={roi:>7.2f}% | "
            f"SR={sr:.4f}"
        )
        self._logger.info(msg)

    def log_training_start(self, model_name: str, config: dict) -> None:
        """
        Loggea el inicio de entrenamiento con la configuración completa.
        Útil para reproducibilidad: registra todos los hiperparámetros
        al inicio de cada run de entrenamiento.

        Uso:
            >>> logger.log_training_start("TradeDQN", {
            ...     "lr": 0.001, "gamma": 0.95, "epsilon": 1.0,
            ...     "batch_size": 64, "target_update": 400
            ... })
        """
        separator = "=" * 60
        self._logger.info(separator)
        self._logger.info(f"INICIANDO ENTRENAMIENTO: {model_name}")
        self._logger.info(f"Timestamp: {datetime.now().strftime(_DATE_FORMAT)}")
        for key, value in config.items():
            self._logger.info(f"  {key:<25} = {value}")
        self._logger.info(separator)

    def log_training_end(
        self,
        model_name: str,
        best_roi: float,
        best_sr: float,
        total_episodes: int,
        weights_path: str
    ) -> None:
        """
        Loggea el resumen final al terminar el entrenamiento de un modelo.

        Uso:
            >>> logger.log_training_end(
            ...     "MainDQN", best_roi=29.93, best_sr=2.74,
            ...     total_episodes=500, weights_path="utils/checkpoints/main_dqn_best.h5"
            ... )
        """
        separator = "=" * 60
        self._logger.info(separator)
        self._logger.info(f"ENTRENAMIENTO COMPLETADO: {model_name}")
        self._logger.info(f"  Episodios totales : {total_episodes}")
        self._logger.info(f"  Mejor ROI         : {best_roi:.2f}%")
        self._logger.info(f"  Mejor SR          : {best_sr:.4f}")
        self._logger.info(f"  Pesos guardados   : {weights_path}")
        self._logger.info(separator)

    def log_step(self, step: int, loss: float, reward: float) -> None:
        """
        Loggea métricas de un step individual de entrenamiento.
        Solo usar con level=DEBUG para no saturar los logs en producción.
        """
        self._logger.debug(
            f"[STEP {step:>6}] loss={loss:.6f} | reward={reward:>10.4f}"
        )

    def log_checkpoint_saved(self, path: str) -> None:
        """
        Loggea confirmación de guardado de checkpoint.
        """
        self._logger.info(f"[CHECKPOINT] Pesos guardados en: {path}")

    def log_checkpoint_loaded(self, path: str) -> None:
        """
        Loggea confirmación de carga de checkpoint.
        """
        self._logger.info(f"[CHECKPOINT] Pesos cargados desde: {path}")

    def log_data_stats(self, dataset_name: str, stats: dict) -> None:
        """
        Loggea estadísticas de un dataset procesado.
        Útil en el módulo de preprocessing para confirmar que los datos
        tienen el shape y rango esperados antes de entrenamiento.

        Uso:
            >>> logger.log_data_stats("integrated_dataset", {
            ...     "rows": 36120,
            ...     "null_values": 0,
            ...     "date_range": "2014-10-01 / 2018-11-14",
            ...     "x1_unique": [-1, 0, 1],
            ...     "x2_range": "[-100, 100]"
            ... })
        """
        self._logger.info(f"[DATASET] {dataset_name}:")
        for key, value in stats.items():
            self._logger.info(f"  {key:<25} = {value}")

    @property
    def name(self) -> str:
        """Retorna el nombre del logger."""
        return self._name

    @property
    def log_file(self) -> Optional[str]:
        """Retorna la ruta del archivo de log (None si no hay)."""
        return self._log_file