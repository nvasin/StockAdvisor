import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger():
    """
    Логи пишутся и в консоль, и в файл с ротацией.
    - logger.debug("Это сообщение уровня DEBUG")
    - logger.info("Это сообщение уровня INFO")
    - logger.warning("Это сообщение уровня WARNING")
    - logger.error("Это сообщение уровня ERROR")
    - logger.critical("Это сообщение уровня CRITICAL")
    """
    
    # Место логгирования
    log_dir = os.path.join(os.getcwd(), "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "StockAdvisor.log")

    # Уровень логгирования
    log_level = logging.INFO

    logger = logging.getLogger("StockAdvisorLogger")
    logger.setLevel(log_level)

    # Формат логов
    log_format = logging.Formatter("%(asctime)s [%(levelname)s] [%(filename)s] %(message)s")

    # Обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)

    # Обработчик для файла с ротацией
    file_handler = RotatingFileHandler(
        log_file_path,  # Имя файла
        maxBytes=5_000_000,  # Максимальный размер файла 5 MB
        backupCount=3        # Количество резервных файлов
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_format)

    # Добавляем обработчики к логгеру
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Создаём экземпляр логгера
logger = setup_logger()
