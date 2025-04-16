import logging
import sys

def setup_logging():
    """Configura la salida de logs tanto en consola como en archivo.

    Se establece el nivel de log a DEBUG para el logger raíz. Se define un formateador
    con la hora, el nivel del mensaje, el nombre del logger y el mensaje (seguido de un salto de línea).
    Se crea un handler para la consola (stdout) que muestra mensajes de nivel INFO o superior,
    y otro handler para archivo ("app.log") que almacena todos los mensajes a nivel DEBUG o superior.
    Finalmente, se agrega ambos handlers al logger raíz y se imprime un mensaje de confirmación.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s\n')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('logging succesfully configured')
