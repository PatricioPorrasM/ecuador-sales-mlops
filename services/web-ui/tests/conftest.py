"""
Configuración de pytest para los tests del servicio web-ui.

Añade el directorio raíz del servicio al sys.path para que los módulos
locales (app, kafka_producer, metrics) sean importables desde los tests.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
