"""
Configuración de pytest para los tests del servicio ml-inference.

Añade el directorio raíz del servicio al sys.path para que los módulos
locales (app, model_loader, kafka_producer, metrics) sean importables.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
