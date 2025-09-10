"""
Services module for vitalDSP webapp.
"""

from .data_service import DataService, get_data_service, data_service

__all__ = [
    'DataService',
    'get_data_service',
    'data_service'
]
