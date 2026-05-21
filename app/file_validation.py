"""Validación de archivos PDF subidos.

Proporciona funciones para validar PDFs antes de procesarlos.
"""
from __future__ import annotations

import os
from typing import NamedTuple


class ValidationError(ValueError):
    """Error de validación de archivo PDF."""
    pass


class FileValidationResult(NamedTuple):
    """Resultado de validación de archivo.
    
    Attributes:
        valid: True si el archivo pasa todas las validaciones
        error: Mensaje de error si valid=False
        size_bytes: Tamaño del archivo en bytes
        is_pdf: True si el archivo es un PDF válido
    """
    valid: bool
    error: str = ""
    size_bytes: int = 0
    is_pdf: bool = False


# Límites configurables
MAX_PDF_SIZE_MB = 50  # 50 MB máximo
MIN_PDF_SIZE_BYTES = 100  # Al menos 100 bytes
PDF_HEADER = b"%PDF-"


def validate_pdf_file(file_bytes: bytes, filename: str) -> FileValidationResult:
    """Valida un archivo PDF antes de procesarlo.
    
    Verificaciones:
    - Nombre de archivo válido (no vacío, sin caracteres peligrosos)
    - Tamaño dentro de límites (100 bytes - 50 MB)
    - Header PDF válido (%PDF-)
    - Extensión .pdf
    
    Args:
        file_bytes: Contenido del archivo en bytes
        filename: Nombre del archivo
        
    Returns:
        FileValidationResult con resultado de validación
        
    Example:
        result = validate_pdf_file(file_contents, "document.pdf")
        if not result.valid:
            show_error(f"Archivo inválido: {result.error}")
    """
    size_bytes = len(file_bytes)
    
    # Validar nombre
    if not filename or not isinstance(filename, str):
        return FileValidationResult(valid=False, error="Nombre de archivo inválido")
    
    filename_clean = filename.strip()
    if not filename_clean:
        return FileValidationResult(valid=False, error="Nombre de archivo no puede estar vacío")
    
    # Validar extensión
    if not filename_clean.lower().endswith(".pdf"):
        return FileValidationResult(
            valid=False,
            error=f"Archivo debe ser PDF. Recibido: {os.path.splitext(filename_clean)[1] or '(sin extensión)'}",
            size_bytes=size_bytes,
            is_pdf=False
        )
    
    # Validar caracteres peligrosos en nombre
    dangerous_chars = ["\\", "/", ":", "*", "?", '"', "<", ">", "|", "\0"]
    for char in dangerous_chars:
        if char in filename_clean:
            return FileValidationResult(
                valid=False,
                error=f"Nombre contiene carácter no permitido: '{char}'",
                size_bytes=size_bytes,
                is_pdf=False
            )
    
    # Validar tamaño mínimo
    if size_bytes < MIN_PDF_SIZE_BYTES:
        return FileValidationResult(
            valid=False,
            error=f"Archivo demasiado pequeño ({size_bytes} bytes). Mínimo: {MIN_PDF_SIZE_BYTES} bytes",
            size_bytes=size_bytes,
            is_pdf=False
        )
    
    # Validar tamaño máximo
    max_size_bytes = MAX_PDF_SIZE_MB * 1024 * 1024
    if size_bytes > max_size_bytes:
        size_mb = size_bytes / (1024 * 1024)
        return FileValidationResult(
            valid=False,
            error=f"Archivo demasiado grande ({size_mb:.1f} MB). Máximo: {MAX_PDF_SIZE_MB} MB",
            size_bytes=size_bytes,
            is_pdf=False
        )
    
    # Validar header PDF
    if not file_bytes.startswith(PDF_HEADER):
        return FileValidationResult(
            valid=False,
            error="Archivo no es un PDF válido (header incorrecto)",
            size_bytes=size_bytes,
            is_pdf=False
        )
    
    return FileValidationResult(
        valid=True,
        error="",
        size_bytes=size_bytes,
        is_pdf=True
    )
