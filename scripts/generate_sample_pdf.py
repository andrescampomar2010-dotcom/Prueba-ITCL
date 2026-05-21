"""Genera el PDF de ejemplo en español para la prueba.

Crea un "Manual de Procedimientos Internos" con texto seleccionable y varias
páginas. Incluye una cifra monetaria (presupuesto) necesaria para el caso de
prueba T3 (conversión de divisa).

Uso:
    python scripts/generate_sample_pdf.py [ruta_salida.pdf]
"""
from __future__ import annotations

import os
import sys

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

DEFAULT_OUTPUT = os.path.join("data", "pdfs", "manual_ejemplo.pdf")


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "titulo": ParagraphStyle("titulo", parent=base["Title"], fontSize=22,
                                 spaceAfter=18),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=15,
                             spaceBefore=6, spaceAfter=10),
        "cuerpo": ParagraphStyle("cuerpo", parent=base["BodyText"], fontSize=11,
                                 leading=16, alignment=TA_JUSTIFY, spaceAfter=8),
    }


# (encabezado_de_seccion, [parrafos]) — cada sección ocupa su propia página.
SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Manual de Procedimientos Internos — ACME Soluciones S.L.",
        [
            "Este manual recoge los procedimientos internos del Departamento de "
            "Operaciones de ACME Soluciones S.L. Su objetivo es servir de "
            "referencia única para el personal en materia de jornada laboral, "
            "gestión de gastos, vacaciones y seguridad de la información.",
            "El documento es de uso interno. Cualquier duda sobre su contenido "
            "debe dirigirse al Departamento de Recursos Humanos. La versión "
            "vigente de este manual es la 2.3, aprobada por la Dirección.",
            "Índice: la Sección 1 trata la jornada laboral; la Sección 2, la "
            "gestión de gastos y el presupuesto; la Sección 3, la política de "
            "vacaciones; la Sección 4, la seguridad de la información; y la "
            "Sección 5, los canales de contacto y soporte.",
        ],
    ),
    (
        "Sección 1 — Jornada laboral y horario",
        [
            "La jornada laboral ordinaria es de 40 horas semanales, distribuidas "
            "de lunes a viernes. El horario general de oficina es de 9:00 a "
            "18:00, con una pausa para comer de una hora.",
            "Se aplica un esquema de horario flexible: la entrada puede "
            "realizarse entre las 8:00 y las 10:00, ajustando en consecuencia la "
            "hora de salida. Las franjas de presencia obligatoria son de 10:00 a "
            "13:00 y de 15:00 a 17:00.",
            "El registro horario es obligatorio y se realiza a través de la "
            "aplicación corporativa de fichaje. Las horas extraordinarias deben "
            "ser autorizadas previamente por la persona responsable del equipo.",
        ],
    ),
    (
        "Sección 2 — Gestión de gastos y presupuesto",
        [
            "El presupuesto anual asignado al Departamento de Operaciones "
            "asciende a 12.500 EUR. Este importe cubre material de oficina, "
            "licencias de software de equipo y gastos de formación.",
            "Todo gasto con cargo a este presupuesto requiere aprobación previa. "
            "Los gastos inferiores a 200 EUR los aprueba la persona responsable "
            "del equipo; los iguales o superiores a 200 EUR requieren la "
            "aprobación de la Dirección de Operaciones.",
            "Las solicitudes de reembolso se presentan en un plazo máximo de "
            "treinta días desde la fecha del gasto, adjuntando siempre la "
            "factura correspondiente. Los reembolsos se abonan en la nómina del "
            "mes siguiente a su aprobación.",
        ],
    ),
    (
        "Sección 3 — Política de vacaciones",
        [
            "Cada empleado dispone de 23 días laborables de vacaciones al año. "
            "El periodo de devengo coincide con el año natural, del 1 de enero "
            "al 31 de diciembre.",
            "Las solicitudes de vacaciones se cursan a través del portal del "
            "empleado con una antelación mínima de quince días naturales. La "
            "persona responsable del equipo debe aprobar o rechazar la solicitud "
            "en un plazo de cinco días laborables.",
            "Se pueden disfrutar como máximo quince días laborables consecutivos, "
            "salvo autorización expresa. Los días no disfrutados pueden trasladarse "
            "hasta el 31 de marzo del año siguiente.",
        ],
    ),
    (
        "Sección 4 — Seguridad de la información",
        [
            "Toda persona usuaria debe proteger la información corporativa a la "
            "que tiene acceso. Las contraseñas deben tener una longitud mínima de "
            "doce caracteres y combinar mayúsculas, minúsculas, números y "
            "símbolos.",
            "Las contraseñas se renuevan obligatoriamente cada noventa días y no "
            "pueden reutilizarse las cinco últimas. El acceso a los sistemas "
            "críticos exige además un segundo factor de autenticación.",
            "Está prohibido compartir credenciales o almacenarlas en texto "
            "plano. Cualquier incidente de seguridad, como la pérdida de un "
            "dispositivo o un correo sospechoso, debe comunicarse de inmediato al "
            "equipo de Seguridad de la Información.",
        ],
    ),
    (
        "Sección 5 — Contacto y soporte",
        [
            "El soporte técnico interno atiende de lunes a viernes de 8:00 a "
            "19:00. Las incidencias se registran a través del portal de soporte "
            "o, en casos urgentes, por el teléfono de extensión 4000.",
            "Las consultas sobre recursos humanos, nóminas y vacaciones se "
            "dirigen al Departamento de Recursos Humanos mediante el portal del "
            "empleado.",
            "Este manual se revisa con periodicidad anual. Las propuestas de "
            "mejora pueden enviarse al Departamento de Operaciones, responsable "
            "de su mantenimiento y actualización.",
        ],
    ),
]


def build_pdf(output_path: str) -> str:
    """Construye el PDF de ejemplo y devuelve la ruta de salida."""
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    styles = _styles()

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=2.5 * cm, bottomMargin=2.5 * cm,
        title="Manual de Procedimientos Internos",
    )

    flowables: list = []
    for index, (heading, paragraphs) in enumerate(SECTIONS):
        style = styles["titulo"] if index == 0 else styles["h1"]
        flowables.append(Paragraph(heading, style))
        flowables.append(Spacer(1, 0.3 * cm))
        for text in paragraphs:
            flowables.append(Paragraph(text, styles["cuerpo"]))
        # Salto de página explícito entre secciones (una sección por página).
        if index < len(SECTIONS) - 1:
            from reportlab.platypus import PageBreak
            flowables.append(PageBreak())

    doc.build(flowables)
    return output_path


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    output = argv[0] if argv else DEFAULT_OUTPUT
    path = build_pdf(output)
    print(f"PDF de ejemplo generado en: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
