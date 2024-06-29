import pandas as pd
import os
import conorm
from xml.etree import ElementTree
import os, shutil
import pathlib
import base64
import datetime
import io
from dash import dash_table


def parse_file(contents, filename, date):
    print("Subiendo fichero...")
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        fichero = io.StringIO(decoded.decode('utf-8'))
    except Exception as e:
        print(e)
        return html.Div([
            'Error leyendo fichero '+filename
        ])
    return fichero


def obtener_subdirectorios(ruta):
    return [d for d in os.listdir(ruta) if os.path.isdir(os.path.join(ruta, d))]
