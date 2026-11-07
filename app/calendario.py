"""Calendario epidemiológico (regla del INS, igual a MMWR): semanas de domingo a sábado.

La semana 1 es la que contiene el 4 de enero, por lo que un año tiene 52 o 53 semanas.
El notebook usa el calendario de los archivos de clima; la app lo calcula y lo coteja
con esos archivos al cargar el histórico.
"""
from datetime import date, timedelta


def inicio_ano_epidemiologico(ano):
    """Domingo en el que empieza la semana 1 del año."""
    cuatro_enero = date(int(ano), 1, 4)
    return cuatro_enero - timedelta(days=(cuatro_enero.weekday() + 1) % 7)


def semanas_del_ano(ano):
    return (inicio_ano_epidemiologico(int(ano) + 1) - inicio_ano_epidemiologico(ano)).days // 7


def semana_epidemiologica(fecha_inicio):
    """Año y semana de una semana que empieza en domingo."""
    if fecha_inicio.weekday() != 6:
        raise ValueError(f'{fecha_inicio:%Y-%m-%d} no es domingo; las semanas epidemiológicas inician en domingo.')
    ano = fecha_inicio.year + 1
    while inicio_ano_epidemiologico(ano) > fecha_inicio:
        ano -= 1
    return ano, (fecha_inicio - inicio_ano_epidemiologico(ano)).days // 7 + 1


def inicio_semana(ano, semana):
    if not 1 <= int(semana) <= semanas_del_ano(ano):
        raise ValueError(f'El año {ano} tiene {semanas_del_ano(ano)} semanas epidemiológicas.')
    return inicio_ano_epidemiologico(ano) + timedelta(weeks=int(semana) - 1)
