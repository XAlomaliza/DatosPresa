"""Entrena y evalúa por separado las siete celdas a un horizonte de 30 días."""

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
from math import isfinite
from pathlib import Path
from statistics import median

import numpy as np
from openpyxl import load_workbook
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


HORIZONTE = timedelta(days=30)
FIN_ENTRENAMIENTO = date(2020, 8, 11)
FIN_VALIDACION = date(2023, 7, 2)
SEMILLA = 42

PERSISTENCIA = "Persistencia"
NO_EVALUABLE = "NO_EVALUABLE"
ESTADOS_VALIDOS = {"DENTRO_IQR", "ATIPICO_IQR"}


def serie_diaria(archivo, instrumento):
    lecturas = defaultdict(list)
    excluidas = 0

    libro = load_workbook(archivo, read_only=True, data_only=True)
    try:
        filas = libro.active.values
        encabezados = next(filas)

        pos_fecha = encabezados.index("Fecha")
        pos_instrumento = encabezados.index("Instrumento")
        pos_delta = encabezados.index("Delta_m_Analisis")
        pos_estado = encabezados.index("Estado_IQR")

        for fila in filas:
            estado = fila[pos_estado]

            # NO_EVALUABLE identifica una fila sin asentamiento válido en KNIME.
            # Se conserva en el Excel para trazabilidad y se excluye del modelo.
            if estado == NO_EVALUABLE:
                excluidas += 1
                continue

            fecha = fila[pos_fecha]
            delta = fila[pos_delta]

            if not isinstance(fecha, date) or not isinstance(delta, (int, float)):
                continue
            if not isfinite(delta):
                continue
            if fila[pos_instrumento] != instrumento:
                raise ValueError(
                    f"{archivo.name} contiene una lectura de otro instrumento."
                )
            if isinstance(fecha, datetime):
                fecha = fecha.date()
            if estado not in ESTADOS_VALIDOS:
                raise ValueError(
                    f"{archivo.name}: Estado_IQR desconocido en {fecha}: {estado!r}."
                )

            # Se conservan DENTRO_IQR y ATIPICO_IQR si el valor es válido.
            lecturas[fecha].append(delta)
    finally:
        libro.close()

    # Si hay varias lecturas válidas el mismo día, se usa su mediana.
    serie = {fecha: median(valores) for fecha, valores in lecturas.items()}
    return serie, excluidas


def separar_pares(serie, instrumento):
    bloques = {"entrenamiento": [], "validacion": [], "prueba": []}

    for fecha in sorted(serie):
        fecha_objetivo = fecha + HORIZONTE

        # Cada par requiere una lectura exactamente 30 días después.
        if fecha_objetivo not in serie:
            continue

        if fecha_objetivo <= FIN_ENTRENAMIENTO:
            bloque = "entrenamiento"
        elif FIN_ENTRENAMIENTO < fecha and fecha_objetivo <= FIN_VALIDACION:
            bloque = "validacion"
        elif fecha > FIN_VALIDACION:
            bloque = "prueba"
        else:
            # Se excluyen los pares con fechas en dos periodos distintos.
            continue

        bloques[bloque].append((serie[fecha], serie[fecha_objetivo]))

    for nombre, pares in bloques.items():
        if not pares:
            raise ValueError(f"{instrumento}: no hay pares a 30 días en {nombre}.")

    return {
        nombre: np.asarray(pares, dtype=float)
        for nombre, pares in bloques.items()
    }


def modelos_candidatos():
    return {
        "Regresión lineal": (LinearRegression(), False),
        "Árbol de decisión": (
            DecisionTreeRegressor(
                max_depth=6,
                min_samples_leaf=50,
                random_state=SEMILLA,
            ),
            False,
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=10,
                random_state=SEMILLA,
                n_jobs=4,
            ),
            False,
        ),
        "Gradient Boosting absoluto": (
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.05,
                max_iter=100,
                max_leaf_nodes=7,
                min_samples_leaf=100,
                early_stopping=False,
                random_state=SEMILLA,
            ),
            True,
        ),
    }


def predecir(pares, modelo, usa_cambio):
    actual = pares[:, 0]

    # La persistencia usa la lectura actual como pronóstico.
    if modelo is None:
        return actual

    prediccion = modelo.predict(pares[:, :1])

    # Si el modelo predice el cambio, se suma a la lectura actual.
    if usa_cambio:
        prediccion += actual

    return prediccion


def calcular_errores(pares, prediccion):
    error_mm = 1000 * (pares[:, 1] - prediccion)
    mae = float(np.abs(error_mm).mean())
    rmse = float(np.sqrt(np.square(error_mm).mean()))
    return mae, rmse


def seleccionar_en_validacion(entrenamiento, validacion):
    modelos = {PERSISTENCIA: (None, False), **modelos_candidatos()}
    resultados = {}

    for nombre, (modelo, usa_cambio) in modelos.items():
        if modelo is not None:
            objetivo = entrenamiento[:, 1]
            if usa_cambio:
                objetivo = objetivo - entrenamiento[:, 0]
            modelo.fit(entrenamiento[:, :1], objetivo)

        prediccion = predecir(validacion, modelo, usa_cambio)
        resultados[nombre] = calcular_errores(validacion, prediccion)

    # Se elige por menor MAE y se usa el RMSE como desempate.
    seleccionado = min(resultados, key=resultados.get)
    return seleccionado, modelos[seleccionado], resultados


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directorio",
        nargs="?",
        type=Path,
        default=Path(r"D:\Datos Represa"),
    )
    carpeta = parser.parse_args().directorio
    selecciones = {}

    for numero in range(1, 8):
        instrumento = f"CA-{numero}"
        archivo = carpeta / f"{instrumento}_limpio.xlsx"

        copia_ca6 = carpeta / "CA-6_limpio(1).xlsx"
        if numero == 6 and copia_ca6.exists():
            archivo = copia_ca6

        serie, excluidas = serie_diaria(archivo, instrumento)
        bloques = separar_pares(serie, instrumento)

        entrenamiento = bloques["entrenamiento"]
        validacion = bloques["validacion"]
        seleccionado, modelo, resultados = seleccionar_en_validacion(
            entrenamiento, validacion
        )
        selecciones[instrumento] = (seleccionado, modelo, bloques["prueba"])

        print(
            f"\n{instrumento} — {len(entrenamiento)} filas de entrenamiento y "
            f"{len(validacion)} de validación"
        )
        print(f"Filas {NO_EVALUABLE} excluidas: {excluidas}")
        print(f"{'Modelo':<30} {'MAE (mm)':>10} {'RMSE (mm)':>10}")

        for nombre, (mae, rmse) in resultados.items():
            print(f"{nombre:<30} {mae:>10.2f} {rmse:>10.2f}")

        print(f"Selección en validación: {seleccionado}")

    # Se calculan las métricas de prueba después de elegir un modelo por celda.
    print("\nPrueba cronológica a 30 días")
    print(
        f"{'Celda':<6} {'Filas':>5} {'Modelo elegido':<30} "
        f"{'MAE elegido':>12} {'RMSE elegido':>12} "
        f"{'MAE base':>10} {'RMSE base':>10}"
    )

    for instrumento, (nombre, modelo, prueba) in selecciones.items():
        mae, rmse = calcular_errores(prueba, predecir(prueba, *modelo))
        mae_base, rmse_base = calcular_errores(prueba, prueba[:, 0])

        print(
            f"{instrumento:<6} {len(prueba):>5} {nombre:<30} "
            f"{mae:>12.2f} {rmse:>12.2f} "
            f"{mae_base:>10.2f} {rmse_base:>10.2f}"
        )

    print("Todas las métricas de prueba se expresan en milímetros.")


if __name__ == "__main__":
    main()