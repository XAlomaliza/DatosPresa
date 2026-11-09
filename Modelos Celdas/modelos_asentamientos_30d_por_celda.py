import argparse, csv, numpy as np
from bisect import bisect_right
from collections import defaultdict
from datetime import date, datetime, timedelta
from math import isfinite
from pathlib import Path
from statistics import median
from openpyxl import load_workbook
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

HORIZONTE = 30
FIN_ENTRENAMIENTO = date(2020, 8, 11)
FIN_VALIDACION = date(2023, 7, 2)
SEMILLA = 42
ESTADOS_VALIDOS = {"DENTRO_IQR", "ATIPICO_IQR"} #EL ESTADO: "NO_EVALUABLE" NO SE USA

def serie_diaria(archivo, celda):
    lecturas = defaultdict(list)
    libro = load_workbook(archivo, read_only=True, data_only=True)
    try:
        filas = libro.active.values
        encabezados = next(filas)
        fecha_col, instrumento_col, delta_col, estado_col = map(
            encabezados.index, ("Fecha", "Instrumento", "Delta_m_Analisis", "Estado_IQR")
        )
        for fila in filas:
            estado = fila[estado_col]
            if estado == "NO_EVALUABLE":
                continue
            fecha, delta = fila[fecha_col], fila[delta_col]
            if not isinstance(fecha, date) or not isinstance(delta, (int, float)):
                continue
            if not isfinite(delta):
                continue
            if fila[instrumento_col] != celda or estado not in ESTADOS_VALIDOS:
                raise ValueError(f"{archivo.name}: instrumento o estado inesperado en {fecha}.")
            if isinstance(fecha, datetime):
                fecha = fecha.date()
            lecturas[fecha].append(delta * 1000)
    finally:
        libro.close()
    return {fecha: median(valores) for fecha, valores in sorted(lecturas.items())}

def variables_historicas(serie):
    fechas = sorted(serie)
    variables = {}
    for fecha in fechas:
        actual = serie[fecha]
        cambios = []
        for dias in (7, 30):
            # Se usa la última lectura en o antes del día buscado.
            indice = bisect_right(fechas, fecha - timedelta(days=dias)) - 1
            anterior = serie[fechas[indice]] if indice >= 0 else actual
            cambios.append(actual - anterior)
        variables[fecha] = (actual, *cambios)
    return variables

def pares(serie, variables, horizontes, hasta, desde=date.min):
    entradas, cambios = [], []
    for fecha, historial in variables.items():
        if fecha < desde or fecha >= hasta:
            continue
        actual = historial[0]
        for dias in horizontes:
            objetivo = fecha + timedelta(days=dias)
            if objetivo <= hasta and objetivo in serie:
                entradas.append((*historial, dias))
                cambios.append(serie[objetivo] - actual)
    if not entradas:
        raise ValueError(f"Faltan pares válidos entre {desde} y {hasta}.")
    return np.asarray(entradas, dtype=float), np.asarray(cambios, dtype=float)

def modelos():
    return {
        "XGBoost": XGBRegressor(
            objective="reg:absoluteerror", base_score=0.0, n_estimators=100,
            max_depth=3, learning_rate=0.05, min_child_weight=20,
            tree_method="hist", random_state=SEMILLA, n_jobs=4,
        ),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_leaf=30,
            random_state=SEMILLA, n_jobs=4,
        ),
        "Árbol de decisión": DecisionTreeRegressor(
            max_depth=4, min_samples_leaf=30, random_state=SEMILLA,
        ),
    }

def errores(real, estimado):
    diferencia = real - estimado
    return float(np.abs(diferencia).mean()), float(np.sqrt(np.square(diferencia).mean()))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directorio", nargs="?", type=Path, default=Path(r"D:\Datos Represa"))
    parser.add_argument("--salida", type=Path, default=Path("pronosticos_cuatro_modelos_30d.csv"))
    argumentos = parser.parse_args()
    horizontes = range(1, HORIZONTE + 1)
    pronosticos = []

    for numero in range(1, 8):
        celda = f"CA-{numero}"
        archivo = argumentos.directorio / f"{celda}_limpio.xlsx"
        copia_ca6 = argumentos.directorio / "CA-6_limpio(1).xlsx"
        if numero == 6 and copia_ca6.exists():
            archivo = copia_ca6

        serie = serie_diaria(archivo, celda)
        variables = variables_historicas(serie)
        entrenamiento = pares(serie, variables, horizontes, FIN_ENTRENAMIENTO)
        x_val, y_val = pares(
            serie, variables, (HORIZONTE,), FIN_VALIDACION,
            desde=FIN_ENTRENAMIENTO + timedelta(days=1),
        )
        # La persistencia supone un cambio de 0 mm desde la lectura actual.
        base = errores(y_val, np.zeros_like(y_val))
        print(f"\n{celda} | {len(entrenamiento[1])} pares de entrenamiento | {len(y_val)} de validación a 30 días")
        print(f"{'Modelo':<20} {'MAE (mm)':>10} {'RMSE (mm)':>10}")
        print(f"{'Persistencia':<20} {base[0]:10.2f} {base[1]:10.2f}")
        candidatos = modelos()
        resultados = {}
        for nombre, modelo in candidatos.items():
            modelo.fit(*entrenamiento)
            mae, rmse = errores(y_val, modelo.predict(x_val))
            resultados[nombre] = (mae, rmse)
            print(f"{nombre:<20} {mae:10.2f} {rmse:10.2f}")

        elegido = min(resultados, key=resultados.get)
        print(f"Mejor entrenado por MAE: {elegido}")

        # Reajuste posterior a la comparación, para pronosticar desde la última lectura disponible.
        ultimo_dia = max(serie)
        modelo_final = candidatos[elegido]
        modelo_final.fit(*pares(serie, variables, horizontes, ultimo_dia))
        entradas_futuras = np.asarray([(*variables[ultimo_dia], dias) for dias in horizontes])
        nivel_actual = serie[ultimo_dia]
        niveles = nivel_actual + modelo_final.predict(entradas_futuras)
        for dias, nivel in enumerate(niveles, 1):
            pronosticos.append((celda, ultimo_dia, ultimo_dia + timedelta(days=dias), dias,
                                elegido, f"{nivel:.6f}", f"{nivel_actual:.6f}", "PRONOSTICO"))

    with argumentos.salida.open("w", encoding="utf-8-sig", newline="") as archivo_csv:
        escritor = csv.writer(archivo_csv)
        escritor.writerow(["Instrumento", "Fecha_Ultima_Lectura", "Fecha_Pronostico", "Horizonte_dias",
                           "Modelo", "Delta_mm_Pronosticado", "Persistencia_mm", "Tipo"])
        escritor.writerows(pronosticos)
    print(f"\n{len(pronosticos)} estimaciones guardadas en {argumentos.salida.resolve()}.")

if __name__ == "__main__":
    main()