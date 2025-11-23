import os
import glob

from train import EntrenadorConnect4


def extraer_ep(desde_ruta: str) -> int:
    """
    Dado un nombre tipo:
        snapshot_20251122_183339_ep0040.json
    devuelve el número de episodio (por ej. 40).
    Si algo falla, devuelve 0 para no romper el orden.
    """
    nombre = os.path.basename(desde_ruta)
    try:
        parte = nombre.split("_ep")[-1]  # "0040.json"
        parte = parte.split(".")[0]      # "0040"
        return int(parte)
    except Exception:
        return 0


def cargar_q_desde(ruta_modelo: str) -> dict:
    """
    Usa EntrenadorConnect4.cargar para leer una Q-table desde un archivo json
    (snapshot_* o modelo final).
    """
    entrenador_tmp = EntrenadorConnect4(archivo_modelo=ruta_modelo)
    return entrenador_tmp.cargar()


def jugar_partida_q_vs_q(
    entrenador: EntrenadorConnect4,
    q_agente: dict,
    q_oponente: dict,
    agente_empieza: bool = True,
) -> int:
    """
    Juega UNA partida entre:
      - Jugador 'agente' con Q-table q_agente
      - Jugador 'oponente' con Q-table q_oponente

    Ambos usan elegir_accion_q del entrenador.
    Devuelve:
      +1 si gana el agente
       0 si hay empate
      -1 si gana el oponente.
    """
    import numpy as np

    tablero = np.zeros((entrenador.FILAS, entrenador.COLUMNAS), dtype=int)

    # En el tablero, -1 siempre empieza
    jugador_tablero = -1

    if agente_empieza:
        jugador_agente = -1
        jugador_oponente = 1
    else:
        jugador_agente = 1
        jugador_oponente = -1

    while True:
        # ¿Alguien ganó?
        if entrenador.hay_ganador(tablero, jugador_agente):
            return 1
        if entrenador.hay_ganador(tablero, jugador_oponente):
            return -1

        # ¿Empate?
        movimientos = entrenador.movimientos_legales(tablero)
        if not movimientos:
            return 0

        # Turno del que mueve ahora en el tablero
        if jugador_tablero == jugador_agente:
            accion = entrenador.elegir_accion_q(tablero, jugador_tablero, q_agente)
        else:
            accion = entrenador.elegir_accion_q(tablero, jugador_tablero, q_oponente)

        if accion is None:
            return 0

        tablero = entrenador.colocar(tablero, accion, jugador_tablero)
        jugador_tablero = -jugador_tablero


def evaluar_q_vs_q(
    entrenador: EntrenadorConnect4,
    q_agente: dict,
    q_oponente: dict,
    partidas: int = 200,
) -> dict:
    """
    Juega muchas partidas agente vs oponente, alternando quién empieza,
    y devuelve winrate y conteo de resultados.
    """
    victorias = 0
    empates = 0
    derrotas = 0

    for i in range(partidas):
        agente_empieza = (i % 2 == 0)
        r = jugar_partida_q_vs_q(entrenador, q_agente, q_oponente, agente_empieza)
        if r > 0:
            victorias += 1
        elif r < 0:
            derrotas += 1
        else:
            empates += 1

    total = victorias + empates + derrotas
    winrate = victorias / total if total > 0 else 0.0

    return {
        "winrate": winrate,
        "victorias": victorias,
        "empates": empates,
        "derrotas": derrotas,
        "partidas": total,
    }


if __name__ == "__main__":
    # 1. Buscar snapshots en el directorio actual
    patrones = ["snapshot_*_ep*.json"]
    snapshots: list[str] = []
    for patron in patrones:
        snapshots.extend(glob.glob(patron))

    if len(snapshots) < 2:
        print("Necesito al menos 2 snapshots para comparar (nuevo vs viejo).")
    else:
        # 2. Ordenar por número de episodio
        snapshots_ordenados = sorted(snapshots, key=extraer_ep)

        print("\n=== Evaluación de snapshots entre SÍ MISMOS (nuevo vs anterior) ===\n")
        print("Nuevo snapshot\t\tEp_nuevo\tViejo snapshot\t\tEp_viejo\tWinrate\tV\tE\tD")

        entrenador = EntrenadorConnect4()

        # 3. Comparar cada snapshot contra el anterior inmediato
        for i in range(1, len(snapshots_ordenados)):
            ruta_viejo = snapshots_ordenados[i - 1]
            ruta_nuevo = snapshots_ordenados[i]

            ep_viejo = extraer_ep(ruta_viejo)
            ep_nuevo = extraer_ep(ruta_nuevo)

            q_viejo = cargar_q_desde(ruta_viejo)
            q_nuevo = cargar_q_desde(ruta_nuevo)

            resultado = evaluar_q_vs_q(entrenador, q_nuevo, q_viejo, partidas=200)
            w = resultado["winrate"]
            v = resultado["victorias"]
            e = resultado["empates"]
            d = resultado["derrotas"]

            print(
                f"{os.path.basename(ruta_nuevo):<28}\t"
                f"{ep_nuevo:>4}\t"
                f"{os.path.basename(ruta_viejo):<28}\t"
                f"{ep_viejo:>4}\t"
                f"{w:0.3f}\t{v}\t{e}\t{d}"
            )

        print("\nListo. Esa tabla te sirve como 'curva de aprendizaje' contra tus propias versiones.")
