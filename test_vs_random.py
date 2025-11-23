import numpy as np
import random

from policy import CetinaSalasSabogal
from groups.Random.random_policy import RandomPolicy

FILAS = 6
COLUMNAS = 7

def columnas_validas(tablero: np.ndarray) -> list[int]:
    """Columnas donde todavía se puede jugar (la casilla de arriba está vacía)."""
    return [c for c in range(COLUMNAS) if tablero[0, c] == 0]

def caer_ficha(tablero: np.ndarray, col: int, jugador: int) -> bool:
    """
    Deja caer una ficha en la columna dada.
    Devuelve True si se pudo colocar, False si la columna estaba llena.
    """
    for fila in range(FILAS - 1, -1, -1):
        if tablero[fila, col] == 0:
            tablero[fila, col] = jugador
            return True
    return False

def cuatro_en_linea(tablero: np.ndarray, jugador: int) -> bool:
    """Comprueba si 'jugador' tiene 4 en línea."""
    # Horizontal
    for f in range(FILAS):
        for c in range(COLUMNAS - 3):
            if np.all(tablero[f, c:c + 4] == jugador):
                return True

    # Vertical
    for f in range(FILAS - 3):
        for c in range(COLUMNAS):
            if np.all(tablero[f:f + 4, c] == jugador):
                return True

    # Diagonal principal (\)
    for f in range(FILAS - 3):
        for c in range(COLUMNAS - 3):
            if all(tablero[f + k, c + k] == jugador for k in range(4)):
                return True

    # Diagonal secundaria (/)
    for f in range(FILAS - 3):
        for c in range(COLUMNAS - 3):
            if all(tablero[f + 3 - k, c + k] == jugador for k in range(4)):
                return True

    return False

def jugar_una_partida(empieza_agente: bool, verbose: bool = False) -> int:
    """
    Juega UNA partida entre:
      - Tu CetinaSalasSabogal
      - RandomPolicy de groups

    empieza_agente = True  → tu agente empieza (fichas -1)
                      False → RandomPolicy empieza

    Devuelve:
        +1 si gana tu agente
         0 si empate
        -1 si gana RandomPolicy
    """
    tablero = np.zeros((FILAS, COLUMNAS), dtype=int)

    agente = CetinaSalasSabogal()
    random_policy = RandomPolicy()

    # Algunos policies tienen mount(timeout), otros mount() sin argumentos
    for p in (agente, random_policy):
        try:
            p.mount(None)
        except TypeError:
            try:
                p.mount()
            except Exception:
                pass

    # Convención: el jugador que está en el tablero:
    # -1 mueve primero, luego 1, luego -1, etc.
    jugador_tablero = -1

    # Asignamos quién es quién
    if empieza_agente:
        jugador_agente = -1
        jugador_random = 1
    else:
        jugador_agente = 1
        jugador_random = -1

    while True:
        # ¿Alguien ganó?
        if cuatro_en_linea(tablero, jugador_agente):
            if verbose:
                print("Gana tu agente.")
            return 1
        if cuatro_en_linea(tablero, jugador_random):
            if verbose:
                print("Gana RandomPolicy.")
            return -1

        # ¿Tablero lleno → empate?
        if not columnas_validas(tablero):
            if verbose:
                print("Empate.")
            return 0

        # Turno de quien corresponde
        if jugador_tablero == jugador_agente:
            col = agente.act(tablero.copy())
        else:
            col = random_policy.act(tablero.copy())

        # Aseguramos que la acción sea legal; si no, jugamos una al azar
        legales = columnas_validas(tablero)
        if col not in legales:
            col = random.choice(legales)

        caer_ficha(tablero, col, jugador_tablero)

        if verbose:
            print(f"Jugador {jugador_tablero} juega columna {col}")
            # print(tablero)  # descomenta si quieres ver el tablero

        jugador_tablero = -jugador_tablero  # cambia de jugador


def main():
    random.seed(0)
    np.random.seed(0)

    n_partidas = 100
    victorias = 0
    empates = 0
    derrotas = 0

    for i in range(n_partidas):
        # Alternar quién empieza: en pares empieza el agente, en impares Random
        empieza_agente = (i % 2 == 0)
        resultado = jugar_una_partida(empieza_agente, verbose=False)

        if resultado > 0:
            victorias += 1
        elif resultado < 0:
            derrotas += 1
        else:
            empates += 1

    print("\n=== RESULTADOS vs RandomPolicy ===")
    print(f"Partidas jugadas: {n_partidas}")
    print(f"Victorias agente: {victorias}")
    print(f"Empates:          {empates}")
    print(f"Derrotas agente:  {derrotas}")
    print(f"Winrate:          {victorias / n_partidas:.3f}")


if __name__ == "__main__":
    main()