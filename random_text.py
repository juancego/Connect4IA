import numpy as np
from groups.Random.random_policy import RandomPolicy
from policy import CetinaSalasSabogal   # tu policy en la raíz


FILAS = 6
COLUMNAS = 7


def movimientos_legales(tablero):
    return [c for c in range(COLUMNAS) if tablero[0, c] == 0]


def colocar(tablero, col, jugador):
    for fila in range(FILAS - 1, -1, -1):
        if tablero[fila, col] == 0:
            nuevo = tablero.copy()
            nuevo[fila, col] = jugador
            return nuevo
    return tablero


def hay_ganador(tablero, jugador):
    # Horizontal
    for f in range(FILAS):
        for c in range(COLUMNAS - 3):
            if np.all(tablero[f, c:c+4] == jugador):
                return True

    # Vertical
    for f in range(FILAS - 3):
        for c in range(COLUMNAS):
            if np.all(tablero[f:f+4, c] == jugador):
                return True

    # Diagonal \
    for f in range(FILAS - 3):
        for c in range(COLUMNAS - 3):
            if all(tablero[f+i, c+i] == jugador for i in range(4)):
                return True

    # Diagonal /
    for f in range(FILAS - 3):
        for c in range(COLUMNAS - 3):
            if all(tablero[f+3-i, c+i] == jugador for i in range(4)):
                return True

    return False


def jugar_partida(policy_agent_empieza=True):
    tablero = np.zeros((FILAS, COLUMNAS), dtype=int)

    agent = CetinaSalasSabogal()
    randomp = RandomPolicy()

    # Montar policies
    try:
        agent.mount(None)
    except:
        agent.mount()

    try:
        randomp.mount(None)
    except:
        randomp.mount()

    jugador_tablero = -1  # en el tablero, -1 siempre empieza

    # Mapeamos quién es quién
    if policy_agent_empieza:
        jugador_agent = -1
        jugador_random = 1
    else:
        jugador_agent = 1
        jugador_random = -1

    while True:
        # ¿Alguien ganó?
        if hay_ganador(tablero, jugador_agent):
            return 1
        if hay_ganador(tablero, jugador_random):
            return -1

        # ¿Empate?
        legales = movimientos_legales(tablero)
        if not legales:
            return 0

        # Turno
        if jugador_tablero == jugador_agent:
            accion = agent.act(tablero)
        else:
            accion = randomp.act(tablero)

        tablero = colocar(tablero, accion, jugador_tablero)
        jugador_tablero = -jugador_tablero



def test_vs_random(partidas=100):
    victorias = 0
    empates = 0
    derrotas = 0

    for i in range(partidas):
        agent_empieza = (i % 2 == 0)
        resultado = jugar_partida(agent_empieza)

        if resultado > 0:
            victorias += 1
        elif resultado < 0:
            derrotas += 1
        else:
            empates += 1

    print("\n===== RESULTADOS =====")
    print(f"Partidas: {partidas}")
    print(f"Ganadas:  {victorias}")
    print(f"Empates:  {empates}")
    print(f"Perdidas: {derrotas}")
    winrate = victorias / partidas
    print(f"Winrate:  {winrate:.3f}")


if __name__ == "__main__":
    test_vs_random(100)
