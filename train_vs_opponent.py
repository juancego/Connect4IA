import json
import numpy as np

from connect4.connect_state import ConnectState
from connect4.policy import Policy
from connect4.utils import find_importable_classes


# ----------------- helpers ----------------- #

def state_key(board, player: int) -> str:
    """
    Serializa (tablero, jugador) en un string para usar como clave en values.json.
    """
    flat = "".join(str(int(x)) for x in board.flatten())
    return f"{player}|{flat}"


def load_value_table(path: str = "values.json") -> dict[str, float]:
    """
    Carga la tabla de valores si existe, si no, devuelve un diccionario vacío.
    """
    try:
        with open(path, "r") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_value_table(value_table: dict[str, float], path: str = "values.json") -> None:
    """
    Guarda la tabla de valores en disco.
    """
    with open(path, "w") as f:
        json.dump(value_table, f)


# ----------------- entrenamiento vs oponentes de groups/ ----------------- #

def main():
    # Nº de episodios de entrenamiento (partidas)
    N_EPISODES = 600  # puedes subirlo si el tiempo te alcanza

    rng = np.random.default_rng()

    # 1) Cargar tabla de valores previa (si existe)
    value_table = load_value_table()

    # 2) Buscar agentes disponibles en groups/
    participants = find_importable_classes("groups", Policy)
    print("Agentes disponibles en groups/:")
    for name in participants.keys():
        print(" -", name)

    # 3) Definir quién es el agente que APRENDE
    LEARNER_NAME = "My Agent"   # <-- que está en groups/My Agent/policy.py

    if LEARNER_NAME not in participants:
        raise ValueError(f"No encontré al aprendiz '{LEARNER_NAME}' en groups/")

    LearningPolicy = participants[LEARNER_NAME]

    # 4) Definir la lista de oponentes (TODOS menos el aprendiz)
    opponent_names = [name for name in participants.keys() if name != LEARNER_NAME]

    if not opponent_names:
        raise ValueError(
            f"No hay oponentes en groups/ distintos de '{LEARNER_NAME}'. "
            "Agrega más agentes a groups/ para poder entrenar."
        )

    print("\nOponentes contra los que entrenará My Agent:")
    for name in opponent_names:
        print(" -", name)

    print(f"\nEntrenando '{LEARNER_NAME}' durante {N_EPISODES} partidas...\n")

    # 5) Bucle de episodios
    for ep in range(N_EPISODES):
        # Elegir oponente aleatorio de la lista
        opp_name = rng.choice(opponent_names)
        OpponentPolicy = participants[opp_name]

        # Nuestro agente que aprende (My Agent) como jugador -1
        learner: Policy = LearningPolicy()
        learner.mount()

        # Oponente como jugador 1
        opponent: Policy = OpponentPolicy()
        opponent.mount()

        # Estado inicial
        state = ConnectState()
        episode_states: list[str] = []

        # Jugar una partida completa
        while not state.is_final():
            if state.player == -1:
                # Turno del aprendiz
                action = learner.act(state.board)
            else:
                # Turno del oponente
                action = opponent.act(state.board)

            # Guardamos el estado actual antes de la transición
            k = state_key(state.board, state.player)
            episode_states.append(k)

            state = state.transition(int(action))

        winner = state.get_winner()  # -1 (learner), 1 (oponente), 0 (empate)

        # 6) Actualizar la tabla de valores desde la perspectiva del aprendiz
        if winner != 0:
            reward = 1.0 if winner == -1 else -1.0
            for k in episode_states:
                if k not in value_table:
                    value_table[k] = 0.0
                value_table[k] += reward

        print(f"Episodio {ep+1}/{N_EPISODES} terminado. "
              f"Oponente: {opp_name}. Ganador: {winner}")

    # 7) Guardar la tabla actualizada
    save_value_table(value_table)
    print("\nEntrenamiento terminado. values.json actualizado.")


if __name__ == "__main__":
    main()