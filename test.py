import numpy as np

from connect4.connect_state import ConnectState
from connect4.policy import Policy

# Importa TU propia policy desde policy.py
from policy import MctsUcbPolicy as MyPolicy


class RandomPolicy(Policy):
    """
    Policy aleatoria: elige una columna legal al azar.
    Sirve como baseline para probar qué tan bueno es nuestro agente.
    """

    def mount(self) -> None:
        self.rng = np.random.default_rng()

    def act(self, s):
        state = ConnectState(board=s.copy())
        legal_actions = state.get_free_cols()
        idx = int(self.rng.integers(len(legal_actions)))
        return legal_actions[idx]


def play_one_game_as_red() -> int:
    """
    Juega UNA partida donde:
      - Mi policy = jugador rojo (-1), mueve primero.
      - RandomPolicy = jugador amarillo (1).

    Devuelve:
      -1 si gana mi policy,
       1 si gana random,
       0 si es empate.
    """
    my_agent = MyPolicy()
    rnd_agent = RandomPolicy()

    my_agent.mount()
    rnd_agent.mount()

    state = ConnectState()  # por defecto empieza player = -1

    while not state.is_final():
        if state.player == -1:
            action = my_agent.act(state.board)
        else:
            action = rnd_agent.act(state.board)
        state = state.transition(int(action))

    return state.get_winner()


def play_one_game_as_yellow() -> int:
    """
    Juega UNA partida donde:
      - RandomPolicy = jugador rojo (-1), mueve primero.
      - Mi policy = jugador amarillo (1), mueve segunda.

    Devuelve:
      -1 si gana el jugador rojo  (random),
       1 si gana el jugador amarillo (mi policy),
       0 si es empate.
    """
    my_agent = MyPolicy()
    rnd_agent = RandomPolicy()

    my_agent.mount()
    rnd_agent.mount()

    state = ConnectState()  # empieza -1 (aquí será random)

    while not state.is_final():
        if state.player == -1:
            action = rnd_agent.act(state.board)
        else:
            action = my_agent.act(state.board)
        state = state.transition(int(action))

    return state.get_winner()


def evaluate_as_red(n_games: int = 100):
    """
    Evalúa mi policy jugando n_games partidas como ROJO (empezando).
    """
    my_wins = 0
    rnd_wins = 0
    draws = 0

    for i in range(n_games):
        result = play_one_game_as_red()
        if result == -1:
            my_wins += 1
        elif result == 1:
            rnd_wins += 1
        else:
            draws += 1

        print(f"[ROJO] Partida {i + 1}/{n_games} → ganador = {result}")

    win_rate = my_wins / n_games
    print("\n===== RESULTADOS COMO ROJO (empezando) =====")
    print(f"Ganadas por mi policy: {my_wins}")
    print(f"Ganadas por random:    {rnd_wins}")
    print(f"Empates:              {draws}")
    print(f"Win rate = {win_rate * 100:.2f}%")
    print("Condición requerida: ≥ 95% de victorias contra random.\n")


def evaluate_as_yellow(n_games: int = 100):
    """
    Evalúa mi policy jugando n_games partidas como AMARILLO (segundo jugador).
    """
    my_wins = 0
    rnd_wins = 0
    draws = 0

    for i in range(n_games):
        result = play_one_game_as_yellow()
        if result == 1:
            my_wins += 1
        elif result == -1:
            rnd_wins += 1
        else:
            draws += 1

        print(f"[AMARILLO] Partida {i + 1}/{n_games} → ganador = {result}")

    win_rate = my_wins / n_games
    print("\n===== RESULTADOS COMO AMARILLO (segundo) =====")
    print(f"Ganadas por mi policy: {my_wins}")
    print(f"Ganadas por random:    {rnd_wins}")
    print(f"Empates:              {draws}")
    print(f"Win rate = {win_rate * 100:.2f}%")
    print("Condición requerida: ≥ 95% de victorias contra random.\n")


if __name__ == "__main__":
    # Puedes cambiar n_games si quieres más estabilidad estadística
    N_GAMES = 100

    print(">> Evaluando mi policy contra RandomPolicy...\n")

    evaluate_as_red(n_games=N_GAMES)
    evaluate_as_yellow(n_games=N_GAMES)