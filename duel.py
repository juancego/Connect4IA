from connect4.policy import Policy
from connect4.utils import find_importable_classes
from connect4.connect_state import ConnectState

import numpy as np


def play_single_game(policy_cls_1: type[Policy], policy_cls_2: type[Policy]) -> int:
    """
    Juega UNA partida entre:
      - policy 1 como jugador -1 (rojo)
      - policy 2 como jugador  1 (amarillo)

    Devuelve:
      -1 si gana el jugador 1
       1 si gana el jugador 2
       0 si es empate
    """
    p1 = policy_cls_1()
    p2 = policy_cls_2()

    p1.mount()
    p2.mount()

    state = ConnectState()

    while not state.is_final():
        current_policy = p1 if state.player == -1 else p2
        action = current_policy.act(state.board)
        state = state.transition(int(action))

    return state.get_winner()


def main():
    # Cargar todas las policies dentro de groups/ (igual que main.py)
    participants = find_importable_classes("groups", Policy)
    print("Agentes disponibles en groups/:")
    for name in participants.keys():
        print(" -", name)

    # 👉 AQUÍ eliges a los dos por nombre EXACTO
    # por ejemplo:
    name_1 = "My Agent"           # heurístico
    name_2 = "Mateo"  # MCTS entrenado

    policy_cls_1 = participants[name_1]
    policy_cls_2 = participants[name_2]

    N_GAMES = 10

    wins_1 = 0
    wins_2 = 0
    draws = 0

    # Para ser más justo, alternamos quién empieza:
    # en partidas pares empieza policy 1, en impares policy 2.
    for i in range(N_GAMES):
        if i % 2 == 0:
            # policy 1 como -1, policy 2 como 1
            result = play_single_game(policy_cls_1, policy_cls_2)
            starter = name_1
        else:
            # invertimos: policy 2 como -1, policy 1 como 1
            result = play_single_game(policy_cls_2, policy_cls_1)
            # pero interpretamos el resultado desde el punto de vista fijo:
            # result = -1 => gana quien jugó -1
            # en este caso, si gana -1 gana name_2
            # si gana 1, gana name_1
            if result == -1:
                # ganó el que empezó (policy 2)
                result = 1   # victoria para name_2
            elif result == 1:
                result = -1  # victoria para name_1
            starter = name_2

        if result == -1:
            wins_1 += 1
        elif result == 1:
            wins_2 += 1
        else:
            draws += 1

        print(f"Partida {i+1}/{N_GAMES} terminada (empezó {starter})")

    print("\nRESUMEN FINAL (100 partidas alternando quién empieza):")
    print(f"{name_1} ganó: {wins_1}")
    print(f"{name_2} ganó: {wins_2}")
    print(f"Empates: {draws}")


if __name__ == "__main__":
    main()