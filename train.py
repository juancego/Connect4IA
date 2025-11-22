import json
import numpy as np
from connect4.connect_state import ConnectState
from policy import MctsUcbPolicy

# cantidad de partidas de auto-entrenamiento
N_EPISODES = 200

value_table = {}

def state_key(board, player):
    return f"{player}|" + "".join(str(int(x)) for x in board.flatten())

def play_self_game():
    p1 = MctsUcbPolicy()
    p2 = MctsUcbPolicy()
    p1.mount()
    p2.mount()

    state = ConnectState()  # tablero vacío
    done = False

    history = []

    while not done:
        key = state_key(state.board, state.player)
        history.append(key)

        if state.player == -1:
            a = p1.act(state.board.copy())
        else:
            a = p2.act(state.board.copy())

        state = state.transition(a)
        done = state.is_final()

    winner = state.get_winner()  # +1, -1, o 0

    # asignar reward final a cada estado visitado
    for k in history:
        if k not in value_table:
            value_table[k] = 0.0
        if winner == 0:
            continue
        elif winner == -1:
            value_table[k] += -1
        elif winner == 1:
            value_table[k] += 1


for i in range(N_EPISODES):
    play_self_game()
    print(f"Episode {i+1}/{N_EPISODES}")

# guardar conocimiento aprendido
with open("values.json", "w") as f:
    json.dump(value_table, f)

print("Entrenamiento finalizado. values.json generado.")