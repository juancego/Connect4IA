import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState

class RandomPolicy(Policy):
    """
    Agente completamente aleatorio.
    Ideal para usar como oponente de entrenamiento.

    - Elige siempre una columna legal al azar.
    - No necesita estado interno.
    """

    def mount(self) -> None:
        self.rng = np.random.default_rng()

    def act(self, s):
        state = ConnectState(board=s.copy())
        legal_actions = state.get_free_cols()
        idx = int(self.rng.integers(len(legal_actions)))
        return legal_actions[idx]