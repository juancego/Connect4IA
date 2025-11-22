import numpy as np

from connect4.policy import Policy
from connect4.connect_state import ConnectState


class DummyPolicy(Policy):
    """
    Agente DUMMIE para Conecta 4.

    Estrategia:
      - Siempre intenta jugar en la columna más a la izquierda posible.
      - Si la columna 0 está llena, intenta la 1, luego la 2, etc.
    Es MUY malo, sirve como baseline tonto.
    """

    def mount(self) -> None:
        # No necesita estado interno
        pass

    def act(self, s: np.ndarray) -> int:
        """
        s: tablero 6x7 con:
            0  = vacío
            -1 = fichas de un jugador
             1 = fichas del otro

        Siempre elige la columna legal más a la izquierda.
        """

        # Determinar de quién es el turno (igual que en tus otros agentes)
        num_tokens = np.count_nonzero(s)
        current_player = -1 if num_tokens % 2 == 0 else 1

        # Creamos el estado para usar las funciones de ConnectState
        state = ConnectState(board=s, player=current_player)

        legal_cols = state.get_free_cols()
        if not legal_cols:
            # Tablero lleno por seguridad
            return 0

        # Devuelve SIEMPRE la columna legal más pequeña (más a la izquierda)
        return int(min(legal_cols))