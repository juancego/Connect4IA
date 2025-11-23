import numpy as np
import random


class HeuristicOpponentEasy:
    """
    Oponente heurístico intermedio:
      1) Si puede ganar en una jugada, gana.
      2) Si el rival gana en una jugada, bloquea.
      3) Prefiere columnas centrales.
      4) Si no hay nada especial, elige al azar dentro de las legales.

    No mira más de 1 jugada hacia adelante, así que es más fuerte que Random,
    pero más débil que un heurístico más sofisticado.
    """

    FILAS = 6
    COLUMNAS = 7

    def mount(self, time_out=None):
        # No necesita inicialización especial
        pass

    # ----------------- helpers internos -----------------

    def _movimientos_legales(self, tablero: np.ndarray) -> list[int]:
        """Columnas cuya casilla superior está vacía."""
        return [c for c in range(self.COLUMNAS) if tablero[0, c] == 0]

    def _colocar(self, tablero: np.ndarray, columna: int, jugador: int) -> np.ndarray | None:
        """Simula dejar caer una ficha en la columna dada."""
        for fila in range(self.FILAS - 1, -1, -1):
            if tablero[fila, columna] == 0:
                nuevo = tablero.copy()
                nuevo[fila, columna] = jugador
                return nuevo
        return None

    def _hay_ganador(self, tablero: np.ndarray, jugador: int) -> bool:
        """Comprueba si 'jugador' tiene un conecta-4 en el tablero."""
        # Horizontal
        for f in range(self.FILAS):
            for c in range(self.COLUMNAS - 3):
                if np.all(tablero[f, c:c + 4] == jugador):
                    return True

        # Vertical
        for f in range(self.FILAS - 3):
            for c in range(self.COLUMNAS):
                if np.all(tablero[f:f + 4, c] == jugador):
                    return True

        # Diagonal principal
        for f in range(self.FILAS - 3):
            for c in range(self.COLUMNAS - 3):
                if all(tablero[f + k, c + k] == jugador for k in range(4)):
                    return True

        # Diagonal secundaria
        for f in range(self.FILAS - 3):
            for c in range(self.COLUMNAS - 3):
                if all(tablero[f + 3 - k, c + k] == jugador for k in range(4)):
                    return True

        return False

    def _turno(self, tablero: np.ndarray) -> int:
        """
        Determina a quién le toca mover:
        - Si hay un número par de fichas en total, mueve -1.
        - Si es impar, mueve 1.
        """
        count = int(np.count_nonzero(tablero))
        return -1 if count % 2 == 0 else 1

    # ----------------- acción principal -----------------

    def act(self, s: np.ndarray) -> int:
        """
        Decide en qué columna jugar desde el estado 's'.
        """
        tablero = np.array(s, dtype=int, copy=True)
        jugador = self._turno(tablero)
        rival = -jugador

        legales = self._movimientos_legales(tablero)
        if not legales:
            # Si no hay movimientos, devolvemos una columna dummy (no debería pasar)
            return 0

        # 1) Si puedo ganar en una jugada, lo hago.
        for c in legales:
            siguiente = self._colocar(tablero, c, jugador)
            if siguiente is not None and self._hay_ganador(siguiente, jugador):
                return c

        # 2) Si el rival puede ganar en una jugada, bloqueo.
        for c in legales:
            siguiente = self._colocar(tablero, c, rival)
            if siguiente is not None and self._hay_ganador(siguiente, rival):
                return c

        # 3) Prefiero columnas centrales (como tu heurística simple).
        prioridad = [3, 2, 4, 1, 5, 0, 6]
        candidatos = [c for c in prioridad if c in legales]
        if candidatos:
            # Para que no sea determinista 100%, elige aleatorio entre las mejores
            # 2 columnas disponibles según la prioridad, si hay más de una.
            top = candidatos[:2] if len(candidatos) > 1 else candidatos
            return random.choice(top)

        # 4) Si por alguna razón no se escogió nada, elige al azar entre las legales.
        return random.choice(legales)
