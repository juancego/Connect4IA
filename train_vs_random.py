import json
import random
from datetime import datetime

import numpy as np

from groups.Random.random_policy import RandomPolicy


class QTrainerConnect4:
    """
    Entrenador simple de Q-learning para Conecta 4 contra RandomPolicy.
    No usa MCTS, solo Q(s,a) tabular con media incremental de recompensas finales.

    El archivo connect4_model.json que se guarda es compatible con tu policy CetinaSalasSabogal:
      { estado_str: { "accion": Q, ... }, ... }
    """

    def __init__(
        self,
        filas: int = 6,
        columnas: int = 7,
        archivo_modelo: str = "connect4_model.json",
        archivo_curva: str = "curva_vs_random.json",
    ) -> None:
        self.FILAS = filas
        self.COLUMNAS = columnas
        self.ARCHIVO_MODELO = archivo_modelo
        self.ARCHIVO_CURVA = archivo_curva

        # Q-table extendida interna: estado -> accion -> {"Q": float, "N": int}
        self.q_stats: dict[str, dict[int, dict[str, float | int]]] = self._cargar_q_stats()

        # Oponente fijo: RandomPolicy
        self.oponente = RandomPolicy()
        try:
            self.oponente.mount(None)
        except TypeError:
            try:
                self.oponente.mount()
            except Exception:
                pass

    # =============================================================
    # ----------------- UTILIDADES DEL JUEGO ----------------------
    # =============================================================

    def _movimientos_legales(self, tablero: np.ndarray) -> list[int]:
        """
        Devuelve las columnas donde la casilla superior (fila 0) está libre.
        """
        return [c for c in range(self.COLUMNAS) if tablero[0, c] == 0]

    def _colocar(self, tablero: np.ndarray, columna: int, jugador: int) -> np.ndarray | None:
        """
        Simula dejar caer una ficha en la columna dada.
        Devuelve un nuevo tablero, o None si la columna está llena.
        """
        for fila in range(self.FILAS - 1, -1, -1):
            if tablero[fila, columna] == 0:
                nuevo = tablero.copy()
                nuevo[fila, columna] = jugador
                return nuevo
        return None

    def _hay_ganador(self, tablero: np.ndarray, jugador: int) -> bool:
        """
        Comprueba si el jugador tiene un conecta 4.
        """
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

    def _turno_actual(self, tablero: np.ndarray) -> int:
        """
        Determina quién mueve según el número de fichas:
        -1 si el número de fichas es par, 1 si es impar.
        """
        count = int(np.count_nonzero(tablero))
        return -1 if count % 2 == 0 else 1

    def _codificar_estado(self, tablero: np.ndarray) -> str:
        """
        Igual esquema que tu policy:
        '<jugador_que_mueve>|<tablero_aplanado>'
        """
        jugador = self._turno_actual(tablero)
        plano = "".join(str(int(x)) for x in tablero.flatten())
        return f"{jugador}|{plano}"

    # =============================================================
    # ---------------------- CARGA / GUARDADO ---------------------
    # =============================================================

    def _cargar_q_stats(self) -> dict:
        """
        Carga connect4_model.json si existe.
        Acepta tanto formato simple:
          { estado: { "accion": Q, ... }, ... }
        como formato extendido:
          { estado: { "accion": { "Q": Q, "N": N }, ... }, ... }
        Devuelve siempre forma extendida interna con N y Q.
        """
        try:
            with open(self.ARCHIVO_MODELO, "r", encoding="utf-8") as f:
                datos = json.load(f)
        except Exception:
            return {}

        if not isinstance(datos, dict):
            return {}

        q_stats: dict[str, dict[int, dict[str, float | int]]] = {}

        for est, acciones in datos.items():
            if not isinstance(acciones, dict):
                continue
            q_stats[est] = {}
            for a_str, val in acciones.items():
                try:
                    a_int = int(a_str)
                except Exception:
                    continue

                if isinstance(val, dict) and "Q" in val:
                    try:
                        q = float(val.get("Q", 0.0))
                    except Exception:
                        q = 0.0
                    try:
                        n = int(val.get("N", 0))
                    except Exception:
                        n = 0
                else:
                    try:
                        q = float(val)
                    except Exception:
                        q = 0.0
                    n = 0

                q_stats[est][a_int] = {"Q": q, "N": n}

        return q_stats

    def _guardar_modelo_simple(self) -> None:
        """
        Guarda solo los Q-values en connect4_model.json,
        en formato compatible con la policy:
          { estado: { "accion": Q, ... }, ... }
        """
        salida: dict[str, dict[str, float]] = {}
        for estado, acciones in self.q_stats.items():
            acciones_q: dict[str, float] = {}
            for a_int, info in acciones.items():
                q_val = float(info.get("Q", 0.0))
                acciones_q[str(a_int)] = q_val
            if acciones_q:
                salida[estado] = acciones_q

        with open(self.ARCHIVO_MODELO, "w", encoding="utf-8") as f:
            json.dump(salida, f)

        print(f"\nSe guardó modelo final en '{self.ARCHIVO_MODELO}' con {len(salida)} estados.")

    # =============================================================
    # --------------------- POLÍTICA DEL AGENTE -------------------
    # =============================================================

    def _elegir_accion_epsilon_greedy(self, tablero: np.ndarray, epsilon: float) -> int:
        """
        Política ε-greedy basada en Q(s,a) sobre las columnas legales.
        Si el estado no tiene Q, se elige columna aleatoria.
        """
        legales = self._movimientos_legales(tablero)
        if not legales:
            # por robustez; en teoría no debería pasar al decidir
            return 0

        estado = self._codificar_estado(tablero)
        acciones_info = self.q_stats.get(estado, {})

        # Exploración
        if random.random() < epsilon or not acciones_info:
            return random.choice(legales)

        # Explotación: mejor Q entre las acciones legales que estén en la tabla
        mejor_accion = None
        mejor_q = -1e18
        for a in legales:
            info = acciones_info.get(a)
            if info is None:
                continue
            q_val = float(info.get("Q", 0.0))
            if q_val > mejor_q:
                mejor_q = q_val
                mejor_accion = a

        if mejor_accion is not None:
            return mejor_accion

        # Si ninguna acción legal está en la tabla, aleatoria
        return random.choice(legales)

    def _elegir_accion_greedy(self, tablero: np.ndarray) -> int:
        """
        Política completamente greedy (ε=0) para evaluación.
        Si no hay Q para el estado, elige columna aleatoria.
        """
        legales = self._movimientos_legales(tablero)
        if not legales:
            return 0

        estado = self._codificar_estado(tablero)
        acciones_info = self.q_stats.get(estado, {})

        mejor_accion = None
        mejor_q = -1e18
        for a in legales:
            info = acciones_info.get(a)
            if info is None:
                continue
            q_val = float(info.get("Q", 0.0))
            if q_val > mejor_q:
                mejor_q = q_val
                mejor_accion = a

        if mejor_accion is not None:
            return mejor_accion

        return random.choice(legales)

    # =============================================================
    # ---------------------- UN EPISODIO RL -----------------------
    # =============================================================

    def _jugar_episodio(self, epsilon: float) -> int:
        """
        Juega una partida completa agente vs RandomPolicy usando ε-greedy.
        Actualiza la Q-table al final con la recompensa global.
        Devuelve la recompensa final desde la perspectiva del agente (-1):
          +1 si gana el agente,
          -1 si gana RandomPolicy,
           0 si hay empate.
        """
        tablero = np.zeros((self.FILAS, self.COLUMNAS), dtype=int)
        agente = -1
        rival = 1

        # Lista de (estado, accion) que tomó el agente durante el episodio
        trayectoria: list[tuple[str, int]] = []

        while True:
            # Comprobación de estado terminal
            if self._hay_ganador(tablero, agente):
                recompensa = 1
                break
            if self._hay_ganador(tablero, rival):
                recompensa = -1
                break
            legales = self._movimientos_legales(tablero)
            if not legales:
                recompensa = 0
                break

            turno = self._turno_actual(tablero)
            if turno == agente:
                # Turno del agente: ε-greedy
                accion = self._elegir_accion_epsilon_greedy(tablero, epsilon)
                estado = self._codificar_estado(tablero)
                trayectoria.append((estado, accion))
                tablero = self._colocar(tablero, accion, agente)
            else:
                # Turno del random
                accion_rival = self.oponente.act(tablero)
                if accion_rival not in legales:
                    # Por si la policy random hace algo raro, caemos a una legal
                    accion_rival = random.choice(legales)
                tablero = self._colocar(tablero, accion_rival, rival)

        # Actualización de Q(s,a) para cada jugada del agente en este episodio
        for estado, accion in trayectoria:
            if estado not in self.q_stats:
                self.q_stats[estado] = {}
            if accion not in self.q_stats[estado]:
                self.q_stats[estado][accion] = {"Q": 0.0, "N": 0}

            info = self.q_stats[estado][accion]
            n = int(info.get("N", 0)) + 1
            q_old = float(info.get("Q", 0.0))
            q_new = q_old + (recompensa - q_old) / n

            info["N"] = n
            info["Q"] = q_new

        return recompensa

    # =============================================================
    # --------------------- EVALUACIÓN VS RANDOM ------------------
    # =============================================================

    def evaluar_vs_random(self, partidas: int = 200) -> dict:
        """
        Evalúa el Q-table actual contra RandomPolicy con política GREEDY (sin exploración).
        """
        victorias = 0
        empates = 0
        derrotas = 0

        for _ in range(partidas):
            tablero = np.zeros((self.FILAS, self.COLUMNAS), dtype=int)
            agente = -1
            rival = 1

            while True:
                if self._hay_ganador(tablero, agente):
                    victorias += 1
                    break
                if self._hay_ganador(tablero, rival):
                    derrotas += 1
                    break
                legales = self._movimientos_legales(tablero)
                if not legales:
                    empates += 1
                    break

                turno = self._turno_actual(tablero)
                if turno == agente:
                    accion = self._elegir_accion_greedy(tablero)
                    tablero = self._colocar(tablero, accion, agente)
                else:
                    accion_rival = self.oponente.act(tablero)
                    if accion_rival not in legales:
                        accion_rival = random.choice(legales)
                    tablero = self._colocar(tablero, accion_rival, rival)

        total = victorias + empates + derrotas
        winrate = victorias / total if total > 0 else 0.0

        return {
            "winrate": winrate,
            "victorias": victorias,
            "empates": empates,
            "derrotas": derrotas,
            "partidas": total,
        }

    # =============================================================
    # --------------------- ENTRENAMIENTO GLOBAL ------------------
    # =============================================================

    def entrenar_contra_random(
        self,
        episodios_totales: int = 2000,
        paso_eval: int = 200,
        partidas_eval: int = 200,
        epsilon_inicial: float = 0.2,
        epsilon_final: float = 0.01,
    ) -> None:
        """
        Entrena el agente contra RandomPolicy con Q-learning tabular.
        Cada 'paso_eval' episodios se evalúa (sin exploración) y se guarda
        la métrica para dibujar la curva de aprendizaje.

        Guarda:
          - connect4_model.json : modelo final (solo Q).
          - curva_vs_random.json: lista de {episodios, winrate, ...}.
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        historial: list[dict] = []
        recompensas_ep: list[float] = []

        print(
            f"\n[RUN {run_id}] Entrenando {episodios_totales} episodios "
            f"contra RandomPolicy (Q-learning tabular)...\n"
        )

        for epi in range(1, episodios_totales + 1):
            # Decaimiento lineal de epsilon
            fraccion = epi / episodios_totales
            epsilon = max(epsilon_final, epsilon_inicial * (1.0 - fraccion))

            r = self._jugar_episodio(epsilon=epsilon)
            recompensas_ep.append(float(r))

            if epi % 50 == 0 or epi == 1:
                media_ultimos = sum(recompensas_ep[-50:]) / min(50, len(recompensas_ep))
                print(
                    f"(RUN {run_id}) Episodio {epi}/{episodios_totales} "
                    f"- recompensa={r}, epsilon={epsilon:.3f}, "
                    f"media_ultimos_50={media_ultimos:.3f}"
                )

            if epi % paso_eval == 0:
                resultado_eval = self.evaluar_vs_random(partidas=partidas_eval)
                punto = {
                    "episodios": epi,
                    "winrate": resultado_eval["winrate"],
                    "victorias": resultado_eval["victorias"],
                    "empates": resultado_eval["empates"],
                    "derrotas": resultado_eval["derrotas"],
                    "partidas_eval": resultado_eval["partidas"],
                }
                historial.append(punto)
                print(
                    f"[RUN {run_id}] [Evaluación ep {epi}] "
                    f"winrate={resultado_eval['winrate']:.3f} "
                    f"(V={resultado_eval['victorias']}, "
                    f"E={resultado_eval['empates']}, "
                    f"D={resultado_eval['derrotas']})"
                )

        # Guardamos modelo final (solo Q)
        self._guardar_modelo_simple()

        # Guardamos la curva
        salida_curva = {
            "metadata": {
                "run_id": run_id,
                "episodios_totales": episodios_totales,
                "paso_eval": paso_eval,
                "partidas_eval": partidas_eval,
                "epsilon_inicial": epsilon_inicial,
                "epsilon_final": epsilon_final,
            },
            "historial": historial,
        }
        with open(self.ARCHIVO_CURVA, "w", encoding="utf-8") as f:
            json.dump(salida_curva, f, indent=2)

        print(f"\n[RUN {run_id}] Curva de aprendizaje guardada en '{self.ARCHIVO_CURVA}'.")
        print(f"[RUN {run_id}] Entrenamiento contra RandomPolicy terminado.")


# =============================================================
# ---------------------- EJECUCIÓN DIRECTA ---------------------
# =============================================================

if __name__ == "__main__":
    entrenador = QTrainerConnect4(
        archivo_modelo="connect4_model.json",
        archivo_curva="curva_vs_random.json",
    )
    entrenador.entrenar_contra_random(
        episodios_totales=200,   # puedes subir esto si quieres más entrenamiento
        paso_eval=1,           # cada 200 episodios, una evaluación completa
        partidas_eval=1,       # número de partidas por evaluación
        epsilon_inicial=0.2,
        epsilon_final=0.01,
    )