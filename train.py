import copy
import json
import math
import random
from datetime import datetime

import numpy as np


class EntrenadorConnect4:
    """
    Clase que encapsula el entrenamiento de un agente de Conecta 4 mediante
    autojuego con MCTS-UCB. El agente que aprende se representa solo por su
    Q-table y funciones auxiliares.

    NO se usan policies externas para aprender. Las evaluaciones se hacen
    únicamente contra snapshots anteriores del propio agente.
    """

    def __init__(
        self,
        filas: int = 6,
        columnas: int = 7,
        archivo_modelo: str = "connect4_model.json",
    ) -> None:
        self.FILAS = filas
        self.COLUMNAS = columnas
        self.ARCHIVO = archivo_modelo

    # =============================================================
    # ----------------- UTILIDADES DEL JUEGO ----------------------
    # =============================================================

    def movimientos_legales(self, tablero: np.ndarray) -> list[int]:
        """
        Se revisan las columnas cuya casilla superior está libre.
        """
        return [c for c in range(self.COLUMNAS) if tablero[0, c] == 0]

    def colocar(self, tablero: np.ndarray, columna: int, jugador: int) -> np.ndarray | None:
        """
        Se simula dejar caer una ficha de un jugador en la columna indicada.
        Se devuelve una copia modificada del tablero, o None si la columna
        está llena.
        """
        for fila in range(self.FILAS - 1, -1, -1):
            if tablero[fila, columna] == 0:
                nuevo = tablero.copy()
                nuevo[fila, columna] = jugador
                return nuevo
        return None

    def hay_ganador(self, tablero: np.ndarray, jugador: int) -> bool:
        """
        Se comprueba si el jugador logró conectar cuatro fichas.
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

    def turno(self, tablero: np.ndarray) -> int:
        """
        Se determina el jugador al que le toca mover según el número de fichas
        ya colocadas: si es par mueve -1, si es impar mueve 1.
        """
        count = int(np.count_nonzero(tablero))
        return -1 if count % 2 == 0 else 1

    def codificar_estado(self, tablero: np.ndarray) -> str:
        """
        Se genera una representación textual del estado:
        '<jugador_que_mueve>|<tablero_aplanado>'.
        """
        jugador = self.turno(tablero)
        plano = "".join(str(int(x)) for x in tablero.flatten())
        return f"{jugador}|{plano}"

    # =============================================================
    # ----------------------------- NODO MCTS ----------------------
    # =============================================================

    class Nodo:
        """
        Nodo del árbol de búsqueda MCTS. Almacena el tablero, el jugador al
        que le toca mover, el padre, la acción que llevó aquí, los hijos ya
        explorados y estadísticas de visitas y recompensa acumulada.
        """

        def __init__(
            self,
            tablero: np.ndarray,
            jugador: int,
            entrenador: "EntrenadorConnect4",
            padre: "EntrenadorConnect4.Nodo | None" = None,
            accion: int | None = None,
        ) -> None:
            self.tablero = tablero
            self.jugador = jugador
            self.padre = padre
            self.accion = accion
            self.ent = entrenador

            self.hijos: dict[int, "EntrenadorConnect4.Nodo"] = {}
            self.no_exploradas: list[int] = entrenador.movimientos_legales(tablero)

            # N almacena las visitas al nodo, W la suma de recompensas desde
            # la perspectiva del jugador raíz de la búsqueda.
            self.N: int = 0
            self.W: float = 0.0

        def terminal(self) -> bool:
            """
            Estado terminal si algún jugador gana o no hay más movimientos legales.
            """
            ent = self.ent
            return (
                ent.hay_ganador(self.tablero, 1)
                or ent.hay_ganador(self.tablero, -1)
                or len(ent.movimientos_legales(self.tablero)) == 0
            )

        def expandir(self) -> "EntrenadorConnect4.Nodo":
            """
            Se toma una acción aún no explorada desde este nodo y se crea un
            hijo correspondiente al estado resultante de aplicar esa acción.
            """
            accion = self.no_exploradas.pop()
            nuevo_tablero = self.ent.colocar(self.tablero, accion, self.jugador)
            hijo = EntrenadorConnect4.Nodo(
                nuevo_tablero,
                -self.jugador,
                self.ent,
                padre=self,
                accion=accion,
            )
            self.hijos[accion] = hijo
            return hijo

        def seleccionar_hijo_ucb(self, c: float = math.sqrt(2.0)) -> "EntrenadorConnect4.Nodo":
            """
            Se selecciona uno de los hijos utilizando la fórmula UCB1.
            """
            mejor = None
            mejor_puntaje = -1e18

            for hijo in self.hijos.values():
                if hijo.N == 0:
                    puntaje = float("inf")
                else:
                    media = hijo.W / hijo.N
                    puntaje = media + c * math.sqrt(math.log(self.N) / hijo.N)

                if puntaje > mejor_puntaje:
                    mejor_puntaje = puntaje
                    mejor = hijo

            return mejor

    # =============================================================
    # --------------------------- ROLLOUT RANDOM ------------------
    # =============================================================

    def rollout(self, tablero: np.ndarray, jugador_raiz: int) -> int:
        """
        Se realiza una simulación completamente aleatoria desde el estado dado
        hasta llegar al final de la partida. Se devuelve la recompensa desde
        la perspectiva del jugador raíz (jugador_raiz): +1 si gana, -1 si
        pierde, 0 si hay empate.
        """
        b = tablero.copy()
        turno_actual = self.turno(b)

        while True:
            # Se revisa si el jugador que movió en el turno anterior ganó.
            if self.hay_ganador(b, -turno_actual):
                ganador = -turno_actual
                if ganador == jugador_raiz:
                    return 1
                else:
                    return -1

            movimientos = self.movimientos_legales(b)
            if not movimientos:
                return 0

            accion = random.choice(movimientos)
            b = self.colocar(b, accion, turno_actual)
            turno_actual = -turno_actual

    # =============================================================
    # ------------------------ UNA PARTIDA MCTS -------------------
    # =============================================================

    def jugar_una_partida(self, q_stats: dict, simulaciones: int = 150) -> tuple[int, int]:
        """
        Se juega una partida completa de autojuego usando MCTS-UCB para ambos
        jugadores. Al final se actualizan los valores Q de todos los estados
        visitados en q_stats y se devuelve:

            - la recompensa final desde el punto de vista del jugador inicial:
                +1 si el jugador inicial gana,
                -1 si el jugador inicial pierde,
                 0 si hay empate,
            - la longitud de la partida en número de movimientos.
        """
        tablero = np.zeros((self.FILAS, self.COLUMNAS), dtype=int)
        jugador = self.turno(tablero)
        jugador_inicial = jugador

        # Se almacena (estado, acción, jugador_que_movio) por cada jugada
        episodio: list[tuple[str, int, int]] = []
        movimientos = 0

        while True:
            # Se comprueba si el último jugador que movió (-jugador) ha ganado.
            if self.hay_ganador(tablero, -jugador):
                ultimo = -jugador
                recompensa = 1 if ultimo == jugador_inicial else -1
                break

            movimientos_legales = self.movimientos_legales(tablero)
            if not movimientos_legales:
                recompensa = 0
                break

            raiz = EntrenadorConnect4.Nodo(tablero, jugador, self)

            # MCTS desde la raíz, recompensa siempre desde perspectiva de jugador raíz (jugador)
            for _ in range(simulaciones):
                nodo = raiz

                # 1) Selección
                while not nodo.no_exploradas and nodo.hijos:
                    nodo = nodo.seleccionar_hijo_ucb()

                # 2) Expansión
                if nodo.no_exploradas:
                    nodo = nodo.expandir()

                # 3) Simulación desde este nodo, con recompensa respecto al jugador raíz
                r = self.rollout(nodo.tablero, jugador_raiz=jugador)

                # 4) Retropropagación: se acumula la misma recompensa para todos los nodos ascendentes
                while nodo is not None:
                    nodo.N += 1
                    nodo.W += r
                    nodo = nodo.padre

            # Se elige la acción real como el hijo más visitado
            mejor_accion = max(raiz.hijos.items(), key=lambda t: t[1].N)[0]

            estado = self.codificar_estado(tablero)
            episodio.append((estado, mejor_accion, jugador))

            tablero = self.colocar(tablero, mejor_accion, jugador)
            jugador = -jugador
            movimientos += 1

        # Actualización de Q en q_stats
        for estado, accion, jugador_que_movio in episodio:
            if estado not in q_stats:
                q_stats[estado] = {}
            if accion not in q_stats[estado]:
                q_stats[estado][accion] = {"N": 0, "Q": 0.0}

            # Si el que movió es el jugador inicial, ve la recompensa tal cual.
            # Si fue el rival, ve la recompensa con signo invertido.
            r_efectiva = recompensa if jugador_que_movio == jugador_inicial else -recompensa

            q_stats[estado][accion]["N"] += 1
            N = q_stats[estado][accion]["N"]
            oldQ = q_stats[estado][accion]["Q"]
            q_stats[estado][accion]["Q"] = oldQ + (r_efectiva - oldQ) / N

        return recompensa, movimientos

    # =============================================================
    # -------------- ELECCIÓN DE ACCIÓN A PARTIR DE Q -------------
    # =============================================================

    def elegir_accion_q(self, tablero: np.ndarray, jugador: int, q_stats: dict) -> int | None:
        """
        Dada la Q-table y un tablero, se elige la acción que el agente
        entrenado jugaría usando únicamente Q(s,a).
        """
        legales = self.movimientos_legales(tablero)
        if not legales:
            return None

        estado = self.codificar_estado(tablero)
        acciones_info = q_stats.get(estado, {})

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

        # Heurística de respaldo si no hay Q para este estado.
        preferencia = [3, 2, 4, 1, 5, 0, 6]
        for a in preferencia:
            if 0 <= a < self.COLUMNAS and a in legales:
                return a

        return legales[0]

    # =============================================================
    # ------- EVALUACIÓN ENTRE VERSIONES (SNAPSHOTS) --------------
    # =============================================================

    def jugar_partida_entre_versiones(
        self,
        q_actual: dict,
        q_oponente: dict,
        actual_empieza: bool = True,
    ) -> int:
        """
        Se juega una partida entre dos versiones del mismo agente:
          - el agente "actual", que usa q_actual,
          - el agente "oponente", que usa q_oponente.

        Se decide quién empieza y se usa elegir_accion_q con la tabla
        correspondiente. Se devuelve +1 si gana la versión actual,
        -1 si gana la versión oponente y 0 si hay empate.
        """
        tablero = np.zeros((self.FILAS, self.COLUMNAS), dtype=int)

        jugador_tablero = -1  # -1 siempre empieza en el tablero

        if actual_empieza:
            jugador_actual = -1
            jugador_oponente = 1
        else:
            jugador_actual = 1
            jugador_oponente = -1

        while True:
            if self.hay_ganador(tablero, jugador_actual):
                return 1
            if self.hay_ganador(tablero, jugador_oponente):
                return -1

            movimientos = self.movimientos_legales(tablero)
            if not movimientos:
                return 0

            if jugador_tablero == jugador_actual:
                accion = self.elegir_accion_q(tablero, jugador_tablero, q_actual)
            else:
                accion = self.elegir_accion_q(tablero, jugador_tablero, q_oponente)

            if accion is None:
                return 0

            tablero = self.colocar(tablero, accion, jugador_tablero)
            jugador_tablero = -jugador_tablero

    def evaluar_entre_versiones(
        self,
        q_actual: dict,
        q_oponente: dict,
        partidas: int = 50,
    ) -> dict:
        """
        Se evalúa la versión actual del agente contra una versión anterior
        (snapshot) usando únicamente Q(s,a) para ambos.

        Se alterna quién empieza para que la comparación sea justa.
        """
        victorias = 0
        empates = 0
        derrotas = 0

        for i in range(partidas):
            actual_empieza = (i % 2 == 0)
            resultado = self.jugar_partida_entre_versiones(
                q_actual, q_oponente, actual_empieza
            )
            if resultado > 0:
                victorias += 1
            elif resultado < 0:
                derrotas += 1
            else:
                empates += 1

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
    # ---------------------- CARGA Y GUARDADO ---------------------
    # =============================================================

    def cargar(self) -> dict:
        """
        Se intenta cargar el archivo de modelo si existe. Se aceptan tanto
        modelos en formato antiguo (estado → acción → {N, Q}) como modelos
        ya guardados en formato simple (estado → acción → Q). Internamente
        siempre se trabaja con la forma extendida que incluye N y Q.
        """
        try:
            with open(self.ARCHIVO, "r") as f:
                datos = json.load(f)
        except Exception:
            return {}

        q_stats: dict = {}
        if not isinstance(datos, dict):
            return {}

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
                        n = int(val.get("N", 1))
                    except Exception:
                        n = 1
                else:
                    try:
                        q = float(val)
                    except Exception:
                        q = 0.0
                    n = 1

                q_stats[est][a_int] = {"N": n, "Q": q}

        return q_stats

    def guardar_solo_Q(self, q_stats: dict, archivo: str | None = None) -> None:
        """
        Se genera una tabla simplificada a partir de q_stats, conservando
        únicamente el valor Q medio por acción. Este es el formato que la
        policy CetinaSalasSabogal utiliza en tiempo de juego.

        Si 'archivo' es None, se usa self.ARCHIVO. Si no, se guarda también
        en el nombre proporcionado (por ejemplo, una copia histórica).
        """
        tabla_simple: dict[str, dict[str, float]] = {}
        for estado, acciones in q_stats.items():
            acciones_q: dict[str, float] = {}
            for a_int, info in acciones.items():
                q_val = float(info.get("Q", 0.0))
                acciones_q[str(a_int)] = q_val
            if acciones_q:
                tabla_simple[estado] = acciones_q

        destino = archivo if archivo is not None else self.ARCHIVO
        with open(destino, "w") as f:
            json.dump(tabla_simple, f)

        print(f"\nSe guardó {destino} con {len(tabla_simple)} estados aprendidos.")

    def guardar_qstats_completo(self, q_stats: dict, archivo: str) -> None:
        """
        Se guarda q_stats completo incluyendo tanto N como Q por acción.
        Este archivo no lo usa la policy en producción, pero sirve para
        análisis e inspección más profunda del proceso de aprendizaje.
        """
        salida: dict[str, dict[str, dict[str, float]]] = {}
        for estado, acciones in q_stats.items():
            acciones_completas: dict[str, dict[str, float]] = {}
            for a_int, info in acciones.items():
                n_val = int(info.get("N", 0))
                q_val = float(info.get("Q", 0.0))
                acciones_completas[str(a_int)] = {"N": n_val, "Q": q_val}
            if acciones_completas:
                salida[estado] = acciones_completas

        with open(archivo, "w") as f:
            json.dump(salida, f, indent=2)

        print(f"\nSe guardó q_stats completo con N y Q en '{archivo}'.")

    # =============================================================
    # -------------------- ENTRENAMIENTO PRINCIPAL ----------------
    # =============================================================

    def entrenar_con_snapshots(
        self,
        episodios_totales: int = 200,
        paso_eval: int = 20,
        partidas_eval: int = 50,
        base_metricas: str = "curvas_aprendizaje_snapshots",
        base_selfplay: str = "selfplay_stats",
        run_id: str | None = None,
    ) -> None:
        """
        Entrena con self-play MCTS y evalúa periódicamente SOLO contra
        snapshots anteriores del propio agente.

        Se guardan:
          - curvas de winrate vs snapshot anterior,
          - recompensas y longitudes de self-play,
          - modelo final (solo Q) para Policy.py,
          - modelo final histórico (solo Q),
          - q_stats completo con N y Q,
          - snapshots intermedios (solo Q) por episodio de evaluación.
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        archivo_metricas = f"{base_metricas}_{run_id}.json"
        archivo_selfplay = f"{base_selfplay}_{run_id}.json"
        archivo_modelo_hist = f"modelo_{run_id}.json"
        archivo_qstats_completo = f"qstats_completo_{run_id}.json"

        # Se carga q_stats previo si existe
        q_stats = self.cargar()

        # Historial de evaluación contra snapshots
        historial_snapshots: list[dict] = []

        # Lista de snapshots guardados (solo metadatos)
        snapshots_meta: list[dict] = []

        recompensas_selfplay: list[float] = []
        longitudes_selfplay: list[int] = []
        episodios_lista: list[int] = []

        # Variables para recordar el último snapshot para comparar
        ultimo_snapshot_q: dict | None = None
        ultimo_snapshot_epi: int | None = None

        print(
            f"\n[RUN {run_id}] Entrenando {episodios_totales} episodios con evaluación cada "
            f"{paso_eval} SOLO contra snapshots anteriores...\n"
        )

        for epi in range(1, episodios_totales + 1):
            recompensa, longitud = self.jugar_una_partida(q_stats)
            recompensas_selfplay.append(float(recompensa))
            longitudes_selfplay.append(int(longitud))
            episodios_lista.append(epi)

            print(
                f"(RUN {run_id}) Episodio {epi}/{episodios_totales} "
                f"- recompensa={recompensa}, longitud={longitud}"
            )

            # Cada 'paso_eval' se evalúa y se guarda un snapshot
            if epi % paso_eval == 0:
                # 1) Evaluación contra snapshot anterior (si existe)
                if ultimo_snapshot_q is not None and ultimo_snapshot_epi is not None:
                    resultado_snap = self.evaluar_entre_versiones(
                        q_actual=q_stats,
                        q_oponente=ultimo_snapshot_q,
                        partidas=partidas_eval,
                    )
                    punto_snap = {
                        "episodios": epi,
                        "vs_snapshot_de": int(ultimo_snapshot_epi),
                        "winrate": resultado_snap["winrate"],
                        "victorias": resultado_snap["victorias"],
                        "empates": resultado_snap["empates"],
                        "derrotas": resultado_snap["derrotas"],
                        "partidas_eval": resultado_snap["partidas"],
                    }
                    historial_snapshots.append(punto_snap)

                    print(
                        f"[RUN {run_id}] [Episodio {epi}] vs snapshot de episodio {ultimo_snapshot_epi}: "
                        f"winrate={resultado_snap['winrate']:.3f} "
                        f"(V={resultado_snap['victorias']}, "
                        f"E={resultado_snap['empates']}, "
                        f"D={resultado_snap['derrotas']})"
                    )

                # 2) Guardar snapshot actual (solo Q) y actualizar referencia
                snapshot_file = f"snapshot_{run_id}_ep{epi:04d}.json"
                self.guardar_solo_Q(q_stats, archivo=snapshot_file)
                snapshots_meta.append({
                    "episodio": epi,
                    "archivo": snapshot_file,
                })

                # Se hace una copia profunda de q_stats para usarla como oponente en la próxima evaluación
                ultimo_snapshot_q = copy.deepcopy(q_stats)
                ultimo_snapshot_epi = epi

        # Guardado del modelo simple (para Policy.py) y del histórico
        self.guardar_solo_Q(q_stats)  # connect4_model.json
        self.guardar_solo_Q(q_stats, archivo=archivo_modelo_hist)  # modelo_<run_id>.json
        self.guardar_qstats_completo(q_stats, archivo=archivo_qstats_completo)  # N y Q

        # Curvas de evaluación vs snapshots
        salida_curvas = {
            "metadata": {
                "run_id": run_id,
                "episodios_totales": episodios_totales,
                "paso_eval": paso_eval,
                "partidas_eval": partidas_eval,
                "snapshots_guardados": snapshots_meta,
            },
            "curvas_vs_snapshots": historial_snapshots,
        }
        with open(archivo_metricas, "w") as f:
            json.dump(salida_curvas, f, indent=2)

        # Estadísticas de self-play episodio a episodio
        selfplay_data = {
            "metadata": {
                "run_id": run_id,
                "episodios_totales": episodios_totales,
            },
            "episodios": episodios_lista,
            "recompensas": recompensas_selfplay,
            "longitudes": longitudes_selfplay,
        }
        with open(archivo_selfplay, "w") as f:
            json.dump(selfplay_data, f, indent=2)

        print(f"\n[RUN {run_id}] Se guardaron las curvas de aprendizaje en '{archivo_metricas}'.")
        print(f"[RUN {run_id}] Se guardaron las estadísticas de self-play en '{archivo_selfplay}'.")
        print(f"[RUN {run_id}] Modelo histórico (solo Q) guardado en '{archivo_modelo_hist}'.")
        print(f"[RUN {run_id}] q_stats completo con N y Q guardado en '{archivo_qstats_completo}'.")
        print(f"[RUN {run_id}] Entrenamiento terminado.")


# =============================================================
# ---------------------- EJECUCIÓN DIRECTA ---------------------
# =============================================================

if __name__ == "__main__":
    entrenador = EntrenadorConnect4()
    entrenador.entrenar_con_snapshots(
        episodios_totales=200,
        paso_eval=20,          # snapshot + evaluación cada 20 episodios
        partidas_eval=50,
        base_metricas="curvas_aprendizaje_snapshots",
        base_selfplay="selfplay_stats",
        run_id=None,  # Si se deja en None, se genera un timestamp automáticamente
    )