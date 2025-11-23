import json
import math
import random
from datetime import datetime

import numpy as np

# Si ya tienes este RandomPolicy en tu proyecto, debería importar bien:
from groups.Random.random_policy import RandomPolicy


# =============================================================
# ----------------- UTILIDADES DEL JUEGO ----------------------
# =============================================================

FILAS = 6
COLUMNAS = 7


def movimientos_legales(tablero: np.ndarray) -> list[int]:
    """Devuelve las columnas cuyo tope está libre."""
    return [c for c in range(COLUMNAS) if tablero[0, c] == 0]


def colocar(tablero: np.ndarray, columna: int, jugador: int) -> np.ndarray | None:
    """
    Deja caer una ficha en la columna dada para 'jugador'.
    Devuelve una copia del tablero resultante, o None si la columna está llena.
    """
    for fila in range(FILAS - 1, -1, -1):
        if tablero[fila, columna] == 0:
            nuevo = tablero.copy()
            nuevo[fila, columna] = jugador
            return nuevo
    return None


def hay_ganador(tablero: np.ndarray, jugador: int) -> bool:
    """Comprueba si 'jugador' tiene cuatro en línea."""
    # Horizontal
    for f in range(FILAS):
        for c in range(COLUMNAS - 3):
            if np.all(tablero[f, c:c + 4] == jugador):
                return True
    # Vertical
    for f in range(FILAS - 3):
        for c in range(COLUMNAS):
            if np.all(tablero[f:f + 4, c] == jugador):
                return True
    # Diagonal principal
    for f in range(FILAS - 3):
        for c in range(COLUMNAS - 3):
            if all(tablero[f + k, c + k] == jugador for k in range(4)):
                return True
    # Diagonal secundaria
    for f in range(FILAS - 3):
        for c in range(COLUMNAS - 3):
            if all(tablero[f + 3 - k, c + k] == jugador for k in range(4)):
                return True
    return False


def turno_actual(tablero: np.ndarray) -> int:
    """
    Determina de quién es el turno según la paridad de fichas colocadas.
    - Si hay cantidad par de fichas, juega -1.
    - Si hay impar, juega 1.
    """
    cantidad = int(np.count_nonzero(tablero))
    return -1 if cantidad % 2 == 0 else 1


def codificar_estado(tablero: np.ndarray) -> str:
    """
    Representación textual del estado, COMPATIBLE con tu policy:

        "<jugador_que_mueve>|<tablero_aplanado>"

    donde el tablero se aplana fila por fila y cada celda es -1, 0 o 1.
    """
    jugador = turno_actual(tablero)
    plano = tablero.flatten()
    representacion = "".join(str(int(celda)) for celda in plano)
    return f"{jugador}|{representacion}"


# =============================================================
# ------------------------ NODO MCTS --------------------------
# =============================================================

class NodoMCTS:
    """
    Nodo del árbol para MCTS-UCB.

    Guarda:
      - tablero y jugador al turno
      - padre y acción que llevó aquí
      - hijos (acción -> nodo)
      - lista de acciones no exploradas
      - N: nº visitas
      - W: suma de recompensas (desde la perspectiva del jugador raíz)
    """

    def __init__(
        self,
        tablero: np.ndarray,
        jugador: int,
        padre: "NodoMCTS | None" = None,
        accion: int | None = None,
    ) -> None:
        self.tablero = tablero
        self.jugador = jugador
        self.padre = padre
        self.accion = accion

        self.hijos: dict[int, NodoMCTS] = {}
        self.no_exploradas: list[int] = movimientos_legales(tablero)

        self.N: int = 0
        self.W: float = 0.0  # suma de recompensas (desde la raíz)

    def es_terminal(self) -> bool:
        """
        Estado terminal si hay ganador o no hay más movimientos legales.
        """
        return (
            hay_ganador(self.tablero, 1)
            or hay_ganador(self.tablero, -1)
            or len(movimientos_legales(self.tablero)) == 0
        )

    def expandir(self) -> "NodoMCTS":
        """
        Toma una acción aún no explorada y crea el nodo hijo.
        """
        accion = self.no_exploradas.pop()
        nuevo_tablero = colocar(self.tablero, accion, self.jugador)
        hijo = NodoMCTS(
            nuevo_tablero,
            -self.jugador,
            padre=self,
            accion=accion,
        )
        self.hijos[accion] = hijo
        return hijo

    def seleccionar_hijo_ucb(self, c: float = math.sqrt(2.0)) -> "NodoMCTS":
        """
        Selecciona hijo usando la fórmula UCB1.
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
# ---------------------- ROLLOUT RANDOM -----------------------
# =============================================================

def rollout(tablero: np.ndarray, jugador_raiz: int) -> int:
    """
    Simulación aleatoria completa desde 'tablero' hasta final.

    Devuelve recompensa desde la perspectiva de jugador_raiz:
      +1 si gana, -1 si pierde, 0 si empate.
    """
    b = tablero.copy()
    turno = turno_actual(b)

    while True:
        # ¿ganó el que jugó antes?
        if hay_ganador(b, -turno):
            ganador = -turno
            if ganador == jugador_raiz:
                return 1
            else:
                return -1

        movs = movimientos_legales(b)
        if not movs:
            return 0

        accion = random.choice(movs)
        b = colocar(b, accion, turno)
        turno = -turno


# =============================================================
# ------------------ SELF-PLAY MCTS PARA ENTRENAR -------------
# =============================================================

def jugar_una_partida_mcts(
    q_stats: dict,
    simulaciones: int = 150,
) -> tuple[int, int]:
    """
    Juega UNA partida de self-play donde ambos jugadores usan MCTS-UCB
    para elegir sus movimientos. Al final, actualiza q_stats:

        q_stats[estado][accion] = {"N": visitas, "Q": valor_medio}

    usando Monte Carlo (promedio de recompensa final).

    Devuelve:
      - recompensa final desde el punto de vista del jugador inicial
      - longitud de la partida
    """
    tablero = np.zeros((FILAS, COLUMNAS), dtype=int)
    jugador = turno_actual(tablero)
    jugador_inicial = jugador

    episodio: list[tuple[str, int, int]] = []  # (estado, accion, jugador_que_movio)
    movimientos = 0

    while True:
        # ¿ganó el que movió en el turno anterior?
        if hay_ganador(tablero, -jugador):
            ganador = -jugador
            recompensa = 1 if ganador == jugador_inicial else -1
            break

        movs = movimientos_legales(tablero)
        if not movs:
            recompensa = 0
            break

        # Nodo raíz del MCTS para la posición actual
        raiz = NodoMCTS(tablero, jugador)

        # MCTS desde la raíz
        for _ in range(simulaciones):
            nodo = raiz

            # 1) Selección
            while not nodo.no_exploradas and nodo.hijos:
                nodo = nodo.seleccionar_hijo_ucb()

            # 2) Expansión
            if nodo.no_exploradas:
                nodo = nodo.expandir()

            # 3) Simulación (siempre respecto a jugador raíz = jugador)
            r = rollout(nodo.tablero, jugador_raiz=jugador)

            # 4) Backpropagation
            while nodo is not None:
                nodo.N += 1
                nodo.W += r
                nodo = nodo.padre

        # Acción real: hijo de la raíz con más visitas
        if not raiz.hijos:
            # fallback muy raro
            mejor_accion = random.choice(movs)
        else:
            mejor_accion = max(raiz.hijos.items(), key=lambda t: t[1].N)[0]

        estado = codificar_estado(tablero)
        episodio.append((estado, mejor_accion, jugador))

        tablero = colocar(tablero, mejor_accion, jugador)
        jugador = -jugador
        movimientos += 1

    # Actualización de Q(s,a) por Monte Carlo para TODOS los pasos del episodio
    for estado, accion, jugador_que_movio in episodio:
        if estado not in q_stats:
            q_stats[estado] = {}
        if accion not in q_stats[estado]:
            q_stats[estado][accion] = {"N": 0, "Q": 0.0}

        # si quien movió es el inicial, ve recompensa tal cual; si fue el rival, con signo invertido
        r_efectiva = recompensa if jugador_que_movio == jugador_inicial else -recompensa

        info = q_stats[estado][accion]
        N = int(info.get("N", 0)) + 1
        oldQ = float(info.get("Q", 0.0))
        newQ = oldQ + (r_efectiva - oldQ) / N

        info["N"] = N
        info["Q"] = newQ

    return recompensa, movimientos


# =============================================================
# --------- ELECCIÓN DE ACCIÓN A PARTIR DE Q (POLICY) ---------
# =============================================================

def elegir_accion_q_con_tactica(tablero: np.ndarray, jugador: int, q_stats: dict) -> int:
    """
    Política que IMITA tu policy final CetinaSalasSabogal:

      1) Si hay jugada ganadora inmediata, la juega.
      2) Si el rival gana en una, bloquea.
      3) Si no, usa Q(s,a) (policy greedy).
      4) Si no hay Q para este estado, usa heurística centrada.
    """
    movs = movimientos_legales(tablero)
    if not movs:
        return 0

    rival = -jugador

    # 1) Ganar en una
    for c in movs:
        siguiente = colocar(tablero, c, jugador)
        if siguiente is not None and hay_ganador(siguiente, jugador):
            return c

    # 2) Bloquear victoria inmediata del rival
    for c in movs:
        siguiente = colocar(tablero, c, rival)
        if siguiente is not None and hay_ganador(siguiente, rival):
            return c

    # 3) Q-greedy
    estado = codificar_estado(tablero)
    acciones_info = q_stats.get(estado, {})

    mejor_accion = None
    mejor_q = -1e18
    for c in movs:
        info = acciones_info.get(c)
        q_val = float(info.get("Q", 0.0)) if info is not None else 0.0
        if q_val > mejor_q:
            mejor_q = q_val
            mejor_accion = c

    if mejor_accion is not None:
        return mejor_accion

    # 4) Heurística centrada
    preferencia = [3, 2, 4, 1, 5, 0, 6]
    for c in preferencia:
        if c in movs:
            return c

    return movs[0]


# =============================================================
# ------------- EVALUACIÓN VS RANDOMPOLICY (Q-SOLO) ----------
# =============================================================

def evaluar_vs_random(q_stats: dict, partidas: int = 200) -> dict:
    """
    Evalúa el agente que juega con Q + táctica (ganar/bloquear) contra
    RandomPolicy. Alterna quién empieza.
    """
    victorias = 0
    empates = 0
    derrotas = 0

    for i in range(partidas):
        tablero = np.zeros((FILAS, COLUMNAS), dtype=int)

        # jugador -1 empieza en el tablero
        jugador_tablero = -1

        # Alternamos quién es el agente
        if i % 2 == 0:
            jugador_agente = -1
            jugador_random = 1
        else:
            jugador_agente = 1
            jugador_random = -1

        oponente = RandomPolicy()
        try:
            oponente.mount(None)
        except TypeError:
            try:
                oponente.mount()
            except Exception:
                pass

        while True:
            # ¿acabó?
            if hay_ganador(tablero, jugador_agente):
                victorias += 1
                break
            if hay_ganador(tablero, jugador_random):
                derrotas += 1
                break

            movs = movimientos_legales(tablero)
            if not movs:
                empates += 1
                break

            if jugador_tablero == jugador_agente:
                accion = elegir_accion_q_con_tactica(tablero, jugador_agente, q_stats)
            else:
                accion = oponente.act(tablero)

            if accion not in movs:
                # jugada ilegal → consideramos derrota de quien la hace
                if jugador_tablero == jugador_agente:
                    derrotas += 1
                else:
                    victorias += 1
                break

            tablero = colocar(tablero, accion, jugador_tablero)
            jugador_tablero = -jugador_tablero

    total = victorias + empates + derrotas
    winrate = victorias / total if total > 0 else 0.0
    return {
        "winrate": winrate,
        "victorias": victorias,
        "empates": empates,
        "derrotas": derrotas,
        "partidas_eval": total,
    }


# =============================================================
# ------------------ GUARDADO DEL MODELO ----------------------
# =============================================================

def guardar_modelo_simple(q_stats: dict, archivo: str) -> None:
    """
    Transforma q_stats en el formato que tu policy espera:

        {
            "estado": { "columna": Q_value, ... },
            ...
        }
    """
    salida: dict[str, dict[str, float]] = {}
    for estado, acciones in q_stats.items():
        acciones_q: dict[str, float] = {}
        for a_int, info in acciones.items():
            if isinstance(info, dict):
                q_val = float(info.get("Q", 0.0))
            else:
                q_val = float(info)
            acciones_q[str(a_int)] = q_val
        if acciones_q:
            salida[estado] = acciones_q

    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(salida, f)

    print(f"\nSe guardó modelo final en '{archivo}' con {len(salida)} estados.")


def guardar_qstats_completo(q_stats: dict, archivo: str) -> None:
    """
    Guarda q_stats incluyendo N y Q por acción.
    """
    salida: dict[str, dict[str, dict[str, float]]] = {}
    for estado, acciones in q_stats.items():
        acciones_completas: dict[str, dict[str, float]] = {}
        for a_int, info in acciones.items():
            if isinstance(info, dict):
                n_val = int(info.get("N", 0))
                q_val = float(info.get("Q", 0.0))
            else:
                n_val = 0
                q_val = float(info)
            acciones_completas[str(a_int)] = {"N": n_val, "Q": q_val}
        if acciones_completas:
            salida[estado] = acciones_completas

    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2)

    print(f"\nSe guardó q_stats completo con N y Q en '{archivo}'.")


def media_movil(lista: list[float], ventana: int = 50) -> list[float]:
    """
    Media móvil simple para suavizar curvas.
    """
    if ventana <= 1:
        return [float(x) for x in lista]

    salida: list[float] = []
    for i in range(len(lista)):
        ini = max(0, i - ventana + 1)
        segmento = lista[ini:i + 1]
        salida.append(sum(segmento) / len(segmento))
    return salida


# =============================================================
# --------------------------- MAIN ----------------------------
# =============================================================

if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[RUN {run_id}] Entrenando con self-play MCTS-UCB...\n")

    # Hiperparámetros
    episodios_totales = 4000         # puedes subir/bajar
    simulaciones_por_movimiento = 120
    eval_interval = 200
    eval_partidas = 200

    # Q-table con N y Q
    q_stats: dict[str, dict[int, dict[str, float]]] = {}

    # métricas de self-play
    episodios_lista: list[int] = []
    recompensas_episodios: list[float] = []
    longitudes_episodios: list[int] = []
    recompensas_ultimos: list[float] = []
    historial_curva: list[dict] = []

    for epi in range(1, episodios_totales + 1):
        r, longitud = jugar_una_partida_mcts(
            q_stats=q_stats,
            simulaciones=simulaciones_por_movimiento,
        )

        episodios_lista.append(epi)
        recompensas_episodios.append(float(r))
        longitudes_episodios.append(int(longitud))

        # media móvil de las últimas 50 recompensas
        recompensas_ultimos.append(r)
        if len(recompensas_ultimos) > 50:
            recompensas_ultimos.pop(0)
        media_ultimos = sum(recompensas_ultimos) / len(recompensas_ultimos)

        if epi == 1 or epi % 50 == 0:
            print(
                f"[RUN {run_id}] Episodio {epi}/{episodios_totales} - "
                f"recompensa={r}, media_ultimos_50={media_ultimos:.3f}, "
                f"longitud={longitud}"
            )

        # evaluación periódica contra Random usando SOLO Q+heurística
        if epi % eval_interval == 0:
            resultado_eval = evaluar_vs_random(q_stats, partidas=eval_partidas)
            punto = {
                "episodios": epi,
                "winrate": resultado_eval["winrate"],
                "victorias": resultado_eval["victorias"],
                "empates": resultado_eval["empates"],
                "derrotas": resultado_eval["derrotas"],
                "partidas_eval": resultado_eval["partidas_eval"],
            }
            historial_curva.append(punto)

            print(
                f"[RUN {run_id}] [Evaluación ep {epi}] vs RandomPolicy: "
                f"winrate={resultado_eval['winrate']:.3f} "
                f"(V={resultado_eval['victorias']}, "
                f"E={resultado_eval['empates']}, "
                f"D={resultado_eval['derrotas']})"
            )

    # ------------------ GUARDADOS ------------------

    # Modelo final para tu policy
    guardar_modelo_simple(q_stats, "connect4_model.json")

    # Copia histórica solo Q
    archivo_modelo_hist = f"modelo_mcts_{run_id}.json"
    guardar_modelo_simple(q_stats, archivo_modelo_hist)

    # q_stats completo
    archivo_qstats_completo = f"qstats_completo_mcts_{run_id}.json"
    guardar_qstats_completo(q_stats, archivo_qstats_completo)

    # Curva de aprendizaje vs Random
    curva = {
        "metadata": {
            "run_id": run_id,
            "episodios_totales": episodios_totales,
            "paso_eval": eval_interval,
            "partidas_eval": eval_partidas,
            "simulaciones_por_movimiento": simulaciones_por_movimiento,
            "oponente": "RandomPolicy",
            "tipo_entrenamiento": "selfplay_MCTS_UCB",
        },
        "historial": historial_curva,
    }
    archivo_curva = f"curva_vs_random_mcts_{run_id}.json"
    with open(archivo_curva, "w", encoding="utf-8") as f:
        json.dump(curva, f, indent=2)

    # Estadísticas de self-play
    victorias_sp = sum(1 for r in recompensas_episodios if r > 0)
    derrotas_sp = sum(1 for r in recompensas_episodios if r < 0)
    empates_sp = sum(1 for r in recompensas_episodios if r == 0)
    total_sp = len(recompensas_episodios)

    recompensas_mm = media_movil(recompensas_episodios, ventana=50)
    longitudes_mm = media_movil([float(l) for l in longitudes_episodios], ventana=50)

    selfplay_data = {
        "metadata": {
            "run_id": run_id,
            "episodios_totales": episodios_totales,
            "simulaciones_por_movimiento": simulaciones_por_movimiento,
            "entrenamiento": "selfplay_MCTS_UCB",
        },
        "episodios": episodios_lista,
        "recompensas": recompensas_episodios,
        "longitudes": longitudes_episodios,
        "recompensas_media_movil_50": recompensas_mm,
        "longitudes_media_movil_50": longitudes_mm,
        "resumen_global": {
            "victorias": victorias_sp,
            "empates": empates_sp,
            "derrotas": derrotas_sp,
            "total": total_sp,
        },
    }
    archivo_selfplay = f"selfplay_mcts_{run_id}.json"
    with open(archivo_selfplay, "w", encoding="utf-8") as f:
        json.dump(selfplay_data, f, indent=2)

    print(f"\n[RUN {run_id}] Curva de aprendizaje guardada en '{archivo_curva}'.")
    print(f"[RUN {run_id}] Estadísticas de self-play guardadas en '{archivo_selfplay}'.")
    print(f"[RUN {run_id}] Modelo histórico (solo Q) guardado en '{archivo_modelo_hist}'.")
    print(f"[RUN {run_id}] q_stats completo con N y Q guardado en '{archivo_qstats_completo}'.")
    print(f"[RUN {run_id}] Entrenamiento con MCTS-UCB terminado.")