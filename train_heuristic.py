import json 
import math
import random
from datetime import datetime

import numpy as np


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

    donde el tablero se aplanada fila por fila y cada celda es -1, 0 o 1.
    """
    jugador = turno_actual(tablero)
    plano = tablero.flatten()
    representacion = "".join(str(int(celda)) for celda in plano)
    return f"{jugador}|{representacion}"


# =============================================================
# -------------- OPONENTE HEURÍSTICO MEDIUM -------------------
# =============================================================

class HeuristicOpponentMedium:
    """
    Oponente determinista "Medium":

      1) Si puede ganar en una jugada, lo hace.
      2) Si el rival gana en una, bloquea.
      3) Prefiere columnas centrales.
      4) Evita (cuando puede) jugadas que den victoria inmediata al rival
         (un paso de lookahead defensivo).
    """

    def elegir_accion(self, tablero: np.ndarray, jugador: int) -> int:
        legales = movimientos_legales(tablero)
        if not legales:
            return 0  # valor por defecto si algo raro pasa

        rival = -jugador

        # 1) Si puedo ganar en una jugada, la juego
        for c in legales:
            siguiente = colocar(tablero, c, jugador)
            if siguiente is not None and hay_ganador(siguiente, jugador):
                return c

        # 2) Si el rival gana en una jugada, bloqueo esa columna
        for c in legales:
            siguiente = colocar(tablero, c, rival)
            if siguiente is not None and hay_ganador(siguiente, rival):
                return c

        # 3) Orden de preferencia por cercanía al centro
        orden_preferencia = sorted(
            legales,
            key=lambda col: abs(col - (COLUMNAS // 2))
        )

        # 4) Evitar jugadas que permiten victoria inmediata del rival
        #    (si hay alguna jugada "segura", se prefiere)
        jugadas_seguras: list[int] = []
        for c in orden_preferencia:
            siguiente = colocar(tablero, c, jugador)
            if siguiente is None:
                continue
            # ¿El rival puede ganar inmediatamente después?
            movs_rival = movimientos_legales(siguiente)
            peligro = False
            for cr in movs_rival:
                siguiente2 = colocar(siguiente, cr, rival)
                if siguiente2 is not None and hay_ganador(siguiente2, rival):
                    peligro = True
                    break
            if not peligro:
                jugadas_seguras.append(c)

        if jugadas_seguras:
            return jugadas_seguras[0]

        # Si no hay jugadas "seguras", se juega según preferencia simple
        return orden_preferencia[0]


# =============================================================
# -------------------- Q-LEARNING TABULAR ---------------------
# =============================================================

def epsilon_greedy(q_table: dict, estado: str, legales: list[int], epsilon: float) -> int:
    """
    Política epsilon-greedy sobre Q-table (estado -> acción -> valor).

    AHORA q_table se asume de la forma:
        q_table[estado][accion] = {"Q": float, "N": int}
    """
    if not legales:
        return 0

    # Explorar
    if random.random() < epsilon:
        return random.choice(legales)

    # Explotar
    acciones_info = q_table.get(estado, {})
    mejor_accion = None
    mejor_q = -1e18
    for a in legales:
        info = acciones_info.get(a)
        q_val = float(info.get("Q", 0.0)) if info is not None else 0.0
        if q_val > mejor_q:
            mejor_q = q_val
            mejor_accion = a

    if mejor_accion is not None:
        return mejor_accion

    # Si no hay nada aprendido, elige una legal cualquiera
    return legales[0]


def actualizar_q(
    q_table: dict,
    estado: str,
    accion: int,
    recompensa: float,
    estado_siguiente: str | None,
    legales_siguiente: list[int] | None,
    alpha: float,
    gamma: float,
) -> None:
    """
    Actualización estándar de Q-learning:

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    Además, se lleva un conteo de visitas N(s,a) para análisis.
    """
    if estado not in q_table:
        q_table[estado] = {}
    if accion not in q_table[estado]:
        q_table[estado][accion] = {"Q": 0.0, "N": 0}

    q_sa = float(q_table[estado][accion]["Q"])

    if estado_siguiente is None or not legales_siguiente:
        objetivo = recompensa
    else:
        acciones_info = q_table.get(estado_siguiente, {})
        max_q_next = 0.0
        for a in legales_siguiente:
            info_next = acciones_info.get(a)
            q_next = float(info_next.get("Q", 0.0)) if info_next is not None else 0.0
            if q_next > max_q_next:
                max_q_next = q_next
        objetivo = recompensa + gamma * max_q_next

    nuevo_q = q_sa + alpha * (objetivo - q_sa)
    q_table[estado][accion]["Q"] = nuevo_q
    q_table[estado][accion]["N"] += 1  # incrementamos visitas de (s,a)


# =============================================================
# ---------------------- ENTRENAMIENTO ------------------------
# =============================================================

def jugar_episodio_qlearning(
    q_table: dict,
    oponente: HeuristicOpponentMedium,
    alpha: float,
    gamma: float,
    epsilon: float,
) -> tuple[int, int, int]:
    """
    Juega UN episodio (partida completa) de Q-learning contra el oponente
    heurístico. El "agente" es siempre el que tiene el papel 'agent_player'
    elegido aleatoriamente en {-1, 1}.

    Devuelve:
      - recompensa_final desde el punto de vista del agente:
          +1 si gana, -1 si pierde, 0 si hay empate.
      - longitud de la partida en número de jugadas totales.
      - agent_player (-1 o 1), para análisis de quién empezó / rol.
    """
    tablero = np.zeros((FILAS, COLUMNAS), dtype=int)

    # Decidimos al azar si el agente es -1 o 1 en este episodio
    agent_player = random.choice([-1, 1])
    opponent_player = -agent_player  # se deja por claridad, aunque no se use directamente

    # Para registrar los pasos del agente: (estado, acción)
    trayectoria: list[tuple[str, int]] = []
    num_movimientos = 0

    while True:
        jugador = turno_actual(tablero)

        # ¿Alguien ganó en la jugada anterior?
        if hay_ganador(tablero, -jugador):
            ganador = -jugador
            if ganador == agent_player:
                recompensa_final = 1
            else:
                recompensa_final = -1
            break

        movs = movimientos_legales(tablero)
        if not movs:
            # Empate
            recompensa_final = 0
            break

        if jugador == agent_player:
            # Turno del agente (controlado por Q-learning)
            estado = codificar_estado(tablero)
            accion = epsilon_greedy(q_table, estado, movs, epsilon)
            tablero_siguiente = colocar(tablero, accion, jugador)
            if tablero_siguiente is None:
                # Columna llena (no debería pasar). Castigo suave.
                recompensa_final = -1
                break

            # Guardamos la transición (estado, acción) para después actualizar
            trayectoria.append((estado, accion))
            tablero = tablero_siguiente
            num_movimientos += 1

        else:
            # Turno del oponente heurístico
            accion = oponente.elegir_accion(tablero, jugador)
            tablero_siguiente = colocar(tablero, accion, jugador)
            if tablero_siguiente is None:
                # Si el oponente hace una jugada ilegal, lo consideramos derrota suya.
                recompensa_final = 1 if agent_player == jugador else -1
                break
            tablero = tablero_siguiente
            num_movimientos += 1

    # Actualización backward de Q para todas las jugadas del agente
    # Recompensa solo al final (r=0 en pasos intermedios, r_final al final).
    # Aquí hacemos un "Monte Carlo con bootstrap": usamos el mismo r_final
    # para todas las decisiones, y bootstrapeamos con max Q del siguiente estado.
    for i, (estado, accion) in enumerate(trayectoria):
        if i == len(trayectoria) - 1:
            # Última decisión del agente: estado terminal después
            actualizar_q(
                q_table,
                estado,
                accion,
                recompensa_final,
                estado_siguiente=None,
                legales_siguiente=None,
                alpha=alpha,
                gamma=gamma,
            )
        else:
            # Estados intermedios: recompensa 0, pero se hace bootstrap con s_{t+1}
            siguiente_estado, _ = trayectoria[i + 1]
            # Para s_{t+1}, estimamos movimientos legales "típicos"
            # (no es perfecto, pero funcionará para el bootstrap)
            # NOTA: para evitar recomputar, podríamos guardar también el tablero,
            # pero así mantenemos el código más simple.
            # Como codificar_estado incluye el tablero, no podemos revertir al tablero
            # exacto sin guardarlo, pero un bootstrap aproximado con r=0 también
            # permite algo de propagación. Si se quiere más fino, se puede almacenar
            # también el tablero en la trayectoria.
            actualizar_q(
                q_table,
                estado,
                accion,
                recompensa=0.0,
                estado_siguiente=siguiente_estado,
                legales_siguiente=list(range(COLUMNAS)),  # aproximación
                alpha=alpha,
                gamma=gamma,
            )

    return recompensa_final, num_movimientos, agent_player


def evaluar_vs_heuristic(q_table: dict, oponente: HeuristicOpponentMedium, partidas: int = 200) -> dict:
    """
    Evalúa el agente (usando política greedy, epsilon=0) contra el oponente
    heurístico. Alterna quién es el agente (a veces -1, a veces 1).
    """
    victorias = 0
    empates = 0
    derrotas = 0

    for _ in range(partidas):
        tablero = np.zeros((FILAS, COLUMNAS), dtype=int)
        agent_player = random.choice([-1, 1])
        opponent_player = -agent_player

        while True:
            jugador = turno_actual(tablero)

            if hay_ganador(tablero, -jugador):
                ganador = -jugador
                if ganador == agent_player:
                    victorias += 1
                elif ganador == opponent_player:
                    derrotas += 1
                else:
                    empates += 1
                break

            movs = movimientos_legales(tablero)
            if not movs:
                empates += 1
                break

            if jugador == agent_player:
                estado = codificar_estado(tablero)
                accion = epsilon_greedy(q_table, estado, movs, epsilon=0.0)  # greedy
                tablero_siguiente = colocar(tablero, accion, jugador)
                if tablero_siguiente is None:
                    # Jugada ilegal del agente → derrota
                    derrotas += 1
                    break
                tablero = tablero_siguiente
            else:
                accion = oponente.elegir_accion(tablero, jugador)
                tablero_siguiente = colocar(tablero, accion, jugador)
                if tablero_siguiente is None:
                    # Jugada ilegal del oponente → victoria del agente
                    victorias += 1
                    break
                tablero = tablero_siguiente

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
# ---------------------- GUARDADO / CARGA ---------------------
# =============================================================

def cargar_modelo_simple_a_qtable(archivo: str) -> dict[str, dict[int, dict[str, float]]]:
    """
    Carga un modelo en formato simple:

        {
          "estado": { "columna": Q_value, ... },
          ...
        }

    (el formato que usa tu policy) y lo convierte a:

        q_table[estado][accion] = {"Q": float, "N": int}

    De esta forma, si el archivo existe, el entrenamiento continúa de forma
    incremental sobre el conocimiento previo.
    """
    try:
        with open(archivo, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except Exception:
        return {}

    if not isinstance(datos, dict):
        return {}

    q_table: dict[str, dict[int, dict[str, float]]] = {}

    for estado, acciones in datos.items():
        if not isinstance(acciones, dict):
            continue
        q_table[estado] = {}
        for a_str, q_val in acciones.items():
            try:
                a_int = int(a_str)
            except Exception:
                continue
            try:
                q_float = float(q_val)
            except Exception:
                q_float = 0.0
            # N se inicializa en 0 porque no sabemos las visitas históricas.
            q_table[estado][a_int] = {"Q": q_float, "N": 0}

    return q_table


def guardar_modelo_simple(q_table: dict, archivo: str) -> None:
    """
    Guarda la Q-table en formato simple:

        {
          "estado": { "columna": Q_value, ... },
          ...
        }

    donde "columna" es un string con el índice de columna.
    ESTE ES EL FORMATO QUE USA TU policy.CetinaSalasSabogal.

    AHORA q_table puede tener la forma:
        q_table[estado][accion] = {"Q": float, "N": int}
    o directamente valores float (para compatibilidad).
    """
    salida: dict[str, dict[str, float]] = {}
    for estado, acciones in q_table.items():
        acciones_q: dict[str, float] = {}
        for a_int, val in acciones.items():
            if isinstance(val, dict):
                q_val = float(val.get("Q", 0.0))
            else:
                q_val = float(val)
            acciones_q[str(a_int)] = q_val
        if acciones_q:
            salida[estado] = acciones_q

    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(salida, f)

    print(f"\nSe guardó modelo final en '{archivo}' con {len(salida)} estados.")


def guardar_qstats_completo(q_table: dict, archivo: str) -> None:
    """
    Guarda la tabla completa incluyendo tanto N como Q por acción.
    Este archivo no lo usa la policy en producción, pero sirve para
    análisis e inspección más profunda del proceso de aprendizaje.

    Se asume que:
        q_table[estado][accion] = {"Q": float, "N": int}
    """
    salida: dict[str, dict[str, dict[str, float]]] = {}
    for estado, acciones in q_table.items():
        acciones_completas: dict[str, dict[str, float]] = {}
        for a_int, info in acciones.items():
            if isinstance(info, dict):
                n_val = int(info.get("N", 0))
                q_val = float(info.get("Q", 0.0))
            else:
                # por compatibilidad, si viniera solo el Q
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
    Calcula media móvil simple sobre 'lista' con una ventana dada.
    Se usa solo para guardar curvas suavizadas en el JSON de self-play.
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
# -------------------------- MAIN -----------------------------
# =============================================================

if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[RUN {run_id}] Entrenando Q-learning contra HeuristicOpponentMedium...\n")

    # Hiperparámetros
    episodios_totales = 4000
    alpha = 0.1
    gamma = 0.95
    epsilon_inicial = 0.2
    epsilon_final = 0.01
    eval_interval = 200
    eval_partidas = 200

    # AHORA q_table es un diccionario de N y Q por acción:
    #   q_table[estado][accion] = {"Q": float, "N": int}
    # Intentamos cargar conocimiento previo desde connect4_model.json
    q_table: dict[str, dict[int, dict[str, float]]]
    q_table = cargar_modelo_simple_a_qtable("connect4_model.json")
    if q_table:
        print(f"[RUN {run_id}] Se cargó modelo previo desde 'connect4_model.json' con {len(q_table)} estados.\n")
    else:
        print(f"[RUN {run_id}] No se encontró modelo previo o estaba vacío. Se inicia desde cero.\n")
        q_table = {}

    oponente = HeuristicOpponentMedium()

    recompensas_ultimos: list[float] = []
    historial_curva: list[dict] = []

    # Listas para guardar métricas de self-play episodio a episodio
    episodios_lista: list[int] = []
    recompensas_episodios: list[float] = []
    longitudes_episodios: list[int] = []
    agent_roles: list[int] = []  # -1 o 1, rol del agente en cada episodio

    for epi in range(1, episodios_totales + 1):
        # Decaimiento lineal sencillo de epsilon
        t = epi / episodios_totales
        epsilon = max(epsilon_final, epsilon_inicial * (1 - t) + epsilon_final * t)

        r, longitud, agent_player = jugar_episodio_qlearning(
            q_table=q_table,
            oponente=oponente,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
        )

        # Métricas de self-play
        episodios_lista.append(epi)
        recompensas_episodios.append(float(r))
        longitudes_episodios.append(int(longitud))
        agent_roles.append(int(agent_player))

        # Ventana móvil de las últimas 50 recompensas
        recompensas_ultimos.append(r)
        if len(recompensas_ultimos) > 50:
            recompensas_ultimos.pop(0)

        media_ultimos = sum(recompensas_ultimos) / len(recompensas_ultimos)

        if epi == 1 or epi % 50 == 0:
            print(
                f"[RUN {run_id}] Episodio {epi}/{episodios_totales} - "
                f"recompensa={r}, epsilon={epsilon:.3f}, "
                f"media_ultimos_50={media_ultimos:.3f}, "
                f"longitud={longitud}, agent_player={agent_player}"
            )

        if epi % eval_interval == 0:
            resultado_eval = evaluar_vs_heuristic(q_table, oponente, partidas=eval_partidas)
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
                f"[RUN {run_id}] [Evaluación ep {epi}] vs HeuristicOpponentMedium: "
                f"winrate={resultado_eval['winrate']:.3f} "
                f"(V={resultado_eval['victorias']}, "
                f"E={resultado_eval['empates']}, "
                f"D={resultado_eval['derrotas']})"
            )

    # ---------------------------------------------------------
    # Guardar modelo final en el formato que usa tu policy
    # (se sobreescribe connect4_model.json con el conocimiento acumulado)
    # ---------------------------------------------------------
    guardar_modelo_simple(q_table, "connect4_model.json")

    # Copia histórica del modelo simple (solo Q)
    archivo_modelo_hist = f"modelo_qlearning_{run_id}.json"
    guardar_modelo_simple(q_table, archivo_modelo_hist)

    # Guardar q_stats completo con N y Q
    archivo_qstats_completo = f"qstats_completo_{run_id}.json"
    guardar_qstats_completo(q_table, archivo_qstats_completo)

    # ---------------------------------------------------------
    # Guardar curva de aprendizaje vs heurístico
    # ---------------------------------------------------------
    curva = {
        "metadata": {
            "run_id": run_id,
            "episodios_totales": episodios_totales,
            "paso_eval": eval_interval,
            "partidas_eval": eval_partidas,
            "epsilon_inicial": epsilon_inicial,
            "epsilon_final": epsilon_final,
            "oponente": "HeuristicOpponentMedium",
        },
        "historial": historial_curva,
    }
    archivo_curva = f"curva_vs_heuristic_medium_{run_id}.json"
    with open(archivo_curva, "w", encoding="utf-8") as f:
        json.dump(curva, f, indent=2)

    # ---------------------------------------------------------
    # Guardar estadísticas de self-play episodio a episodio
    # ---------------------------------------------------------
    # Resumen global de victorias/empates/derrotas (desde el punto de vista del agente)
    victorias_sp = sum(1 for r in recompensas_episodios if r > 0)
    derrotas_sp = sum(1 for r in recompensas_episodios if r < 0)
    empates_sp = sum(1 for r in recompensas_episodios if r == 0)
    total_sp = len(recompensas_episodios)

    # Medias móviles (por si quieres graficar curvas suavizadas directo del JSON)
    recompensas_mm = media_movil(recompensas_episodios, ventana=50)
    longitudes_mm = media_movil([float(l) for l in longitudes_episodios], ventana=50)

    selfplay_data = {
        "metadata": {
            "run_id": run_id,
            "episodios_totales": episodios_totales,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_inicial": epsilon_inicial,
            "epsilon_final": epsilon_final,
            "oponente": "HeuristicOpponentMedium",
        },
        "episodios": episodios_lista,
        "recompensas": recompensas_episodios,
        "longitudes": longitudes_episodios,
        "agent_player": agent_roles,
        "recompensas_media_movil_50": recompensas_mm,
        "longitudes_media_movil_50": longitudes_mm,
        "resumen_global": {
            "victorias": victorias_sp,
            "empates": empates_sp,
            "derrotas": derrotas_sp,
            "total": total_sp,
        },
    }
    archivo_selfplay = f"selfplay_qlearning_vs_heuristic_medium_{run_id}.json"
    with open(archivo_selfplay, "w", encoding="utf-8") as f:
        json.dump(selfplay_data, f, indent=2)

    print(f"\n[RUN {run_id}] Curva de aprendizaje guardada en '{archivo_curva}'.")
    print(f"[RUN {run_id}] Estadísticas de self-play guardadas en '{archivo_selfplay}'.")
    print(f"[RUN {run_id}] Modelo histórico (solo Q) guardado en '{archivo_modelo_hist}'.")
    print(f"[RUN {run_id}] q_stats completo con N y Q guardado en '{archivo_qstats_completo}'.")
    print(f"[RUN {run_id}] Entrenamiento contra HeuristicOpponentMedium terminado.")