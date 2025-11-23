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
    ''' Se devuelve la lista de columnas cuyo tope en la fila superior está
    libre, considerándolas como movimientos legales disponibles. '''
    
    return [c for c in range(COLUMNAS) if tablero[0, c] == 0]

def colocar(tablero: np.ndarray, columna: int, jugador: int) -> np.ndarray | None:
    ''' Se deja caer una ficha en la columna indicada para el jugador dado y se
    devuelve una copia del tablero resultante. Si la columna ya está llena y no
    es posible colocar una ficha adicional, se devuelve None. '''
    
    for fila in range(FILAS - 1, -1, -1):
        if tablero[fila, columna] == 0:
            nuevo = tablero.copy()
            nuevo[fila, columna] = jugador
            return nuevo
    return None

def hay_ganador(tablero: np.ndarray, jugador: int) -> bool:
    ''' Se comprueba si el jugador indicado ha conseguido conectar cuatro fichas
    consecutivas en alguna dirección (horizontal, vertical o diagonal). '''
    
    # Se revisan todas las posibles alineaciones horizontales de cuatro fichas consecutivas.
    for f in range(FILAS):
        for c in range(COLUMNAS - 3):
            if np.all(tablero[f, c:c + 4] == jugador):
                
                return True
            
    # Se revisan todas las posibles alineaciones verticales de cuatro fichas consecutivas.
    for f in range(FILAS - 3):
        for c in range(COLUMNAS):
            if np.all(tablero[f:f + 4, c] == jugador):
                
                return True
            
    # Se revisan las diagonales principales con cuatro fichas consecutivas.
    for f in range(FILAS - 3):
        for c in range(COLUMNAS - 3):
            if all(tablero[f + k, c + k] == jugador for k in range(4)):
                
                return True
            
    # Se revisan las diagonales secundarias con cuatro fichas consecutivas.
    for f in range(FILAS - 3):
        for c in range(COLUMNAS - 3):
            if all(tablero[f + 3 - k, c + k] == jugador for k in range(4)):
               
                return True
            
    return False

def turno_actual(tablero: np.ndarray) -> int:
    ''' Se determina de quién es el turno a partir de la paridad del número de
    fichas colocadas en el tablero: si la cantidad de fichas es par, le
    corresponde jugar al jugador -1; si es impar, le corresponde jugar al
    jugador 1. '''
    
    cantidad = int(np.count_nonzero(tablero))
    
    return -1 if cantidad % 2 == 0 else 1

def codificar_estado(tablero: np.ndarray) -> str:
    ''' Se construye una representación textual del estado, compatible con la
    policy, utilizando el formato "<jugador_que_mueve>|<tablero_aplanado>",
    donde el tablero se recorre fila por fila y cada celda se codifica como
    -1, 0 o 1. '''

    jugador = turno_actual(tablero)
    plano = tablero.flatten()
    representacion = "".join(str(int(celda)) for celda in plano)

    return f"{jugador}|{representacion}"

# =============================================================
# -------------- OPONENTE HEURÍSTICO MEDIUM -------------------
# =============================================================

class HeuristicOpponentMedium:
    ''' Se define un oponente determinista de dificultad media que sigue una
    estrategia táctica basada en cuatro reglas principales: (1) si puede
    ganar en una jugada, ejecuta esa victoria; (2) si el rival puede ganar
    en una jugada, bloquea esa amenaza; (3) se priorizan las columnas
    centrales del tablero; y (4) se evitan, cuando es posible, jugadas que
    permitan una victoria inmediata del rival mediante un paso de lookahead
    defensivo. '''

    def elegir_accion(self, tablero: np.ndarray, jugador: int) -> int:

        legales = movimientos_legales(tablero)
        # Se devuelve una columna por defecto si no queda ningún movimiento legal, aunque esta situación no debería ocurrir.
        if not legales:
            
            return 0

        rival = -jugador

        # Se verifica si existe alguna jugada inmediata de victoria para el jugador actual.
        for c in legales:
            siguiente = colocar(tablero, c, jugador)
            if siguiente is not None and hay_ganador(siguiente, jugador):
                
                return c

        # Se verifica si el rival podría ganar en una jugada y se intenta bloquear esa columna.
        for c in legales:
            siguiente = colocar(tablero, c, rival)
            if siguiente is not None and hay_ganador(siguiente, rival):
                
                return c

        # Se ordenan las columnas legales según su cercanía a la columna central del tablero.
        orden_preferencia = sorted(
            legales,
            key=lambda col: abs(col - (COLUMNAS // 2))
        )

        # Se intenta evitar jugadas que permitan una victoria inmediata del rival,
        # priorizando aquellas columnas donde el rival no consiga ganar en la siguiente jugada.
        jugadas_seguras: list[int] = []
        for c in orden_preferencia:
            siguiente = colocar(tablero, c, jugador)
            if siguiente is None:
                continue
            # Se comprueba si el rival puede ganar inmediatamente después de esta jugada.
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

        # Si no se encuentra ninguna jugada considerada segura, se selecciona la primera columna del orden de preferencia simple.
        return orden_preferencia[0]

# =============================================================
# -------------------- Q-LEARNING TABULAR ---------------------
# =============================================================

def epsilon_greedy(q_table: dict, estado: str, legales: list[int], epsilon: float) -> int:
    ''' Se implementa una política epsilon-greedy sobre la Q-table
    (estado → acción → valor). En esta representación se asume que:

        q_table[estado][accion] = {"Q": float, "N": int}

    En este archivo se utiliza únicamente para evaluación con epsilon=0.0
    (política greedy), dado que el entrenamiento se realiza con UCB. '''
    
    if not legales:
       
        return 0

    # Se realiza exploración eligiendo una acción aleatoria con probabilidad epsilon.
    if random.random() < epsilon:
        
        return random.choice(legales)

    # Se realiza explotación seleccionando la acción legal con mayor valor Q.
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

    # Si no se dispone de información aprendida para el estado, se elige una columna legal de forma determinista.
    return legales[0]

def ucb_action(q_table: dict, estado: str, legales: list[int], c: float = 1.4) -> int:
    ''' Se selecciona una acción aplicando la regla UCB1 sobre la Q-table,
    suponiendo que la estructura interna es:

        q_table[estado][accion] = {"Q": float, "N": int}. '''
    
    if not legales:

        return 0

    acciones_info = q_table.get(estado, {})

    # Se priorizan primero las acciones que no han sido exploradas nunca (N = 0).
    acciones_no_vistas = []
    for a in legales:
        info = acciones_info.get(a)
        n_sa = int(info.get("N", 0)) if info is not None else 0
        if n_sa == 0:
            acciones_no_vistas.append(a)

    if acciones_no_vistas:
        
        return random.choice(acciones_no_vistas)

    # Si todas las acciones han sido visitadas al menos una vez, se aplica la fórmula UCB.
    n_s = sum(int(info.get("N", 0)) for info in acciones_info.values())
    if n_s <= 0:
        # Se aplica una política greedy sobre Q como respaldo si se detecta algún problema con los contadores.
        mejor_a = legales[0]
        mejor_q = -1e18
        for a in legales:
            info = acciones_info.get(a, {"Q": 0.0})
            q_val = float(info.get("Q", 0.0))
            if q_val > mejor_q:
                mejor_q = q_val
                mejor_a = a

        return mejor_a

    mejor_accion = None
    mejor_ucb = -1e18
    for a in legales:
        info = acciones_info.get(a, {"Q": 0.0, "N": 1})
        q_val = float(info.get("Q", 0.0))
        n_sa = int(info.get("N", 1))
        bonus = c * math.sqrt(math.log(n_s) / n_sa)
        ucb_val = q_val + bonus
        if ucb_val > mejor_ucb:
            mejor_ucb = ucb_val
            mejor_accion = a

    return mejor_accion if mejor_accion is not None else legales[0]


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
    ''' Se actualiza el valor Q(s,a) siguiendo la regla estándar de Q-learning:

        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)],

    y se mantiene adicionalmente un conteo de visitas N(s,a) para facilitar el
    análisis del proceso de aprendizaje. '''

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
    # Se incrementa el número de visitas del par (estado, acción).
    q_table[estado][accion]["N"] += 1

# =============================================================
# ---------------------- ENTRENAMIENTO ------------------------
# =============================================================

def jugar_episodio_qlearning(
    q_table: dict,
    oponente: HeuristicOpponentMedium,
    alpha: float,
    gamma: float,
    c_ucb: float,
) -> tuple[int, int, int]:
    ''' Se ejecuta un episodio completo (partida) de Q-learning contra el
    oponente heurístico. El agente controlado por la Q-table adopta el rol
    indicado por agent_player, que se elige aleatoriamente entre -1 y 1.

    En este procedimiento, la política de selección de acciones del agente
    utiliza UCB en lugar de epsilon-greedy.

    Se devuelve una tupla con:
      - la recompensa final desde el punto de vista del agente
        (+1 si gana, -1 si pierde, 0 si hay empate),
      - la longitud de la partida medida en número total de jugadas,
      - el valor de agent_player (-1 o 1), para poder analizar el rol que tuvo
        el agente en la partida. '''
    
    tablero = np.zeros((FILAS, COLUMNAS), dtype=int)

    # Se decide al azar si el agente juega como -1 o como 1 en este episodio.
    agent_player = random.choice([-1, 1])
    opponent_player = -agent_player  # se deja por claridad, aunque no se use directamente

    # Se registra la trayectoria de decisiones del agente como pares (estado, acción).
    trayectoria: list[tuple[str, int]] = []
    num_movimientos = 0

    while True:
        jugador = turno_actual(tablero)

        # Se verifica si algún jugador ganó en la jugada anterior.
        if hay_ganador(tablero, -jugador):
            ganador = -jugador
            if ganador == agent_player:
                recompensa_final = 1
            else:
                recompensa_final = -1
            break

        movs = movimientos_legales(tablero)
        if not movs:
            # Se considera empate si no quedan movimientos legales.
            recompensa_final = 0
            break

        if jugador == agent_player:
            # Se selecciona la acción del agente mediante la política basada en UCB.
            estado = codificar_estado(tablero)
            accion = ucb_action(q_table, estado, movs, c=c_ucb)
            tablero_siguiente = colocar(tablero, accion, jugador)
            if tablero_siguiente is None:
                # Se penaliza con derrota si se intenta jugar en una columna llena (situación que no debería ocurrir).
                recompensa_final = -1
                break

            # Se almacena la transición (estado, acción) para la posterior actualización.
            trayectoria.append((estado, accion))
            tablero = tablero_siguiente
            num_movimientos += 1

        else:
            # Se delega la acción en el oponente heurístico cuando no juega el agente.
            accion = oponente.elegir_accion(tablero, jugador)
            tablero_siguiente = colocar(tablero, accion, jugador)
            if tablero_siguiente is None:
                # Se considera derrota del jugador que realiza la jugada ilegal; desde el punto de vista del agente se ajusta la recompensa.
                recompensa_final = 1 if agent_player == jugador else -1
                break
            tablero = tablero_siguiente
            num_movimientos += 1

    ''' Se realiza una actualización hacia atrás de los valores Q para todas las jugadas del agente.
    Se utiliza recompensa solo al final (r = 0 en pasos intermedios y r_final en el estado terminal),
    implementando una aproximación tipo "Monte Carlo con bootstrap" donde se usa el mismo r_final
    para todas las decisiones y se bootstrapea con el máximo Q del siguiente estado. '''
    for i, (estado, accion) in enumerate(trayectoria):
        if i == len(trayectoria) - 1:
            # Se actualiza la última decisión del agente considerando que después se llega a un estado terminal.
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
            # Se actualizan los estados intermedios usando recompensa 0 y bootstrap con el estado siguiente.
            siguiente_estado, _ = trayectoria[i + 1]
            # Para el estado siguiente se aproxima el conjunto de acciones legales como todas las columnas del tablero.
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
    ''' Se evalúa el agente controlado por la Q-table frente al oponente
    heurístico, utilizando siempre una política greedy (epsilon = 0) para la
    selección de acciones. El rol del agente se alterna entre -1 y 1 en cada
    partida. Se devuelven métricas agregadas de victorias, empates, derrotas
    y el winrate resultante. '''
    
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
                # Se ejecuta la política greedy (epsilon = 0) durante la evaluación.
                accion = epsilon_greedy(q_table, estado, movs, epsilon=0.0)
                tablero_siguiente = colocar(tablero, accion, jugador)
                if tablero_siguiente is None:
                    # Se considera derrota del agente si realiza una jugada ilegal durante la evaluación.
                    derrotas += 1
                    break
                tablero = tablero_siguiente
            else:
                accion = oponente.elegir_accion(tablero, jugador)
                tablero_siguiente = colocar(tablero, accion, jugador)
                if tablero_siguiente is None:
                    # Se considera victoria del agente si el oponente comete una jugada ilegal.
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
# ---------------------- GUARDADO MODELO ----------------------
# =============================================================

def guardar_modelo_simple(q_table: dict, archivo: str) -> None:
    ''' Se guarda la Q-table en un formato simplificado de la forma:

        {
          "estado": { "columna": Q_value, ... },
          ...
        }

    donde "columna" es una cadena con el índice de la columna. Se utiliza el
    mismo formato que emplea la clase policy.CetinaSalasSabogal. La Q-table
    puede almacenar las acciones como diccionarios con Q y N
    (q_table[estado][accion] = {"Q": float, "N": int}) o como valores float
    directos para compatibilidad. '''

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
    ''' Se guarda una versión completa de la tabla Q que incluye tanto el
    conteo de visitas N como los valores Q por acción. Este archivo no se
    utiliza en la policy de producción, pero resulta útil para análisis e
    inspecciones detalladas del proceso de aprendizaje. Se asume que la
    Q-table tiene la forma:

        q_table[estado][accion] = {"Q": float, "N": int}. '''
    
    salida: dict[str, dict[str, dict[str, float]]] = {}
    for estado, acciones in q_table.items():
        acciones_completas: dict[str, dict[str, float]] = {}
        for a_int, info in acciones.items():
            if isinstance(info, dict):
                n_val = int(info.get("N", 0))
                q_val = float(info.get("Q", 0.0))
            else:
                # Se mantiene compatibilidad si en alguna posición solo se almacenó el valor Q.
                n_val = 0
                q_val = float(info)
            acciones_completas[str(a_int)] = {"N": n_val, "Q": q_val}
        if acciones_completas:
            salida[estado] = acciones_completas

    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2)

    print(f"\nSe guardó q_stats completo con N y Q en '{archivo}'.")

def media_movil(lista: list[float], ventana: int = 50) -> list[float]:
    ''' Se calcula la media móvil simple sobre la lista de valores indicada,
    utilizando una ventana de tamaño configurable. Esta utilidad se emplea
    para suavizar las curvas de aprendizaje que se guardan en los archivos
    JSON de self-play. '''

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
    print(f"[RUN {run_id}] Entrenando Q-learning (UCB) contra HeuristicOpponentMedium...\n")

    # Se especifican los hiperparámetros principales del procedimiento de entrenamiento.
    episodios_totales = 12000
    alpha = 0.1
    gamma = 0.95
    c_ucb = 1.4  # Se define el parámetro de exploración para UCB.
    eval_interval = 200
    eval_partidas = 200

    # Se utiliza una Q-table que, para cada estado y acción, almacena tanto el valor Q como el número de visitas N.
    #   q_table[estado][accion] = {"Q": float, "N": int}
    q_table: dict[str, dict[int, dict[str, float]]] = {}
    oponente = HeuristicOpponentMedium()

    recompensas_ultimos: list[float] = []
    historial_curva: list[dict] = []

    # Se inicializan listas para guardar métricas de self-play en cada episodio.
    episodios_lista: list[int] = []
    recompensas_episodios: list[float] = []
    longitudes_episodios: list[int] = []
    agent_roles: list[int] = []  # -1 o 1, rol del agente en cada episodio

    for epi in range(1, episodios_totales + 1):
        r, longitud, agent_player = jugar_episodio_qlearning(
            q_table=q_table,
            oponente=oponente,
            alpha=alpha,
            gamma=gamma,
            c_ucb=c_ucb,
        )

        # Se registran las métricas de self-play para el episodio actual.
        episodios_lista.append(epi)
        recompensas_episodios.append(float(r))
        longitudes_episodios.append(int(longitud))
        agent_roles.append(int(agent_player))

        # Se mantiene una ventana móvil con las últimas 50 recompensas observadas.
        recompensas_ultimos.append(r)
        if len(recompensas_ultimos) > 50:
            recompensas_ultimos.pop(0)

        media_ultimos = sum(recompensas_ultimos) / len(recompensas_ultimos)

        if epi == 1 or epi % 50 == 0:
            print(
                f"[RUN {run_id}] Episodio {epi}/{episodios_totales} - "
                f"recompensa={r}, "
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

    # Se guarda el modelo final en el formato utilizado por la policy.
    guardar_modelo_simple(q_table, "connect4_model.json")

    # Se guarda una copia histórica del modelo final (solo valores Q).
    archivo_modelo_hist = f"modelo_qlearning_ucb_{run_id}.json"
    guardar_modelo_simple(q_table, archivo_modelo_hist)

    # Se guarda la tabla completa q_stats con N y Q por acción.
    archivo_qstats_completo = f"qstats_completo_ucb_{run_id}.json"
    guardar_qstats_completo(q_table, archivo_qstats_completo)

    # Se construye y guarda la curva de aprendizaje frente al oponente heurístico.
    curva = {
        "metadata": {
            "run_id": run_id,
            "episodios_totales": episodios_totales,
            "paso_eval": eval_interval,
            "partidas_eval": eval_partidas,
            "oponente": "HeuristicOpponentMedium",
            "c_ucb": c_ucb,
        },
        "historial": historial_curva,
    }
    archivo_curva = f"curva_vs_heuristic_medium_ucb_{run_id}.json"
    with open(archivo_curva, "w", encoding="utf-8") as f:
        json.dump(curva, f, indent=2)

    # Se calcula un resumen global de victorias, empates y derrotas desde el punto de vista del agente.
    victorias_sp = sum(1 for r in recompensas_episodios if r > 0)
    derrotas_sp = sum(1 for r in recompensas_episodios if r < 0)
    empates_sp = sum(1 for r in recompensas_episodios if r == 0)
    total_sp = len(recompensas_episodios)

    # Se calculan medias móviles para recompensas y longitudes, útiles para graficar curvas suavizadas a partir del JSON.
    recompensas_mm = media_movil(recompensas_episodios, ventana=50)
    longitudes_mm = media_movil([float(l) for l in longitudes_episodios], ventana=50)

    selfplay_data = {
        "metadata": {
            "run_id": run_id,
            "episodios_totales": episodios_totales,
            "alpha": alpha,
            "gamma": gamma,
            "oponente": "HeuristicOpponentMedium",
            "c_ucb": c_ucb,
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
    archivo_selfplay = f"selfplay_qlearning_ucb_vs_heuristic_medium_{run_id}.json"
    with open(archivo_selfplay, "w", encoding="utf-8") as f:
        json.dump(selfplay_data, f, indent=2)

    print(f"\n[RUN {run_id}] Curva de aprendizaje guardada en '{archivo_curva}'.")
    print(f"[RUN {run_id}] Estadísticas de self-play guardadas en '{archivo_selfplay}'.")
    print(f"[RUN {run_id}] Modelo histórico (solo Q) guardado en '{archivo_modelo_hist}'.")
    print(f"[RUN {run_id}] q_stats completo con N y Q guardado en '{archivo_qstats_completo}'.")
    print(f"[RUN {run_id}] Entrenamiento (UCB) contra HeuristicOpponentMedium terminado.")