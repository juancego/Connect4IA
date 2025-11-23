import os
import json
import numpy as np

from connect4.policy import Policy

# Se recorre cada columna del tablero y se verifica cuáles todavía permiten colocar
# una ficha. Para eso, se observa la casilla de la fila superior (fila 0) en cada
# columna. Si en esa posición todavía hay un cero, significa que en esa columna
# aún cabe al menos una ficha más en alguna fila inferior, así que se considera
# una columna válida para jugar.
def columnas_validas(tablero: np.ndarray, num_columnas: int) -> list[int]:
    columnas_libres: list[int] = []
    for indice_columna in range(num_columnas):
        if tablero[0, indice_columna] == 0:
            columnas_libres.append(indice_columna)
    return columnas_libres

# Se simula lo que ocurre si se deja caer una ficha en una columna concreta.
# La idea es buscar desde la parte de abajo del tablero hacia arriba la primera
# casilla libre en esa columna. Si se encuentra una posición vacía, se crea una
# copia del tablero y se coloca allí la ficha del jugador indicado. Si en cambio
# la columna ya está completamente llena, no se puede hacer ninguna jugada en
# esa columna y se devuelve None.
def simular_caida(tablero: np.ndarray, columna: int, jugador: int, num_filas: int) -> np.ndarray | None:
    fila_objetivo = -1
    fila = num_filas - 1
    while fila >= 0:
        if tablero[fila, columna] == 0:
            fila_objetivo = fila
            break
        fila -= 1

    # Si no se encontró ninguna fila libre, la columna está llena.
    if fila_objetivo < 0:
        return None

    nuevo_tablero = tablero.copy()
    nuevo_tablero[fila_objetivo, columna] = jugador
    return nuevo_tablero

# Se verifica si un jugador ha conseguido colocar cuatro fichas consecutivas
# en alguna dirección. Se revisan todas las posibles alineaciones horizontales,
# verticales y diagonales. Si se encuentra una secuencia de cuatro posiciones
# contiguas que pertenecen al mismo jugador, se considera que hay un conecta cuatro.
def cuatro_en_linea(tablero: np.ndarray, jugador: int, num_filas: int, num_columnas: int) -> bool:
    # Comprobación horizontal: se fija una fila y se miran grupos de cuatro celdas contiguas.
    for f in range(num_filas):
        for c in range(num_columnas - 3):
            if np.all(tablero[f, c:c + 4] == jugador):
                return True

    # Comprobación vertical: se fija una columna y se revisan segmentos de cuatro filas seguidas.
    for f in range(num_filas - 3):
        for c in range(num_columnas):
            if np.all(tablero[f:f + 4, c] == jugador):
                return True

    # Comprobación diagonal principal: se revisan diagonales que avanzan hacia abajo y a la derecha.
    for f in range(num_filas - 3):
        for c in range(num_columnas - 3):
            valido = True
            for d in range(4):
                if tablero[f + d, c + d] != jugador:
                    valido = False
                    break
            if valido:
                return True

    # Comprobación diagonal secundaria: se revisan diagonales que avanzan hacia abajo y a la izquierda.
    for f in range(num_filas - 3):
        for c in range(num_columnas - 3):
            valido = True
            for d in range(4):
                if tablero[f + 3 - d, c + d] != jugador:
                    valido = False
                    break
            if valido:
                return True

    # Si no se encuentra ninguna alineación de cuatro fichas, no hay conecta cuatro.
    return False

# Se determina de quién es el turno observando cuántas fichas hay colocadas ya
# en el tablero. Como el juego empieza con el jugador -1, se utiliza la paridad
# del número de fichas: si la cantidad de fichas es par, le corresponde mover a -1;
# si es impar, le corresponde mover a 1.
def turno_actual(tablero: np.ndarray) -> int:
    cantidad = int(np.count_nonzero(tablero))
    return -1 if cantidad % 2 == 0 else 1

# Se construye una representación textual del estado para poder usarlo como clave
# dentro de la Q-table. La clave combina el jugador al que le toca mover con el
# contenido completo del tablero aplanado en una sola cadena. De esta forma, el
# mismo tablero con otro jugador en turno se considera un estado distinto.
def codificar_posicion(tablero: np.ndarray) -> str:
    jugador = turno_actual(tablero)
    plano = tablero.flatten()
    representacion = "".join(str(int(celda)) for celda in plano)
    return f"{jugador}|" + representacion
class CetinaSalasSabogal(Policy):
    # Se definen las dimensiones estándar del tablero de Conecta 4 y el nombre del
    # archivo donde se almacena la Q-table entrenada.
    FILAS: int = 6
    COLUMNAS: int = 7
    ARCHIVO_MODELO: str = "connect4_model.json"

    def __init__(self) -> None:
        # En este diccionario se guarda la tabla Q una vez se ha cargado desde disco.
        # La estructura es: clave de estado (string) -> diccionario de acción a valor Q.
        self._q: dict[str, dict[int, float]] = {}

    # Se ejecuta antes de que empiece la partida. El objetivo principal es intentar
    # cargar desde el archivo JSON la Q-table que se haya generado durante el entrenamiento.
    # Si por cualquier motivo el archivo no existe o no es válido, se inicializa
    # una tabla vacía y el agente se comporta solo con su lógica táctica y heurística.
    def mount(self, time_out: float | None = None) -> None:
        ruta = self.ARCHIVO_MODELO

        # Si el archivo no está presente, no se intenta leer nada y se deja la Q-table vacía.
        if not os.path.exists(ruta):
            self._q = {}
            return

        try:
            with open(ruta, "r", encoding="utf-8") as archivo:
                contenido = archivo.read().strip()
                # Si el archivo está vacío, no se realiza ningún procesamiento adicional.
                if not contenido:
                    self._q = {}
                    return
                datos = json.loads(contenido)

            # Se comprueba que el contenido del JSON tenga la forma de un diccionario.
            if not isinstance(datos, dict):
                self._q = {}
                return

            tabla_q: dict[str, dict[int, float]] = {}

            # Se recorre cada estado almacenado y se procesan sus acciones asociadas,
            # convirtiendo las claves de texto en enteros y los valores a flotantes.
            for clave_estado, acciones in datos.items():
                if not isinstance(acciones, dict):
                    continue

                acciones_procesadas: dict[int, float] = {}
                for accion_txt, valor in acciones.items():
                    try:
                        accion_int = int(accion_txt)
                        valor_float = float(valor)
                    except (TypeError, ValueError):
                        # Si alguna entrada no se puede interpretar correctamente,
                        # se ignora esa acción concreta y se continúa con el resto.
                        continue
                    acciones_procesadas[accion_int] = valor_float

                # Solo se guarda el estado si al menos una acción se pudo interpretar bien.
                if acciones_procesadas:
                    tabla_q[str(clave_estado)] = acciones_procesadas

            self._q = tabla_q

        except Exception:
            # Si ocurre cualquier error durante la lectura o el parseo del archivo,
            # se prefiere continuar sin Q-table antes que detener el funcionamiento del agente.
            self._q = {}

    # A partir de un tablero dado, se decide en qué columna se coloca la ficha.
    # Se siguen varios pasos: primero se busca una victoria inmediata, luego se
    # intenta bloquear una posible victoria del rival, después se consulta la Q-table
    # y, si no hay información suficiente, se recurre a una heurística basada en
    # la estructura del tablero (prioridad por las columnas centrales).
    def act(self, s: np.ndarray) -> int:
        # Se crea una copia del tablero de entrada para evitar modificar el original.
        tablero = np.array(s, dtype=int, copy=True)

        jugador = turno_actual(tablero)
        rival = -jugador

        disponibles = columnas_validas(tablero, self.COLUMNAS)
        # Si no queda ninguna columna disponible, se devuelve una columna por defecto.
        if not disponibles:
            return 0

        # Primero se comprueba si existe alguna columna disponible en la que,
        # tras colocar una ficha, el jugador actual consiga un conecta cuatro inmediato.
        for col in disponibles:
            sim = simular_caida(tablero, col, jugador, self.FILAS)
            if sim is not None and cuatro_en_linea(sim, jugador, self.FILAS, self.COLUMNAS):
                return int(col)

        # A continuación se analiza si el rival podría ganar en el siguiente movimiento
        # colocando una ficha en alguna de las columnas disponibles. Si se detecta
        # una jugada de ese tipo, se prioriza colocar la ficha en esa columna para bloquear.
        for col in disponibles:
            sim = simular_caida(tablero, col, rival, self.FILAS)
            if sim is not None and cuatro_en_linea(sim, rival, self.FILAS, self.COLUMNAS):
                return int(col)

        # Si se dispone de una Q-table para el estado actual, se utiliza esa información
        # para escoger la acción con el valor Q más alto entre las columnas legales.
        if self._q:
            clave = codificar_posicion(tablero)
            acciones_estado = self._q.get(clave)
            if acciones_estado:
                mejor = None
                mejor_q = -float("inf")
                for accion, qvalor in acciones_estado.items():
                    if accion in disponibles and qvalor > mejor_q:
                        mejor_q = qvalor
                        mejor = accion
                # Si se encuentra una acción con buen valor Q, se devuelve esa columna.
                if mejor is not None:
                    return int(mejor)

        # Si la Q-table no aporta información útil para este estado, se recurre a
        # una heurística fija. Esta heurística da prioridad a la columna central
        # y luego a las columnas que están más cerca del centro, ya que suelen ser
        # posiciones más flexibles y potentes en Conecta 4.
        prioridad = (3, 2, 4, 1, 5, 0, 6)
        for col in prioridad:
            if col in disponibles:
                return int(col)

        # Si por alguna razón no se ha elegido nada antes, se selecciona simplemente
        # la primera columna disponible como opción de reserva.
        return int(disponibles[0])