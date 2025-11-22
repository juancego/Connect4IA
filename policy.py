import math
import json
import numpy as np

from connect4.policy import Policy
from connect4.connect_state import ConnectState


class MctsUcbPolicy(Policy):
    """
    Agente de Conecta 4 con:

    - MCTS + UCB1 para planear en cada jugada.
    - Tabla de valores (values.json) como PRIOR SUAVE aprendida por self-play.

    Diseño:
      * Dentro de la partida:
          Q(s,a) se estima con Monte Carlo Tree Search (MCTS):
          selección UCB1 → expansión → simulación (rollout) → backup.
      * Entre partidas:
          values.json guarda una tabla de valores V(s) aprendida en train.py.
          Esos valores se usan como "visitas virtuales" para los nodos
          cuyo estado ya se ha visto en entrenamiento.

      * Para no romper el tiempo límite de Gradescope:
          - n_simulations moderado (120).
          - Rollouts ligeros, solo con sesgo al centro.
          - Sin heurística pesada en los rollouts.
    """

    def __init__(self) -> None:
        # simulaciones de MCTS por jugada (ajustable)
        self.n_simulations: int = 120

        # constante de exploración UCB1
        self.c_ucb: float = math.sqrt(2.0)

        # RNG
        self.rng = np.random.default_rng()

        # tabla de valores aprendida
        self.value_table: dict[str, float] = {}

        # timeout opcional (Gradescope puede pasar algo aquí)
        self.timeout: float | None = None

    # ------------------------- UTIL -------------------------

    @staticmethod
    def _state_key(board: np.ndarray, player: int) -> str:
        """
        Serializa (tablero, jugador) en una clave de texto para usar en values.json.
        Debe ser consistente con la función de entrenamiento en train.py.
        """
        flat = "".join(str(int(x)) for x in board.flatten())
        return f"{player}|{flat}"

    # ------------------------- MOUNT -------------------------

    def mount(self, timeout: float | None = None) -> None:
        """
        Se llama al inicio de cada partida.

        Gradescope la llama como mount(POLICY_ACTION_TIMEOUT), así que
        hay que aceptar un parámetro opcional.

        - timeout: tiempo máximo (en segundos) que debería usar act().
                   Aquí lo guardamos por si se quiere ajustar n_simulations.
        """
        self.rng = np.random.default_rng()
        self.timeout = timeout

        # si el timeout es muy pequeño, reducimos simulaciones
        if self.timeout is not None and self.timeout < 0.5:
            self.n_simulations = 60
        else:
            self.n_simulations = 120

        # cargar values.json si existe
        try:
            with open("values.json", "r") as f:
                content = f.read().strip()
                self.value_table = json.loads(content) if content else {}
        except (FileNotFoundError, json.JSONDecodeError):
            self.value_table = {}

    # -------------------------- ACT --------------------------

    def act(self, s: np.ndarray) -> int:
        """
        Decide la columna donde jugar a partir del tablero s (6x7).
        """

        # ----- jugador actual y estado raíz -----
        num_tokens = np.count_nonzero(s)
        root_player = -1 if num_tokens % 2 == 0 else 1
        opponent = -root_player

        root_state = ConnectState(board=s, player=root_player)
        legal_actions = root_state.get_free_cols()
        if not legal_actions:
            return 0

        opp_state = ConnectState(board=s, player=opponent)

        # ================== 1) TÁCTICA INMEDIATA ==================

        # 1.a) Ganar en una jugada
        for c in legal_actions:
            if root_state.is_applicable(c):
                nxt = root_state.transition(c)
                if nxt.is_final() and nxt.get_winner() == root_player:
                    return c

        # 1.b) Bloquear victoria inmediata del rival
        for c in legal_actions:
            if opp_state.is_applicable(c):
                nxt = opp_state.transition(c)
                if nxt.is_final() and nxt.get_winner() == opponent:
                    return c

        # ================== 2) MCTS CON PRIORS ==================

        value_table = self.value_table
        c_ucb = self.c_ucb
        rng = self.rng

        def make_key(board: np.ndarray, player: int) -> str:
            return MctsUcbPolicy._state_key(board, player)

        class Node:
            __slots__ = (
                "state",
                "player",
                "parent",
                "action_from_parent",
                "children",
                "untried_actions",
                "N",
                "W",
            )

            def __init__(
                self,
                state: ConnectState,
                player: int,
                parent: "Node | None" = None,
                action_from_parent: int | None = None,
            ) -> None:
                self.state = state
                self.player = player   # jugador al turno en este nodo
                self.parent = parent
                self.action_from_parent = action_from_parent

                self.children: dict[int, "Node"] = {}
                self.untried_actions: list[int] = self.state.get_free_cols()

                # Contadores MCTS
                self.N: int = 0    # visitas
                self.W: float = 0  # suma de recompensas

                # ----- PRIOR SUAVE desde values.json -----
                key = make_key(self.state.board, self.player)
                if key in value_table:
                    prior = float(value_table[key])
                    virtual_N = 2       # pocas visitas virtuales → prior suave
                    self.N = virtual_N
                    self.W = prior * virtual_N

            def is_terminal(self) -> bool:
                return self.state.is_final()

            def is_fully_expanded(self) -> bool:
                return len(self.untried_actions) == 0

        root = Node(root_state, root_player)

        # -------------------- Selección UCB1 --------------------

        def select_child_ucb(node: Node) -> Node:
            logN = math.log(max(1, node.N))
            best_score = -math.inf
            best_children: list[Node] = []

            for child in node.children.values():
                if child.N == 0:
                    ucb = math.inf
                else:
                    Q = child.W / child.N
                    ucb = Q + c_ucb * math.sqrt(logN / child.N)

                if ucb > best_score:
                    best_score = ucb
                    best_children = [child]
                elif ucb == best_score:
                    best_children.append(child)

            return rng.choice(best_children)

        # -------------------- Rollout policy ligera --------------------

        def rollout_policy(state: ConnectState) -> int:
            """
            Política de simulación sencilla:
            - Elige entre columnas legales.
            - Sesgo hacia el centro (3,2,4,1,5,0,6).
            """
            actions = state.get_free_cols()
            if not actions:
                return 0

            pref_order = [3, 2, 4, 1, 5, 0, 6]
            ordered = [a for a in pref_order if a in actions]
            if not ordered:
                ordered = actions
            return int(rng.choice(ordered))

        # -------------------- Simulación --------------------

        def simulate(node: Node) -> int:
            sim_state = node.state
            while not sim_state.is_final():
                a = rollout_policy(sim_state)
                if not sim_state.is_applicable(a):
                    acts = sim_state.get_free_cols()
                    if not acts:
                        break
                    a = acts[0]
                sim_state = sim_state.transition(a)

            winner = sim_state.get_winner()
            if winner == 0:
                return 0
            return 1 if winner == root_player else -1

        # -------------------- Bucle principal de MCTS --------------------

        for _ in range(self.n_simulations):
            node = root

            # 1) Selección
            while (
                not node.is_terminal()
                and node.is_fully_expanded()
                and node.children
            ):
                node = select_child_ucb(node)

            # 2) Expansión
            if not node.is_terminal() and node.untried_actions:
                a = node.untried_actions.pop()
                if node.state.is_applicable(a):
                    ns = node.state.transition(a)
                    child = Node(
                        state=ns,
                        player=ns.player,
                        parent=node,
                        action_from_parent=a,
                    )
                    node.children[a] = child
                    node = child

            # 3) Simulación
            reward = simulate(node)

            # 4) Backpropagation
            while node is not None:
                node.N += 1
                if node.player == root_player:
                    node.W += reward
                else:
                    node.W -= reward
                node = node.parent

        # ================== 3) ELEGIR ACCIÓN FINAL ==================

        if not root.children:
            # fallback por seguridad
            if 3 in legal_actions:
                return 3
            return legal_actions[0]

        best_N = -1
        best_actions: list[int] = []

        for a, child in root.children.items():
            if child.N > best_N:
                best_N = child.N
                best_actions = [a]
            elif child.N == best_N:
                best_actions.append(a)

        # preferimos la columna central si está entre las mejores
        if 3 in best_actions:
            return 3

        return int(best_actions[0])