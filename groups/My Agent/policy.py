import math
import json
import numpy as np

from connect4.policy import Policy
from connect4.connect_state import ConnectState


class MctsUcbPolicy(Policy):
    """
    Agente de Conecta 4 con:
    - MCTS + UCB1 (aprendizaje dentro de las simulaciones)
    - Tabla de valores (values.json) aprendida por self-play

    Aprendizaje:
      * Dentro de la partida:
          Q(s,a) se estima como promedio de retornos (Monte Carlo)
          y se actualiza en cada simulación (W, N).
      * Entre partidas:
          Se carga values.json y se usan esos valores como priors
          en los nodos del árbol (aprendizaje acumulado por self-play).
    """

    def __init__(self) -> None:
        # simulaciones por llamada a act
        self.n_simulations = 600
        # constante de exploración de UCB1
        self.c_ucb = math.sqrt(2.0)
        # RNG
        self.rng = np.random.default_rng()
        # tabla de valores aprendida
        self.value_table: dict[str, float] = {}

    # ------------------------- UTIL -------------------------

    @staticmethod
    def _state_key(board: np.ndarray, player: int) -> str:
        """Serializa tablero + jugador en una clave de texto para JSON."""
        return f"{player}|" + "".join(str(int(x)) for x in board.flatten())

    # ------------------------- MOUNT -------------------------

    def mount(self) -> None:
        """Se llama al inicio de cada partida. Carga values.json si existe."""
        self.rng = np.random.default_rng()
        try:
            with open("values.json", "r") as f:
                self.value_table = json.load(f)
        except FileNotFoundError:
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

        # ================== 1) TÁCTICA BÁSICA ==================

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

        # ================== 2) NODO MCTS ==================

        # Alias locales para que Node pueda usarlos (aquí está el arreglo al bug)
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

            def __init__(self, state: ConnectState, player: int,
                         parent: "Node | None" = None,
                         action_from_parent: int | None = None) -> None:
                self.state = state
                self.player = player
                self.parent = parent
                self.action_from_parent = action_from_parent

                self.children: dict[int, Node] = {}
                self.untried_actions: list[int] = self.state.get_free_cols()

                # Contadores MCTS
                self.N = 0
                self.W = 0.0

                # ----- PRIOR desde values.json -----
                key = make_key(self.state.board, self.player)
                if key in value_table:
                    prior = value_table[key]
                    # Se interpreta como valor esperado aproximado:
                    # inicializamos con algunas "visitas virtuales"
                    self.N = 5
                    self.W = prior * 5

            def is_terminal(self) -> bool:
                return self.state.is_final()

            def is_fully_expanded(self) -> bool:
                return len(self.untried_actions) == 0

        root = Node(root_state, root_player)

        # ================== 3) UCB1 ==================

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

        # ================== 4) ROLLOUT ==================

        def rollout_policy(state: ConnectState) -> int:
            actions = state.get_free_cols()
            if not actions:
                return 0
            # ligero sesgo al centro
            pref = [3, 2, 4, 1, 5, 0, 6]
            ordered = [a for a in pref if a in actions]
            if not ordered:
                ordered = actions
            return rng.choice(ordered)

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

            w = sim_state.get_winner()
            if w == 0:
                return 0
            return 1 if w == root_player else -1

        # ================== 5) LOOP MCTS ==================

        for _ in range(self.n_simulations):
            node = root

            # Selección
            while (not node.is_terminal()) and node.is_fully_expanded() and node.children:
                node = select_child_ucb(node)

            # Expansión
            if (not node.is_terminal()) and node.untried_actions:
                a = node.untried_actions.pop()
                if node.state.is_applicable(a):
                    ns = node.state.transition(a)
                    child = Node(ns, ns.player, parent=node, action_from_parent=a)
                    node.children[a] = child
                    node = child

            # Simulación
            reward = simulate(node)

            # Backpropagation
            while node is not None:
                node.N += 1
                if node.player == root_player:
                    node.W += reward
                else:
                    node.W -= reward
                node = node.parent

        # ================== 6) ELEGIR ACCIÓN FINAL ==================

        if not root.children:
            # Si el árbol no se expandió, elegimos algo legal (preferencia centro).
            if 3 in legal_actions:
                return 3
            return legal_actions[0]

        bestN = -1
        best_actions: list[int] = []
        for a, child in root.children.items():
            if child.N > bestN:
                bestN = child.N
                best_actions = [a]
            elif child.N == bestN:
                best_actions.append(a)

        if 3 in best_actions:
            return 3
        return best_actions[0]