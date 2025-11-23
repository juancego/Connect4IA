import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from connect4.connect_state import ConnectState
from connect4.policy import Policy
from connect4.utils import find_importable_classes



def state_key(board, player: int) -> str:
    """Serializa (tablero, jugador) en un string para usar como clave en values.json."""
    flat = "".join(str(int(x)) for x in board.flatten())
    return f"{player}|{flat}"


def load_value_table(path: str = "values.json") -> dict[str, float]:
    """Carga la tabla de valores si existe, si no, devuelve un diccionario vacío."""
    try:
        with open(path, "r") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class ModelEvaluator:
    """Evaluador para modelos de Connect4 entrenados contra agentes de groups/"""
    
    def __init__(self, learner_name: str = "My Agent", model_path: str = "values.json"):
        self.learner_name = learner_name
        self.model_path = model_path
        self.value_table = {}
        self.participants = {}
        self.opponent_names = []
        
        # Métricas de evaluación
        self.results_by_opponent = defaultdict(lambda: {'wins': 0, 'draws': 0, 'losses': 0, 'steps': []})
        self.all_results = []
        self.column_usage = defaultdict(int)
        self.win_positions = []
        
    def load_model(self):
        """Carga el modelo entrenado y los agentes disponibles"""
        # Cargar tabla de valores
        self.value_table = load_value_table(self.model_path)
        if not self.value_table:
            print(f"Advertencia: {self.model_path} está vacío o no existe")
            print("Se evaluará sin valores previos")
        else:
            print(f"Modelo cargado: {self.model_path}")
            print(f"Estados en tabla: {len(self.value_table):,}")
        
        # Cargar agentes disponibles
        self.participants = find_importable_classes("groups", Policy)
        
        if not self.participants:
            print("Error: No se encontraron agentes en groups/")
            return False
        
        print(f"\n Agentes disponibles en groups/:")
        for name in self.participants.keys():
            marker = " *" if name == self.learner_name else ""
            print(f"   • {name}{marker}")
        
        # Verificar que existe el agente que aprende
        if self.learner_name not in self.participants:
            print(f"\n Error: No se encontró '{self.learner_name}' en groups/")
            return False
        
        # Definir oponentes (todos menos el aprendiz)
        self.opponent_names = [name for name in self.participants.keys() 
                               if name != self.learner_name]
        
        if not self.opponent_names:
            print(f"\n Error: No hay oponentes distintos de '{self.learner_name}'")
            return False
        
        print(f"\n Agente a evaluar: {self.learner_name}")
        print(f" Oponentes disponibles ({len(self.opponent_names)}):")
        for name in self.opponent_names:
            print(f"   • {name}")
        
        return True
    
    def evaluate_against_all(self, games_per_opponent: int = 100, verbose: bool = True):
        """
        Evalúa el agente contra todos los oponentes disponibles
        
        Args:
            games_per_opponent: Número de partidas contra cada oponente
            verbose: Si True, muestra progreso detallado
        """
        rng = np.random.default_rng()
        
        LearningPolicy = self.participants[self.learner_name]
        total_games = len(self.opponent_names) * games_per_opponent
        
        print(f"\n{'='*80}")
        print(f" EVALUANDO {self.learner_name}")
        print(f"{'='*80}")
        print(f"Partidas por oponente: {games_per_opponent}")
        print(f"Total de partidas: {total_games}")
        print(f"{'='*80}\n")
        
        game_counter = 0
        
        for opp_name in self.opponent_names:
            OpponentPolicy = self.participants[opp_name]
            
            if verbose:
                print(f"\n Contra {opp_name} ({games_per_opponent} partidas):")
                print("─" * 70)
            
            for game_num in range(games_per_opponent):
                game_counter += 1
                
                # Crear instancias de los agentes
                learner: Policy = LearningPolicy()
                learner.mount()
                
                opponent: Policy = OpponentPolicy()
                opponent.mount()
                
                # Jugar partida
                state = ConnectState()
                steps = 0
                first_move = None
                
                while not state.is_final():
                    steps += 1
                    
                    if state.player == -1:
                        # Turno del agente evaluado
                        action = learner.act(state.board)
                        if steps == 1:
                            first_move = int(action)
                            self.column_usage[first_move] += 1
                    else:
                        # Turno del oponente
                        action = opponent.act(state.board)
                    
                    state = state.transition(int(action))
                
                winner = state.get_winner()
                
                # Registrar resultado
                if winner == -1:
                    result = "WIN"
                    self.results_by_opponent[opp_name]['wins'] += 1
                    self.win_positions.append(steps)
                elif winner == 0:
                    result = "DRAW"
                    self.results_by_opponent[opp_name]['draws'] += 1
                else:
                    result = "LOSS"
                    self.results_by_opponent[opp_name]['losses'] += 1
                
                self.results_by_opponent[opp_name]['steps'].append(steps)
                
                self.all_results.append({
                    'game': game_counter,
                    'opponent': opp_name,
                    'result': result,
                    'steps': steps,
                    'first_move': first_move
                })
                
                # Mostrar progreso cada 20 partidas
                if verbose and (game_num + 1) % 20 == 0:
                    current_wins = self.results_by_opponent[opp_name]['wins']
                    current_wr = current_wins / (game_num + 1)
                    print(f"  Partida {game_num + 1:3d}/{games_per_opponent} | "
                          f"Win Rate: {current_wr:.2%} | "
                          f"W:{self.results_by_opponent[opp_name]['wins']} "
                          f"D:{self.results_by_opponent[opp_name]['draws']} "
                          f"L:{self.results_by_opponent[opp_name]['losses']}")
            
            # Resumen contra este oponente
            opp_stats = self.results_by_opponent[opp_name]
            opp_total = games_per_opponent
            opp_wr = opp_stats['wins'] / opp_total
            
            if verbose:
                print(f"  {'─'*70}")
                print(f"  Resumen vs {opp_name}: Win Rate = {opp_wr:.2%} "
                      f"({opp_stats['wins']}W {opp_stats['draws']}D {opp_stats['losses']}L)")
        
        # Calcular estadísticas globales
        total_wins = sum(stats['wins'] for stats in self.results_by_opponent.values())
        total_draws = sum(stats['draws'] for stats in self.results_by_opponent.values())
        total_losses = sum(stats['losses'] for stats in self.results_by_opponent.values())
        
        all_steps = [r['steps'] for r in self.all_results]
        avg_steps_all = np.mean(all_steps) if all_steps else 0
        avg_steps_wins = np.mean(self.win_positions) if self.win_positions else 0
        
        global_stats = {
            'total_games': total_games,
            'wins': total_wins,
            'draws': total_draws,
            'losses': total_losses,
            'win_rate': total_wins / total_games,
            'draw_rate': total_draws / total_games,
            'loss_rate': total_losses / total_games,
            'avg_steps_all': avg_steps_all,
            'avg_steps_wins': avg_steps_wins
        }
        
        return global_stats
    
    def print_detailed_report(self, global_stats):
        """Imprime reporte detallado de evaluación"""
        print("\n" + "="*80)
        print(" REPORTE DETALLADO DE EVALUACIÓN")
        print("="*80)
        
        # Resultados globales
        print(f"\n RESULTADOS GLOBALES ({global_stats['total_games']} partidas):")
        print(f"  • Victorias:  {global_stats['wins']:4d} ({global_stats['win_rate']:.2%})  " + 
              self._get_performance_emoji(global_stats['win_rate']))
        print(f"  • Empates:    {global_stats['draws']:4d} ({global_stats['draw_rate']:.2%})")
        print(f"  • Derrotas:   {global_stats['losses']:4d} ({global_stats['loss_rate']:.2%})")
        
        # Eficiencia
        print(f"\n EFICIENCIA:")
        print(f"  • Pasos promedio (todas):     {global_stats['avg_steps_all']:.2f}")
        print(f"  • Pasos promedio (victorias): {global_stats['avg_steps_wins']:.2f}")
        
        if global_stats['avg_steps_wins'] > 0:
            if global_stats['avg_steps_wins'] < 15:
                efficiency = "Muy eficiente "
            elif global_stats['avg_steps_wins'] < 20:
                efficiency = "Eficiente "
            elif global_stats['avg_steps_wins'] < 25:
                efficiency = "Aceptable "
            else:
                efficiency = "Lento "
            print(f"  • Evaluación: {efficiency}")
        
        # Resultados por oponente (ordenados por win rate)
        print(f"\n RENDIMIENTO POR OPONENTE:")
        print(f"  {'Oponente':<25} {'W':>4} {'D':>4} {'L':>4}  {'Win Rate':>9}  {'Evaluación'}")
        print(f"  {'-'*75}")
        
        # Ordenar por win rate descendente
        opp_sorted = sorted(
            self.results_by_opponent.items(),
            key=lambda x: x[1]['wins'] / (x[1]['wins'] + x[1]['draws'] + x[1]['losses']),
            reverse=True
        )
        
        for opp_name, stats in opp_sorted:
            total = stats['wins'] + stats['draws'] + stats['losses']
            wr = stats['wins'] / total if total > 0 else 0
            emoji = self._get_performance_emoji(wr, compact=True)
            print(f"  {opp_name:<25} {stats['wins']:>4} {stats['draws']:>4} {stats['losses']:>4}  "
                  f"{wr:>8.1%}  {emoji}")
        
        # Análisis de primeras jugadas
        if self.column_usage:
            print(f"\n PREFERENCIA DE PRIMERA JUGADA:")
            sorted_cols = sorted(self.column_usage.items(), key=lambda x: x[1], reverse=True)
            total_first_moves = sum(self.column_usage.values())
            
            for i, (col, count) in enumerate(sorted_cols[:5], 1):
                pct = (count / total_first_moves) * 100
                bar = "|" * int(pct / 2)
                star = " *" if col == 3 else ""
                print(f"  {i}. Columna {col}: {count:4d} ({pct:5.1f}%) {bar}{star}")
            
            center_usage = self.column_usage.get(3, 0)
            center_pct = (center_usage / total_first_moves) * 100
            if center_pct > 30:
                print(f"   Buena estrategia: Prefiere centro ({center_pct:.1f}%)")
            elif center_pct > 15:
                print(f"    Estrategia aceptable: Centro usado {center_pct:.1f}%")
            else:
                print(f"   Estrategia débil: Centro poco usado ({center_pct:.1f}%)")
        
        # Distribución de longitud de victorias
        if self.win_positions:
            print(f"\n ANÁLISIS DE VICTORIAS:")
            print(f"  • Victoria más rápida:  {min(self.win_positions)} pasos")
            print(f"  • Victoria más lenta:   {max(self.win_positions)} pasos")
            print(f"  • Mediana:              {np.median(self.win_positions):.0f} pasos")
            
            # Categorizar victorias
            fast_wins = sum(1 for s in self.win_positions if s <= 15)
            medium_wins = sum(1 for s in self.win_positions if 15 < s <= 25)
            slow_wins = sum(1 for s in self.win_positions if s > 25)
            
            print(f"\n  Distribución por velocidad:")
            print(f"    • Rápidas (≤15):   {fast_wins:4d} ({fast_wins/len(self.win_positions):.1%})")
            print(f"    • Medias (16-25):  {medium_wins:4d} ({medium_wins/len(self.win_positions):.1%})")
            print(f"    • Lentas (>25):    {slow_wins:4d} ({slow_wins/len(self.win_positions):.1%})")
        
        # Evaluación general
        print(f"\n EVALUACIÓN GENERAL:")
        self._print_overall_assessment(global_stats)
        
        print("="*80 + "\n")
    
    def _get_performance_emoji(self, win_rate, compact=False):
        """Retorna emoji según el win rate"""
        if compact:
            if win_rate >= 0.80:
                return "E"
            elif win_rate >= 0.70:
                return "MB"
            elif win_rate >= 0.60:
                return "B"
            elif win_rate >= 0.50:
                return "A"
            else:
                return "I"
        else:
            if win_rate >= 0.80:
                return " Excelente"
            elif win_rate >= 0.70:
                return " Muy bueno"
            elif win_rate >= 0.60:
                return " Bueno"
            elif win_rate >= 0.50:
                return "  Aceptable"
            else:
                return " Insuficiente"
    
    def _print_overall_assessment(self, stats):
        """Imprime evaluación general del modelo"""
        issues = []
        strengths = []
        
        # Evaluar win rate global
        if stats['win_rate'] >= 0.70:
            strengths.append("Domina contra la mayoría de oponentes")
        elif stats['win_rate'] >= 0.55:
            strengths.append("Supera nivel promedio")
        else:
            issues.append("Win rate bajo - necesita más entrenamiento")
        
        # Evaluar consistencia entre oponentes
        win_rates = []
        for stats_opp in self.results_by_opponent.values():
            total = stats_opp['wins'] + stats_opp['draws'] + stats_opp['losses']
            if total > 0:
                win_rates.append(stats_opp['wins'] / total)
        
        if win_rates:
            wr_std = np.std(win_rates)
            if wr_std < 0.15:
                strengths.append("Rendimiento consistente contra diferentes oponentes")
            elif wr_std > 0.25:
                issues.append("Rendimiento muy variable - débil contra algunos oponentes")
        
        # Evaluar eficiencia
        if stats['avg_steps_wins'] > 0 and stats['avg_steps_wins'] < 18:
            strengths.append("Estrategia eficiente (gana rápido)")
        elif stats['avg_steps_wins'] > 25:
            issues.append("Victorias muy lentas - juega muy defensivo")
        
        # Evaluar empates
        if stats['draw_rate'] > 0.25:
            issues.append(f"Muchos empates ({stats['draw_rate']:.1%}) - muy conservador")
        
        # Evaluar primera jugada
        if self.column_usage:
            center_pct = (self.column_usage.get(3, 0) / sum(self.column_usage.values())) * 100
            if center_pct > 30:
                strengths.append("Buena preferencia por columna central")
            elif center_pct < 15:
                issues.append("No prioriza columna central")
        
        # Imprimir fortalezas
        if strengths:
            print("   Fortalezas:")
            for s in strengths:
                print(f"     • {s}")
        
        # Imprimir áreas de mejora
        if issues:
            print("    Áreas de mejora:")
            for i in issues:
                print(f"     • {i}")
        
        # Recomendación final
        if stats['win_rate'] >= 0.70 and not issues:
            print(f"\n   El modelo está listo para competir")
        elif stats['win_rate'] >= 0.55:
            print(f"\n   El modelo es funcional pero puede mejorar")
        else:
            print(f"\n   Recomendación: Re-entrenar con más episodios")
    
    def plot_results(self):
        """Genera gráficas completas de evaluación"""
        if not self.all_results:
            print("  No hay resultados para graficar")
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Win Rate por oponente
        ax1 = plt.subplot(2, 3, 1)
        opp_names = []
        win_rates = []
        colors_list = []
        
        for opp_name, stats in sorted(self.results_by_opponent.items()):
            total = stats['wins'] + stats['draws'] + stats['losses']
            wr = stats['wins'] / total if total > 0 else 0
            opp_names.append(opp_name[:15])  # Truncar nombres largos
            win_rates.append(wr)
            
            # Color según rendimiento
            if wr >= 0.7:
                colors_list.append('#2ecc71')
            elif wr >= 0.5:
                colors_list.append('#f39c12')
            else:
                colors_list.append('#e74c3c')
        
        bars = ax1.barh(opp_names, win_rates, color=colors_list, edgecolor='black', linewidth=1.5)
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
        ax1.set_title(' Win Rate por Oponente', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Win Rate')
        ax1.set_xlim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Distribución global de resultados
        ax2 = plt.subplot(2, 3, 2)
        results_count = {'WIN': 0, 'DRAW': 0, 'LOSS': 0}
        for r in self.all_results:
            results_count[r['result']] += 1
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        sizes = [results_count['WIN'], results_count['DRAW'], results_count['LOSS']]
        labels = [f"Wins\n{results_count['WIN']}\n({results_count['WIN']/len(self.all_results):.1%})",
                  f"Draws\n{results_count['DRAW']}\n({results_count['DRAW']/len(self.all_results):.1%})",
                  f"Losses\n{results_count['LOSS']}\n({results_count['LOSS']/len(self.all_results):.1%})"]
        
        ax2.pie(sizes, labels=labels, colors=colors, startangle=90,
                textprops={'fontsize': 11, 'weight': 'bold'})
        ax2.set_title(' Distribución Global', fontsize=13, fontweight='bold')
        
        # 3. Longitud de partidas
        ax3 = plt.subplot(2, 3, 3)
        steps_all = [r['steps'] for r in self.all_results]
        ax3.hist(steps_all, bins=25, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(steps_all), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(steps_all):.1f}')
        ax3.set_title(' Longitud de Partidas', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Pasos')
        ax3.set_ylabel('Frecuencia')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Primera jugada preferida
        ax4 = plt.subplot(2, 3, 4)
        columns = list(range(7))
        usage = [self.column_usage.get(c, 0) for c in columns]
        colors_bar = ['gold' if c == 3 else 'steelblue' for c in columns]
        bars = ax4.bar(columns, usage, color=colors_bar, edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax4.set_title(' Primera Jugada', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Columna (3 = Centro)')
        ax4.set_ylabel('Veces Seleccionada')
        ax4.set_xticks(columns)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Win rate acumulado
        ax5 = plt.subplot(2, 3, 5)
        cumulative_wr = []
        wins_so_far = 0
        
        for i, r in enumerate(self.all_results, 1):
            if r['result'] == 'WIN':
                wins_so_far += 1
            cumulative_wr.append(wins_so_far / i)
        
        ax5.plot(range(1, len(cumulative_wr) + 1), cumulative_wr, linewidth=2, color='darkblue')
        ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Baseline (50%)')
        ax5.set_title(' Win Rate Acumulado', fontsize=13, fontweight='bold')
        ax5.set_xlabel('Número de Partida')
        ax5.set_ylabel('Win Rate')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1)
        
        # 6. Victorias por longitud
        ax6 = plt.subplot(2, 3, 6)
        if self.win_positions:
            ax6.hist(self.win_positions, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
            ax6.axvline(np.mean(self.win_positions), color='darkgreen', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(self.win_positions):.1f}')
            ax6.set_title(' Longitud de Victorias', fontsize=13, fontweight='bold')
            ax6.set_xlabel('Pasos para Ganar')
            ax6.set_ylabel('Frecuencia')
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout(pad=3.0)
        plt.savefig('evaluation_results.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        print(" Gráficas guardadas en: evaluation_results.png")
    
    def save_detailed_results(self, filename: str = "evaluation_detailed.json"):
        """Guarda resultados detallados en JSON"""
        data = {
            'learner': self.learner_name,
            'results_by_opponent': {
                name: {
                    'wins': stats['wins'],
                    'draws': stats['draws'],
                    'losses': stats['losses'],
                    'avg_steps': np.mean(stats['steps']) if stats['steps'] else 0
                }
                for name, stats in self.results_by_opponent.items()
            },
            'all_games': self.all_results,
            'column_usage': dict(self.column_usage)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Resultados detallados guardados en: {filename}")

# SCRIPT PRINCIPAL DE EVALUACIÓN

if __name__ == "__main__":
    import sys
    
    # Configuración
    LEARNER_NAME = "My Agent"    # Nombre del agente en groups/
    MODEL_PATH = "values.json"   # Ruta del modelo entrenado
    GAMES_PER_OPPONENT = 100     # Partidas contra cada oponente
    VERBOSE = True               # Mostrar progreso
    
    print("="*80)
    print("EVALUADOR DE MODELO CONNECT4")
    print("="*80)
    
    # Crear evaluador
    evaluator = ModelEvaluator(learner_name=LEARNER_NAME, model_path=MODEL_PATH)
    
    # Cargar modelo y agentes
    if not evaluator.load_model():
        sys.exit(1)
    
    # Evaluar contra todos los oponentes
    global_stats = evaluator.evaluate_against_all(
        games_per_opponent=GAMES_PER_OPPONENT,
        verbose=VERBOSE
    )
    
    # Mostrar reporte detallado
    evaluator.print_detailed_report(global_stats)
    
    # Generar gráficas
    print("Generando visualizaciones...")
    evaluator.plot_results()
    
    # Guardar resultados detallados
    evaluator.save_detailed_results()
    
    print("\n Evaluación completada exitosamente!")