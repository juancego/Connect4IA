import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple

from connect4.connect_state import ConnectState
from connect4.policy import Policy
from connect4.utils import find_importable_classes


def state_key(board, player: int) -> str:
    """Serializa (tablero, jugador) en un string"""
    flat = "".join(str(int(x)) for x in board.flatten())
    return f"{player}|{flat}"


def load_value_table(path: str = "values.json") -> dict[str, float]:
    """Carga la tabla de valores"""
    try:
        with open(path, "r") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class PolicyHeatmapAnalyzer:
    """Analizador de mapas de calor para políticas de Connect4"""
    
    def __init__(self, learner_name: str = "My Agent", model_path: str = "values.json"):
        self.learner_name = learner_name
        self.model_path = model_path
        self.value_table = {}
        self.participants = {}
        
        # Estructuras para análisis
        self.column_win_rates = defaultdict(lambda: {'wins': 0, 'total': 0})
        self.opening_move_analysis = defaultdict(lambda: {'wins': 0, 'draws': 0, 'losses': 0})
        self.position_heatmap = np.zeros((6, 7))  # Para acumular valores
        self.position_visit_count = np.zeros((6, 7))  # Contador de visitas
        
    def load_model(self):
        """Carga el modelo y agentes"""
        self.value_table = load_value_table(self.model_path)
        
        if not self.value_table:
            print(f"  Advertencia: {self.model_path} vacío o no existe")
        else:
            print(f" Modelo cargado: {len(self.value_table):,} estados")
        
        self.participants = find_importable_classes("groups", Policy)
        
        if self.learner_name not in self.participants:
            print(f" Error: '{self.learner_name}' no encontrado en groups/")
            return False
        
        print(f" Agente cargado: {self.learner_name}")
        return True
    
    def analyze_column_preferences(self, n_games: int = 500, opponent_name: str = None):
        """
        Analiza win rate por columna jugando múltiples partidas
        
        Args:
            n_games: Número de partidas a analizar
            opponent_name: Nombre del oponente (None = aleatorio)
        """
        print(f"\n Analizando preferencias por columna ({n_games} partidas)...")
        
        LearningPolicy = self.participants[self.learner_name]
        
        # Elegir oponente
        if opponent_name and opponent_name in self.participants:
            OpponentPolicy = self.participants[opponent_name]
            print(f"   Oponente: {opponent_name}")
        else:
            # Crear una política aleatoria simple
            class RandomPolicy(Policy):
                def act(self, board):
                    state = ConnectState()
                    state.board = board.copy()
                    valid = state.get_valid_actions()
                    return np.random.choice(valid)
            OpponentPolicy = RandomPolicy
            print(f"   Oponente: Random")
        
        for game in range(n_games):
            learner = LearningPolicy()
            learner.mount()
            
            opponent = OpponentPolicy()
            opponent.mount()
            
            state = ConnectState()
            first_move = None
            
            while not state.is_final():
                if state.player == -1:
                    action = learner.act(state.board)
                    if first_move is None:
                        first_move = int(action)
                else:
                    action = opponent.act(state.board)
                
                state = state.transition(int(action))
            
            winner = state.get_winner()
            
            # Registrar resultado por columna inicial
            if first_move is not None:
                self.column_win_rates[first_move]['total'] += 1
                if winner == -1:
                    self.column_win_rates[first_move]['wins'] += 1
                    self.opening_move_analysis[first_move]['wins'] += 1
                elif winner == 0:
                    self.opening_move_analysis[first_move]['draws'] += 1
                else:
                    self.opening_move_analysis[first_move]['losses'] += 1
            
            if (game + 1) % 100 == 0:
                print(f"   Progreso: {game + 1}/{n_games}")
        
        print(" Análisis completado")
    
    def analyze_board_values(self, max_states: int = 5000):
        """
        Analiza valores de estados en la tabla para crear heatmap del tablero
        
        Args:
            max_states: Máximo de estados a analizar (por rendimiento)
        """
        print(f"\n🔍 Analizando valores de tablero (máx {max_states} estados)...")
        
        if not self.value_table:
            print("  No hay tabla de valores para analizar")
            return
        
        states_analyzed = 0
        
        for state_str, value in list(self.value_table.items())[:max_states]:
            try:
                # Parsear el estado
                parts = state_str.split('|')
                if len(parts) != 2:
                    continue
                
                player = int(parts[0])
                board_str = parts[1]
                
                if len(board_str) != 42:  # 6x7 = 42
                    continue
                
                # Reconstruir tablero
                board = np.array([int(c) for c in board_str]).reshape(6, 7)
                
                # Solo analizar desde perspectiva del jugador -1
                if player == -1:
                    # Acumular valores por posición
                    for i in range(6):
                        for j in range(7):
                            if board[i, j] == -1:  # Ficha del agente
                                self.position_heatmap[i, j] += value
                                self.position_visit_count[i, j] += 1
                
                states_analyzed += 1
                
            except Exception:
                continue
        
        # Promediar valores
        with np.errstate(divide='ignore', invalid='ignore'):
            self.position_heatmap = np.where(
                self.position_visit_count > 0,
                self.position_heatmap / self.position_visit_count,
                0
            )
        
        print(f" Analizados {states_analyzed} estados")
    
    def create_column_heatmap(self):
        """Genera heatmap de win rate por columna"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Preparar datos
        columns = list(range(7))
        win_rates = []
        total_games = []
        
        for col in columns:
            stats = self.column_win_rates[col]
            if stats['total'] > 0:
                wr = stats['wins'] / stats['total']
                win_rates.append(wr * 100)
                total_games.append(stats['total'])
            else:
                win_rates.append(0)
                total_games.append(0)
        
        # ========== GRÁFICA 1: Win Rate por Columna ==========
        ax1 = axes[0]
        
        # Colores según win rate
        colors = []
        for wr in win_rates:
            if wr >= 70:
                colors.append('#2ecc71')  # Verde
            elif wr >= 50:
                colors.append('#f39c12')  # Naranja
            else:
                colors.append('#e74c3c')  # Rojo
        
        bars = ax1.bar(columns, win_rates, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
        
        # Agregar valores encima
        for i, (bar, wr, total) in enumerate(zip(bars, win_rates, total_games)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{wr:.1f}%\n({total})',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # Agregar estrella en columna central
            if i == 3:
                ax1.text(bar.get_x() + bar.get_width()/2., -8,
                        '*', ha='center', va='top', fontsize=20)
        
        ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Baseline (50%)')
        ax1.set_title(' Win Rate por Columna de Apertura', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Columna', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Win Rate (%)', fontsize=13, fontweight='bold')
        ax1.set_xticks(columns)
        ax1.set_xticklabels([f'Col {i}' for i in columns], fontsize=11)
        ax1.set_ylim(0, 100)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # ========== GRÁFICA 2: Distribución de Resultados por Columna ==========
        ax2 = axes[1]
        
        # Preparar datos apilados
        wins = []
        draws = []
        losses = []
        
        for col in columns:
            stats = self.opening_move_analysis[col]
            total = stats['wins'] + stats['draws'] + stats['losses']
            if total > 0:
                wins.append(stats['wins'] / total * 100)
                draws.append(stats['draws'] / total * 100)
                losses.append(stats['losses'] / total * 100)
            else:
                wins.append(0)
                draws.append(0)
                losses.append(0)
        
        x = np.arange(len(columns))
        width = 0.6
        
        p1 = ax2.bar(x, wins, width, label='Victorias', color='#2ecc71', edgecolor='black', linewidth=1.5)
        p2 = ax2.bar(x, draws, width, bottom=wins, label='Empates', color='#f39c12', edgecolor='black', linewidth=1.5)
        p3 = ax2.bar(x, losses, width, bottom=np.array(wins) + np.array(draws), 
                     label='Derrotas', color='#e74c3c', edgecolor='black', linewidth=1.5)
        
        ax2.set_title(' Distribución de Resultados por Columna', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Columna', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Porcentaje (%)', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Col {i}' for i in columns], fontsize=11)
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=11, loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('heatmap_column_preferences.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        print(" Gráfica guardada: heatmap_column_preferences.png")
    
    def create_board_value_heatmap(self):
        """Genera heatmap del tablero con valores promedio por posición"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # ========== HEATMAP 1: Valores Promedio ==========
        ax1 = axes[0]
        
        # Crear heatmap
        sns.heatmap(self.position_heatmap, 
                    annot=True, 
                    fmt='.2f',
                    cmap='RdYlGn',
                    center=0,
                    cbar_kws={'label': 'Valor Promedio'},
                    linewidths=1,
                    linecolor='black',
                    ax=ax1,
                    vmin=-1,
                    vmax=1)
        
        ax1.set_title(' Mapa de Calor: Valores por Posición', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Columna', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Fila (0=arriba)', fontsize=12, fontweight='bold')
        ax1.set_xticklabels(range(7), fontsize=10)
        ax1.set_yticklabels(range(6), fontsize=10)
        
        # ========== HEATMAP 2: Frecuencia de Visitas ==========
        ax2 = axes[1]
        
        sns.heatmap(self.position_visit_count,
                    annot=True,
                    fmt='.0f',
                    cmap='Blues',
                    cbar_kws={'label': 'Visitas'},
                    linewidths=1,
                    linecolor='black',
                    ax=ax2)
        
        ax2.set_title(' Frecuencia de Visitas por Posición', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Columna', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Fila (0=arriba)', fontsize=12, fontweight='bold')
        ax2.set_xticklabels(range(7), fontsize=10)
        ax2.set_yticklabels(range(6), fontsize=10)
        
        plt.tight_layout()
        plt.savefig('heatmap_board_values.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        print(" Gráfica guardada: heatmap_board_values.png")
    
    def create_strategic_bias_analysis(self):
        """Analiza y visualiza sesgos estratégicos del agente"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Preparar datos
        columns = list(range(7))
        win_rates = []
        usage_counts = []
        
        for col in columns:
            stats = self.column_win_rates[col]
            if stats['total'] > 0:
                win_rates.append(stats['wins'] / stats['total'])
                usage_counts.append(stats['total'])
            else:
                win_rates.append(0)
                usage_counts.append(0)
        
        total_usage = sum(usage_counts)
        usage_percentages = [c / total_usage * 100 if total_usage > 0 else 0 for c in usage_counts]
        
        # ========== GRÁFICA 1: Preferencia vs Performance ==========
        ax1 = axes[0, 0]
        
        scatter = ax1.scatter(usage_percentages, [wr * 100 for wr in win_rates], 
                             s=500, c=columns, cmap='viridis', 
                             edgecolors='black', linewidth=2, alpha=0.8)
        
        # Etiquetar puntos
        for i, (x, y) in enumerate(zip(usage_percentages, [wr * 100 for wr in win_rates])):
            label = f'Col {i}'
            if i == 3:
                label += ' *'
            ax1.annotate(label, (x, y), fontsize=11, fontweight='bold',
                        ha='center', va='center')
        
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Win Rate 50%')
        ax1.set_title(' Preferencia vs Performance', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Frecuencia de Uso (%)', fontsize=11)
        ax1.set_ylabel('Win Rate (%)', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ========== GRÁFICA 2: Sesgo Centro vs Bordes ==========
        ax2 = axes[0, 1]
        
        center_cols = [2, 3, 4]  # Columnas centrales
        edge_cols = [0, 1, 5, 6]  # Columnas de borde
        
        center_usage = sum(usage_counts[i] for i in center_cols)
        edge_usage = sum(usage_counts[i] for i in edge_cols)
        
        center_wins = sum(self.column_win_rates[i]['wins'] for i in center_cols)
        center_total = sum(self.column_win_rates[i]['total'] for i in center_cols)
        
        edge_wins = sum(self.column_win_rates[i]['wins'] for i in edge_cols)
        edge_total = sum(self.column_win_rates[i]['total'] for i in edge_cols)
        
        center_wr = center_wins / center_total * 100 if center_total > 0 else 0
        edge_wr = edge_wins / edge_total * 100 if edge_total > 0 else 0
        
        categories = ['Centro\n(2,3,4)', 'Bordes\n(0,1,5,6)']
        usage = [center_usage / total_usage * 100 if total_usage > 0 else 0,
                 edge_usage / total_usage * 100 if total_usage > 0 else 0]
        performance = [center_wr, edge_wr]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, usage, width, label='Frecuencia de Uso (%)',
                       color='steelblue', edgecolor='black', linewidth=2)
        bars2 = ax2.bar(x + width/2, performance, width, label='Win Rate (%)',
                       color='orange', edgecolor='black', linewidth=2)
        
        # Agregar valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{height:.1f}%', ha='center', va='bottom', 
                        fontsize=11, fontweight='bold')
        
        ax2.set_title(' Sesgo: Centro vs Bordes', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Porcentaje (%)', fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # ========== GRÁFICA 3: Análisis de Racionalidad ==========
        ax3 = axes[1, 0]
        
        # Calcular "índice de racionalidad": correlación entre uso y win rate
        if len([wr for wr in win_rates if wr > 0]) > 1:
            # Normalizar para comparar
            norm_usage = np.array(usage_percentages) / max(usage_percentages) if max(usage_percentages) > 0 else np.zeros(7)
            norm_wr = np.array(win_rates)
            
            correlation = np.corrcoef(norm_usage, norm_wr)[0, 1] if not np.isnan(np.corrcoef(norm_usage, norm_wr)[0, 1]) else 0
            
            # Visualizar correlación
            ax3.scatter(norm_usage, norm_wr, s=400, c=columns, cmap='coolwarm',
                       edgecolors='black', linewidth=2, alpha=0.8)
            
            for i, (x, y) in enumerate(zip(norm_usage, norm_wr)):
                ax3.annotate(f'{i}', (x, y), fontsize=11, fontweight='bold',
                           ha='center', va='center')
            
            # Línea de tendencia
            if not np.isnan(correlation):
                z = np.polyfit(norm_usage, norm_wr, 1)
                p = np.poly1d(z)
                ax3.plot(norm_usage, p(norm_usage), "r--", alpha=0.8, linewidth=2,
                        label=f'Correlación: {correlation:.2f}')
            
            ax3.set_title(' Índice de Racionalidad', fontsize=13, fontweight='bold')
            ax3.set_xlabel('Uso Normalizado', fontsize=11)
            ax3.set_ylabel('Win Rate', fontsize=11)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Interpretación
            if correlation > 0.5:
                interpretation = " Alta racionalidad: Prefiere jugadas ganadoras"
            elif correlation > 0:
                interpretation = " Racionalidad moderada"
            else:
                interpretation = " Baja racionalidad: Preferencias no correlacionan con éxito"
            
            ax3.text(0.5, -0.15, interpretation, transform=ax3.transAxes,
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ========== GRÁFICA 4: Matriz de Dominancia ==========
        ax4 = axes[1, 1]
        
        # Crear matriz de "dominancia" (win rate relativo)
        dominance_matrix = np.zeros((7, 7))
        
        for i in range(7):
            for j in range(7):
                if i == j:
                    dominance_matrix[i, j] = win_rates[i]
                else:
                    # Diferencia relativa
                    dominance_matrix[i, j] = win_rates[i] - win_rates[j]
        
        sns.heatmap(dominance_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlGn',
                    center=0,
                    cbar_kws={'label': 'Diferencia de Win Rate'},
                    linewidths=1,
                    linecolor='black',
                    ax=ax4,
                    xticklabels=range(7),
                    yticklabels=range(7))
        
        ax4.set_title('🏆 Matriz de Dominancia entre Columnas', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Columna Comparada', fontsize=11)
        ax4.set_ylabel('Columna Base', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('heatmap_strategic_bias.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        print(" Gráfica guardada: heatmap_strategic_bias.png")
    
    def print_analysis_report(self):
        """Imprime reporte textual del análisis"""
        print("\n" + "="*80)
        print(" REPORTE DE ANÁLISIS DE POLÍTICA")
        print("="*80)
        
        # Win rates por columna
        print("\n WIN RATE POR COLUMNA DE APERTURA:")
        print(f"  {'Columna':<10} {'Partidas':<10} {'Win Rate':<12} {'Evaluación'}")
        print(f"  {'-'*70}")
        
        for col in range(7):
            stats = self.column_win_rates[col]
            if stats['total'] > 0:
                wr = stats['wins'] / stats['total']
                emoji = "E" if wr >= 0.7 else "B" if wr >= 0.6 else "A" if wr >= 0.5 else "I"
                star = " *" if col == 3 else ""
                print(f"  Col {col}{star:<6} {stats['total']:<10} {wr:>10.1%}  {emoji}")
        
        # Análisis centro vs bordes
        print("\n ANÁLISIS CENTRO VS BORDES:")
        
        center_cols = [2, 3, 4]
        edge_cols = [0, 1, 5, 6]
        
        center_total = sum(self.column_win_rates[i]['total'] for i in center_cols)
        edge_total = sum(self.column_win_rates[i]['total'] for i in edge_cols)
        total = center_total + edge_total
        
        if total > 0:
            center_usage = center_total / total * 100
            edge_usage = edge_total / total * 100
            
            print(f"  • Centro (2,3,4):  {center_usage:.1f}% de las aperturas")
            print(f"  • Bordes (0,1,5,6): {edge_usage:.1f}% de las aperturas")
            
            if center_usage > 60:
                print(f"   Preferencia clara por centro (estrategia sólida)")
            elif center_usage > 40:
                print(f"    Preferencia moderada por centro")
            else:
                print(f"   Poca preferencia por centro (revisar estrategia)")
        
        # Mejor y peor columna
        print("\n🏆 MEJORES Y PEORES COLUMNAS:")
        
        valid_cols = [(col, stats) for col, stats in self.column_win_rates.items() 
                      if stats['total'] > 10]  # Mínimo 10 partidas
        
        if valid_cols:
            sorted_cols = sorted(valid_cols, 
                               key=lambda x: x[1]['wins'] / x[1]['total'], 
                               reverse=True)
            
            best_col, best_stats = sorted_cols[0]
            worst_col, worst_stats = sorted_cols[-1]
            
            best_wr = best_stats['wins'] / best_stats['total']
            worst_wr = worst_stats['wins'] / worst_stats['total']
            
            print(f"  • Mejor:  Columna {best_col} ({best_wr:.1%} win rate)")
            print(f"  • Peor:   Columna {worst_col} ({worst_wr:.1%} win rate)")
            print(f"  • Diferencia: {(best_wr - worst_wr) * 100:.1f} puntos porcentuales")
        
        print("="*80 + "\n")


# SCRIPT PRINCIPAL

def main():
    # Configuración
    LEARNER_NAME = "My Agent"
    MODEL_PATH = "values.json"
    N_GAMES = 500  # Partidas para analizar preferencias
    OPPONENT = None  # None = aleatorio, o nombre de agente en groups/
    
    print("="*80)
    print(" GENERADOR DE MAPAS DE CALOR - ANÁLISIS DE POLÍTICA")
    print("="*80)
    
    # Crear analizador
    analyzer = PolicyHeatmapAnalyzer(learner_name=LEARNER_NAME, model_path=MODEL_PATH)
    
    # Cargar modelo
    if not analyzer.load_model():
        return
    
    # Paso 1: Analizar preferencias por columna
    print("\n" + "─"*80)
    print("PASO 1: ANÁLISIS DE PREFERENCIAS POR COLUMNA")
    print("─"*80)
    analyzer.analyze_column_preferences(n_games=N_GAMES, opponent_name=OPPONENT)
    
    # Paso 2: Analizar valores del tablero
    print("\n" + "─"*80)
    print("PASO 2: ANÁLISIS DE VALORES DE TABLERO")
    print("─"*80)
    analyzer.analyze_board_values(max_states=5000)
    
    # Paso 3: Generar visualizaciones
    print("\n" + "─"*80)
    print("PASO 3: GENERANDO VISUALIZACIONES")
    print("─"*80)
    
    print("\n Creando heatmap de columnas...")
    analyzer.create_column_heatmap()
    
    print("\n Creando heatmap de tablero...")
    analyzer.create_board_value_heatmap()
    
    print("\n Creando análisis de sesgos estratégicos...")
    analyzer.create_strategic_bias_analysis()
    
    # Paso 4: Imprimir reporte textual
    print("\n" + "─"*80)
    print("PASO 4: REPORTE DE ANÁLISIS")
    print("─"*80)
    analyzer.print_analysis_report()
    
    # Resumen final
    print("="*80)
    print(" ANÁLISIS COMPLETADO")
    print("="*80)
    print("\n Archivos generados:")
    print("   • heatmap_column_preferences.png  - Win rate y distribución por columna")
    print("   • heatmap_board_values.png        - Valores y visitas por posición")
    print("   • heatmap_strategic_bias.png      - Análisis de sesgos estratégicos")
    print("\n Interpretación de los mapas de calor:")
    print("    Verde = Buena estrategia / Alto valor")
    print("    Amarillo = Estrategia neutral")
    print("    Rojo = Mala estrategia / Bajo valor")
    print("\n Usa estos gráficos en tu presentación para:")
    print("   ✓ Demostrar que el agente prefiere columnas centrales")
    print("   ✓ Mostrar racionalidad (usa más las jugadas que ganan)")
    print("   ✓ Identificar posiciones más valiosas en el tablero")
    print("   ✓ Revelar sesgos y patrones estratégicos aprendidos")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()