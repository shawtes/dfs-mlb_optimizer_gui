import sys
import logging
import traceback
import psutil
import pulp
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import concurrent.futures
from itertools import combinations
import csv
from collections import defaultdict

# ... existing imports ...

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
SALARY_CAP = 50000
POSITION_LIMITS = {
    'P': 2,
    'C': 1,
    '1B': 1,
    '2B': 1,
    '3B': 1,
    'SS': 1,
    'OF': 3
}
REQUIRED_TEAM_SIZE = 10

def optimize_single_lineup(args):
    df, stack_type, team_projected_runs, team_selections = args
    logging.debug(f"optimize_single_lineup: Starting with stack type {stack_type}")
    
    problem = pulp.LpProblem("Stack_Optimization", pulp.LpMaximize)
    player_vars = {idx: pulp.LpVariable(f"player_{idx}", cat='Binary') for idx in df.index}

    # Objective: Maximize projected points
    problem += pulp.lpSum([df.at[idx, 'My Proj'] * player_vars[idx] for idx in df.index])

    # Basic constraints
    problem += pulp.lpSum(player_vars.values()) == REQUIRED_TEAM_SIZE
    problem += pulp.lpSum([df.at[idx, 'Salary'] * player_vars[idx] for idx in df.index]) <= SALARY_CAP
    for position, limit in POSITION_LIMITS.items():
        problem += pulp.lpSum([player_vars[idx] for idx in df.index if position in df.at[idx, 'Pos']]) == limit

    # Implement stacking
    stack_sizes = [int(size) for size in stack_type.split('|')]
    total_stack_size = sum(stack_sizes)
    non_stack_size = REQUIRED_TEAM_SIZE - total_stack_size

    for i, size in enumerate(stack_sizes):
        team_vars = {team: pulp.LpVariable(f"team_{team}_{i}", cat='Binary') for team in team_selections[size]}
        problem += pulp.lpSum(team_vars.values()) == 1
        
        for team in team_selections[size]:
            team_players = df[(df['Team'] == team) & (~df['Pos'].str.contains('P'))].index
            problem += pulp.lpSum([player_vars[idx] for idx in team_players]) >= size * team_vars[team]

    # Ensure the correct number of non-stack players
    problem += pulp.lpSum([player_vars[idx] for idx in df.index if 'P' in df.at[idx, 'Pos']]) + \
               pulp.lpSum([player_vars[idx] for idx in df.index if 'P' not in df.at[idx, 'Pos']]) - \
               pulp.lpSum([size * pulp.lpSum(team_vars.values()) for size in stack_sizes]) == non_stack_size

    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
    status = problem.solve(solver)

    if pulp.LpStatus[status] == 'Optimal':
        lineup = df.loc[[idx for idx in df.index if player_vars[idx].varValue > 0.5]]
        logging.debug(f"optimize_single_lineup: Found optimal solution with {len(lineup)} players")
        return lineup, stack_type
    else:
        logging.debug(f"optimize_single_lineup: No optimal solution found. Status: {pulp.LpStatus[status]}")
        logging.debug(f"Constraints: {problem.constraints}")
        return pd.DataFrame(), stack_type
def simulate_iteration(df):
    random_factors = np.random.normal(1, 0.1, size=len(df))
    df = df.copy()
    df['My Proj'] = df['My Proj'] * random_factors
    df['My Proj'] = df['My Proj'].clip(lower=1)
    return df

class OptimizationWorker(QThread):
    optimization_done = pyqtSignal(dict, dict, dict)

    def __init__(self, df_players, salary_cap, position_limits, included_players, stack_settings, min_exposure, max_exposure, min_points, monte_carlo_iterations,num_lineups):
        super().__init__()
        self.df_players = df_players
        self.num_lineups = num_lineups
        self.salary_cap = salary_cap
        self.position_limits = position_limits
        self.included_players = included_players
        self.stack_settings = stack_settings
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.team_projected_runs = self.calculate_team_projected_runs(df_players)
        
        self.max_workers = multiprocessing.cpu_count()  # Or set a specific number
        self.min_points = min_points
        self.monte_carlo_iterations = monte_carlo_iterations
        self.team_selections = {}  # This will be populated in preprocess_data

    def run(self):
        logging.debug("OptimizationWorker: Starting optimization")
        results, team_exposure, stack_exposure = self.optimize_lineups()
        logging.debug(f"OptimizationWorker: Optimization complete. Results: {len(results)}")
        self.optimization_done.emit(results, team_exposure, stack_exposure)

    def optimize_lineups(self):
        df_filtered = self.preprocess_data()
        logging.debug(f"optimize_lineups: Starting with {len(df_filtered)} players")

        results = {}
        team_exposure = defaultdict(int)
        stack_exposure = defaultdict(int)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for stack_type in self.stack_settings:
                for _ in range(self.num_lineups):
                    future = executor.submit(optimize_single_lineup, (df_filtered.copy(), stack_type, self.team_projected_runs, self.team_selections))
                    futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    lineup, stack_type = future.result()
                    if lineup.empty:
                        logging.debug(f"optimize_lineups: Empty lineup returned for stack type {stack_type}")
                    else:
                        total_points = lineup['My Proj'].sum()
                        results[len(results)] = {'total_points': total_points, 'lineup': lineup}
                        for team in lineup['Team'].unique():
                            team_exposure[team] += 1
                        stack_exposure[stack_type] += 1
                        logging.debug(f"optimize_lineups: Found valid lineup for stack type {stack_type}")
                except Exception as e:
                    logging.error(f"Error in optimization: {str(e)}")

        logging.debug(f"optimize_lineups: Completed. Found {len(results)} valid lineups")
        logging.debug(f"Team exposure: {dict(team_exposure)}")
        logging.debug(f"Stack exposure: {dict(stack_exposure)}")
        
        return results, team_exposure, stack_exposure
    def preprocess_data(self):
        logging.debug("preprocess_data: Starting")
        df_filtered = self.df_players[self.df_players['My Proj'] > 0]  # Filter out players with 0 or negative projections
        df_filtered = df_filtered[df_filtered['Salary'] > 0]  # Filter out players with 0 or negative salary
        
        if self.included_players:
            df_filtered = df_filtered[df_filtered['Name'].isin(self.included_players)]
        
        # Create team_selections based on available teams
        available_teams = df_filtered['Team'].unique()
        self.team_selections = {
            stack_size: available_teams
            for stack_size in set(int(size) for stack in self.stack_settings for size in stack.split('|'))
        }
        
        logging.debug(f"preprocess_data: Filtered data shape: {df_filtered.shape}")
        logging.debug(f"preprocess_data: Available teams: {available_teams}")
        logging.debug(f"preprocess_data: Team selections: {self.team_selections}")
        return df_filtered
    def calculate_team_projected_runs(self, df):
        return {team: self.calculate_projected_runs(group) 
                for team, group in df.groupby('Team')}

    def calculate_projected_runs(self, team_players):
        if 'Saber Total' in team_players.columns:
            return team_players['Saber Total'].mean()
        elif 'My Proj' in team_players.columns:
            return team_players['My Proj'].sum() * 0.5
        else:
            logging.warning(f"No projection data available for team {team_players['Team'].iloc[0]}")
            return 0

class FantasyBaseballApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced MLB DFS Optimizer")
        self.setGeometry(100, 100, 1600, 1000)
        self.setup_ui()
        self.included_players = []
        self.stack_settings = {}
        self.min_exposure = {}
        self.max_exposure = {}
        self.min_points = 1
        self.monte_carlo_iterations = 100

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        self.tabs = QTabWidget()
        self.splitter.addWidget(self.tabs)

        self.df_players = None
        self.df_entries = None
        self.player_exposure = {}
        self.optimized_lineups = []

        self.create_players_tab()
        self.create_team_stack_tab()
        self.create_stack_exposure_tab()
        self.create_control_panel()

    def create_players_tab(self):
        players_tab = QWidget()
        self.tabs.addTab(players_tab, "Players")

        players_layout = QVBoxLayout(players_tab)

        position_tabs = QTabWidget()
        players_layout.addWidget(position_tabs)

        self.player_tables = {}

        positions = ["All Batters", "C", "1B", "2B", "3B", "SS", "OF", "P"]
        for position in positions:
            sub_tab = QWidget()
            position_tabs.addTab(sub_tab, position)
            layout = QVBoxLayout(sub_tab)

            select_all_button = QPushButton("Select All")
            deselect_all_button = QPushButton("Deselect All")
            select_all_button.clicked.connect(lambda _, p=position: self.select_all(p))
            deselect_all_button.clicked.connect(lambda _, p=position: self.deselect_all(p))
            button_layout = QHBoxLayout()
            button_layout.addWidget(select_all_button)
            button_layout.addWidget(deselect_all_button)
            layout.addLayout(button_layout)

            table = QTableWidget(0, 11)
            table.setHorizontalHeaderLabels(["Select", "Name", "Team", "Pos", "Salary", "My Proj", "Own", "Min Exp", "Max Exp", "Actual Exp (%)", "My Proj"])
            layout.addWidget(table)

            self.player_tables[position] = table

    def create_team_stack_tab(self):
        team_stack_tab = QWidget()
        self.tabs.addTab(team_stack_tab, "Team Stacks")

        layout = QVBoxLayout(team_stack_tab)

        stack_size_tabs = QTabWidget()
        layout.addWidget(stack_size_tabs)

        stack_sizes = ["All Stacks", "2 Stack", "3 Stack", "4 Stack", "5 Stack"]
        self.team_stack_tables = {}

        for stack_size in stack_sizes:
            sub_tab = QWidget()
            stack_size_tabs.addTab(sub_tab, stack_size)
            sub_layout = QVBoxLayout(sub_tab)

            table = QTableWidget(0, 8)
            table.setHorizontalHeaderLabels(["Select", "Teams", "Status", "Time", "Proj Runs", "Min Exp", "Max Exp", "Actual Exp (%)"])
            sub_layout.addWidget(table)

            self.team_stack_tables[stack_size] = table

        self.team_stack_table = self.team_stack_tables["All Stacks"]

        refresh_button = QPushButton("Refresh Team Stacks")
        refresh_button.clicked.connect(self.refresh_team_stacks)
        layout.addWidget(refresh_button)

    def refresh_team_stacks(self):
        self.populate_team_stack_table()

    def create_stack_exposure_tab(self):
        stack_exposure_tab = QWidget()
        self.tabs.addTab(stack_exposure_tab, "Stack Exposure")
    
        layout = QVBoxLayout(stack_exposure_tab)
    
        self.stack_exposure_table = QTableWidget(0, 7)
        self.stack_exposure_table.setHorizontalHeaderLabels(["Select", "Stack Type", "Min Exp", "Max Exp", "Lineup Exp", "Pool Exp", "Entry Exp"])
        layout.addWidget(self.stack_exposure_table)
    
        stack_types = ["4|2|2", "4|2", "3|3|2", "3|2|2", "2|2|2", "5|3", "5|2", "No Stacks"]
        for stack_type in stack_types:
            row_position = self.stack_exposure_table.rowCount()
            self.stack_exposure_table.insertRow(row_position)
    
            checkbox = QCheckBox()
            checkbox_widget = QWidget()
            layout_checkbox = QHBoxLayout(checkbox_widget)
            layout_checkbox.addWidget(checkbox)
            layout_checkbox.setAlignment(Qt.AlignCenter)
            layout_checkbox.setContentsMargins(0, 0, 0, 0)
            self.stack_exposure_table.setCellWidget(row_position, 0, checkbox_widget)
    
            self.stack_exposure_table.setItem(row_position, 1, QTableWidgetItem(stack_type))
            min_exp_item = QTableWidgetItem("0")
            min_exp_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
            self.stack_exposure_table.setItem(row_position, 2, min_exp_item)
    
            max_exp_item = QTableWidgetItem("100")
            max_exp_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
            self.stack_exposure_table.setItem(row_position, 3, max_exp_item)
    
            self.stack_exposure_table.setItem(row_position, 4, QTableWidgetItem("0.0%"))
            self.stack_exposure_table.setItem(row_position, 5, QTableWidgetItem("0.0%"))
            self.stack_exposure_table.setItem(row_position, 6, QTableWidgetItem("0.0%"))

    def create_control_panel(self):
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_layout = QVBoxLayout(control_panel)

        self.splitter.addWidget(control_panel)

        load_button = QPushButton('Load CSV')
        load_button.clicked.connect(self.load_file)
        control_layout.addWidget(load_button)

        load_entries_button = QPushButton('Load Entries CSV')
        load_entries_button.clicked.connect(self.load_entries_csv)
        control_layout.addWidget(load_entries_button)

        self.min_unique_label = QLabel('Min Unique:')
        self.min_unique_input = QLineEdit()
        control_layout.addWidget(self.min_unique_label)
        control_layout.addWidget(self.min_unique_input)

        self.sorting_label = QLabel('Sorting Method:')
        self.sorting_combo = QComboBox()
        self.sorting_combo.addItems(["Points", "Value", "Salary"])
        control_layout.addWidget(self.sorting_label)
        control_layout.addWidget(self.sorting_combo)

        run_button = QPushButton('Run Contest Sim')
        run_button.clicked.connect(self.run_optimization)
        control_layout.addWidget(run_button)

        save_button = QPushButton('Save CSV for DraftKings')
        save_button.clicked.connect(self.save_csv)
        control_layout.addWidget(save_button)

        self.results_table = QTableWidget(0, 9)
        self.results_table.setHorizontalHeaderLabels(["Player", "Team", "Pos", "Salary", "My Proj", "Total Salary", "Total Points", "Exposure (%)", "Max Exp (%)"])
        control_layout.addWidget(self.results_table)

        self.status_label = QLabel('')
        control_layout.addWidget(self.status_label)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open CSV', '', 'CSV Files (*.csv)')
        if file_path:
            self.df_players = self.load_players(file_path)
            self.populate_player_tables()

    def load_entries_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Entries CSV', '', 'CSV Files (*.csv)')
        if file_path:
            self.df_entries = self.load_and_standardize_csv(file_path)
            if self.df_entries is not None:
                self.status_label.setText('Entries CSV loaded and standardized successfully.')
            else:
                self.status_label.setText('Failed to standardize Entries CSV.')

    def load_players(self, csv_path):
        df = pd.read_csv(csv_path)
        required_columns = ['Name', 'Team', 'Opp', 'Pos', 'My Proj', 'Salary']
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan
        df['My Proj'] = pd.to_numeric(df['My Proj'], errors='coerce')
        df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
        df['Positions'] = df['Pos'].apply(lambda x: x.split('/') if pd.notna(x) else [])
        return df

    def load_and_standardize_csv(self, file_path):
        try:
            df = pd.read_csv(file_path, skiprows=6, on_bad_lines='skip')
            df.columns = ['ID', 'Name', 'Other Columns...'] + df.columns[3:].tolist()
            return df
        except Exception as e:
            logging.error(f"Error loading or processing file: {e}")
            return None

    def populate_player_tables(self):
        positions = ["All Batters", "C", "1B", "2B", "3B", "SS", "OF", "P"]
        
        for position in positions:
            table = self.player_tables[position]
            table.setRowCount(0)

            if self.df_players is not None:
                if position == "P":
                    df_filtered = self.df_players[self.df_players['Pos'].str.contains('SP|RP|P', na=False)]
                elif position == "All Batters":
                    df_filtered = self.df_players[~self.df_players['Pos'].str.contains('SP|RP|P', na=False)]
                else:
                    df_filtered = self.df_players[self.df_players['Positions'].apply(lambda x: position in x)]
            
                for _, row in df_filtered.iterrows():
                    row_position = table.rowCount()
                    table.insertRow(row_position)

                    checkbox = QCheckBox()
                    checkbox_widget = QWidget()
                    layout_checkbox = QHBoxLayout(checkbox_widget)
                    layout_checkbox.addWidget(checkbox)
                    layout_checkbox.setAlignment(Qt.AlignCenter)
                    layout_checkbox.setContentsMargins(0, 0, 0, 0)
                    table.setCellWidget(row_position, 0, checkbox_widget)
        
                    table.setItem(row_position, 1, QTableWidgetItem(str(row['Name'])))
                    table.setItem(row_position, 2, QTableWidgetItem(str(row['Team'])))
                    table.setItem(row_position, 3, QTableWidgetItem(str(row['Pos'])))
                    table.setItem(row_position, 4, QTableWidgetItem(str(row['Salary'])))
                    table.setItem(row_position, 5, QTableWidgetItem(str(row['My Proj'])))
        
                    min_exp_spinbox = QSpinBox()
                    min_exp_spinbox.setRange(0, 100)
                    min_exp_spinbox.setValue(0)
                    table.setCellWidget(row_position, 7, min_exp_spinbox)
        
                    max_exp_spinbox = QSpinBox()
                    max_exp_spinbox.setRange(0, 100)
                    max_exp_spinbox.setValue(100)
                    table.setCellWidget(row_position, 8, max_exp_spinbox)
        
                    actual_exp_label = QLabel("")
                    table.setCellWidget(row_position, 9, actual_exp_label)

                    if row['Name'] not in self.player_exposure:
                        self.player_exposure[row['Name']] = 0

        self.populate_team_stack_table()
        
    def populate_team_stack_table(self):
        team_runs = self.calculate_team_projected_runs()
        selected_teams = self.get_selected_teams()

        for stack_size, table in self.team_stack_tables.items():
            table.setRowCount(0)
            for team in selected_teams:
                self.add_team_to_stack_table(table, team, team_runs.get(team, 0))

    def get_selected_teams(self):
        selected_teams = set()
        for position in self.player_tables:
            table = self.player_tables[position]
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget is not None:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox is not None and checkbox.isChecked():
                        selected_teams.add(table.item(row, 2).text())
        return selected_teams
    def calculate_team_projected_runs(self):
        if self.df_players is None:
            return {}
        return {team: self.calculate_projected_runs(group) 
                for team, group in self.df_players.groupby('Team')}

    def calculate_projected_runs(self, team_group):
        if 'Saber Total' in team_group.columns:
            return team_group['Saber Total'].mean()
        elif 'My Proj' in team_group.columns:
            return team_group['My Proj'].sum() * 0.5
        else:
            logging.warning(f"No projection data available for team {team_group['Team'].iloc[0]}")
            return 0


    def add_team_to_stack_table(self, table, team, proj_runs):
        row_position = table.rowCount()
        table.insertRow(row_position)

        checkbox = QCheckBox()
        checkbox_widget = QWidget()
        layout_checkbox = QHBoxLayout(checkbox_widget)
        layout_checkbox.addWidget(checkbox)
        layout_checkbox.setAlignment(Qt.AlignCenter)
        layout_checkbox.setContentsMargins(0, 0, 0, 0)
        table.setCellWidget(row_position, 0, checkbox_widget)

        table.setItem(row_position, 1, QTableWidgetItem(team))
        table.setItem(row_position, 2, QTableWidgetItem("Playing"))
        table.setItem(row_position, 3, QTableWidgetItem("7:00 PM"))
        table.setItem(row_position, 4, QTableWidgetItem(f"{proj_runs:.2f}"))

        min_exp_spinbox = QSpinBox()
        min_exp_spinbox.setRange(0, 100)
        min_exp_spinbox.setValue(0)
        table.setCellWidget(row_position, 5, min_exp_spinbox)

        max_exp_spinbox = QSpinBox()
        max_exp_spinbox.setRange(0, 100)
        max_exp_spinbox.setValue(100)
        table.setCellWidget(row_position, 6, max_exp_spinbox)

        actual_exp_label = QLabel("")
        table.setCellWidget(row_position, 7, actual_exp_label)
    def select_all(self, position):
            table = self.player_tables[position]
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget is not None:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox is not None:
                        checkbox.setChecked(True)
            self.populate_team_stack_table()

    def deselect_all(self, position):
            table = self.player_tables[position]
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget is not None:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox is not None:
                        checkbox.setChecked(False)
            self.populate_team_stack_table()

    def run_optimization(self):
        logging.debug("Starting run_optimization method")
        if self.df_players is None or self.df_players.empty:
            self.status_label.setText("No player data loaded. Please load a CSV first.")
            logging.debug("No player data loaded")
            return
        
        logging.debug(f"df_players shape: {self.df_players.shape}")
        logging.debug(f"df_players columns: {self.df_players.columns}")
        logging.debug(f"df_players sample:\n{self.df_players.head()}")
        
        self.included_players = self.get_included_players()
        self.stack_settings = self.collect_stack_settings()
        self.min_exposure, self.max_exposure = self.collect_exposure_settings()
        
        logging.debug(f"Included players: {len(self.included_players)}")
        logging.debug(f"Stack settings: {self.stack_settings}")
        
        self.optimization_thread = OptimizationWorker(
            df_players=self.df_players,
            salary_cap=SALARY_CAP,
            position_limits=POSITION_LIMITS,
            included_players=self.included_players,
            stack_settings=self.stack_settings,
            min_exposure=self.min_exposure,
            max_exposure=self.max_exposure,
            min_points=self.min_points,
            monte_carlo_iterations=self.monte_carlo_iterations,
            num_lineups=100 
        )
        self.optimization_thread.optimization_done.connect(self.display_results)
        logging.debug("Starting optimization thread")
        self.optimization_thread.start()
        
        self.status_label.setText("Running optimization... Please wait.")

    def display_results(self, results, team_exposure, stack_exposure):
        logging.debug(f"display_results: Received {len(results)} results")
        self.results_table.setRowCount(0)
        total_lineups = len(results)

        sorted_results = sorted(results.items(), key=lambda x: x[1]['total_points'], reverse=True)

        self.optimized_lineups = []
        for _, lineup_data in sorted_results:
            self.add_lineup_to_results(lineup_data, total_lineups)
            self.optimized_lineups.append(lineup_data['lineup'])

        self.update_exposure_in_all_tabs(total_lineups, team_exposure, stack_exposure)
        self.refresh_team_stacks()
        self.status_label.setText(f"Optimization complete. Generated {total_lineups} lineups.")

    def add_lineup_to_results(self, lineup_data, total_lineups):
        total_points = lineup_data['total_points']
        lineup = lineup_data['lineup']
        total_salary = lineup['Salary'].sum()

        for _, player in lineup.iterrows():
            row_position = self.results_table.rowCount()
            self.results_table.insertRow(row_position)
            self.results_table.setItem(row_position, 0, QTableWidgetItem(str(player['Name'])))
            self.results_table.setItem(row_position, 1, QTableWidgetItem(str(player['Team'])))
            self.results_table.setItem(row_position, 2, QTableWidgetItem(str(player['Pos'])))
            self.results_table.setItem(row_position, 3, QTableWidgetItem(str(player['Salary'])))
            self.results_table.setItem(row_position, 4, QTableWidgetItem(f"{player['My Proj']:.2f}"))
            self.results_table.setItem(row_position, 5, QTableWidgetItem(str(total_salary)))
            self.results_table.setItem(row_position, 6, QTableWidgetItem(f"{total_points:.2f}"))

            player_name = player['Name']
            if player_name in self.player_exposure:
                self.player_exposure[player_name] += 1
            else:
                self.player_exposure[player_name] = 1

            exposure = self.player_exposure.get(player_name, 0) / total_lineups * 100
            self.results_table.setItem(row_position, 7, QTableWidgetItem(f"{exposure:.2f}%"))
            self.results_table.setItem(row_position, 8, QTableWidgetItem(f"{self.max_exposure.get(player_name, 100):.2f}%"))

    def update_exposure_in_all_tabs(self, total_lineups, team_exposure, stack_exposure):
        if total_lineups > 0:
            for position in self.player_tables:
                table = self.player_tables[position]
                for row in range(table.rowCount()):
                    player_name = table.item(row, 1).text()
                    actual_exposure = min(self.player_exposure.get(player_name, 0) / total_lineups * 100, 100)
                    actual_exposure_label = table.cellWidget(row, 9)
                    if isinstance(actual_exposure_label, QLabel):
                        actual_exposure_label.setText(f"{actual_exposure:.2f}%")

            for stack_size, table in self.team_stack_tables.items():
                for row in range(table.rowCount()):
                    team_name = table.item(row, 1).text()
                    actual_exposure = min(team_exposure.get(team_name, 0) / total_lineups * 100, 100)
                    table.setItem(row, 7, QTableWidgetItem(f"{actual_exposure:.2f}%"))

            for row in range(self.stack_exposure_table.rowCount()):
                stack_type = self.stack_exposure_table.item(row, 1).text()
                actual_exposure = min(stack_exposure.get(stack_type, 0) / total_lineups * 100, 100)
                self.stack_exposure_table.setItem(row, 4, QTableWidgetItem(f"{actual_exposure:.2f}%"))

    def save_csv(self):
        if not hasattr(self, 'optimized_lineups') or not self.optimized_lineups:
            self.status_label.setText('No optimized lineups to save. Please run optimization first.')
            return

        output_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")

        if not output_path:
            self.status_label.setText('Save operation canceled.')
            return

        try:
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF'])
                
                for lineup in self.optimized_lineups:
                    row = []
                    for _, player in lineup.iterrows():
                        row.append(player['Name'])
                    writer.writerow(row)
            
            self.status_label.setText(f'Optimized lineups saved successfully to {output_path}')
        except Exception as e:
            self.status_label.setText(f'Error saving CSV: {str(e)}')

    def generate_output(self, entries_df, players_df, output_path):
        optimized_output = players_df[["Name", "Team", "Pos", "Salary", "My Proj"]]
        optimized_output.to_csv(output_path, index=False)

    def get_included_players(self):
        included_players = []
        for position in self.player_tables:
            table = self.player_tables[position]
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget is not None:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox is not None and checkbox.isChecked():
                        included_players.append(table.item(row, 1).text())
        return included_players

    def collect_stack_settings(self):
        stack_settings = {}
        for row in range(self.stack_exposure_table.rowCount()):
            checkbox_widget = self.stack_exposure_table.cellWidget(row, 0)
            if checkbox_widget is not None:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox is not None and checkbox.isChecked():
                    stack_type = self.stack_exposure_table.item(row, 1).text()
                    stack_settings[stack_type] = True
        return stack_settings

    def collect_exposure_settings(self):
        min_exposure = {}
        self.max_exposure = {}
        for position in self.player_tables:
            table = self.player_tables[position]
            for row in range(table.rowCount()):
                player_name = table.item(row, 1).text()
                min_exp_widget = table.cellWidget(row, 7)
                max_exp_widget = table.cellWidget(row, 8)
                if isinstance(min_exp_widget, QSpinBox) and isinstance(max_exp_widget, QSpinBox):
                    min_exposure[player_name] = min_exp_widget.value() / 100
                    self.max_exposure[player_name] = max_exp_widget.value() / 100
        return min_exposure, self.max_exposure

    def collect_team_selections(self):
        team_selections = {}
        for stack_size, table in self.team_stack_tables.items():
            if stack_size != "All Stacks":
                team_selections[int(stack_size.split()[0])] = []
                for row in range(table.rowCount()):
                    checkbox_widget = table.cellWidget(row, 0)
                    if checkbox_widget is not None:
                        checkbox = checkbox_widget.findChild(QCheckBox)
                        if checkbox is not None and checkbox.isChecked():
                            team_selections[int(stack_size.split()[0])].append(table.item(row, 1).text())
        return team_selections

if __name__ == "__main__":
    logging.debug(f"PuLP version: {pulp.__version__}")
    logging.debug(f"PuLP test results: {pulp.pulpTestAll()}")
    
    app = QApplication(sys.argv)
    window = FantasyBaseballApp()
    window.show()
    sys.exit(app.exec_())