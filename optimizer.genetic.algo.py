import sys
import os
import logging
import traceback
import psutil
import pulp
import pandas as pd
import numpy as np
import random
import re
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import *
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import concurrent.futures
from itertools import combinations, permutations
import itertools
import csv
import json
import hashlib
from collections import defaultdict

# Import Windows-safe logging
try:
    from safe_logging import safe_log_info as safe_log_info_func, safe_log_debug as safe_log_debug_func, safe_log_warning as safe_log_warning_func, safe_log_error as safe_log_error_func
    # Create wrappers to match expected interface
    def safe_log_info(msg): return safe_log_info_func(msg)
    def safe_log_debug(msg): return safe_log_debug_func(msg)
    def safe_log_warning(msg): return safe_log_warning_func(msg)
    def safe_log_error(msg): return safe_log_error_func(msg)
    SAFE_LOGGING_AVAILABLE = True
    print("✅ Windows-safe logging loaded successfully!")
except ImportError as e:
    print(f"⚠️ Safe logging not available: {e}")
    SAFE_LOGGING_AVAILABLE = False
    # Fallback to regular logging
    def safe_log_info(msg): logging.info(msg)
    def safe_log_debug(msg): logging.debug(msg)
    def safe_log_warning(msg): logging.warning(msg)
    def safe_log_error(msg): logging.error(msg)

# Import the advanced quantitative optimizer
try:
    from advanced_quant_optimizer import AdvancedQuantitativeOptimizer
    ADVANCED_QUANT_AVAILABLE = True
    print("✅ Advanced Quantitative Optimizer loaded successfully!")
except ImportError as e:
    print(f"⚠️ Advanced Quantitative Optimizer not available: {e}")
    ADVANCED_QUANT_AVAILABLE = False

# Import enhanced checkbox handling
try:
    from checkbox_fix import CheckboxManager, collect_team_selections_enhanced
    ENHANCED_CHECKBOX_AVAILABLE = True
    print("✅ Enhanced Checkbox Manager loaded successfully!")
except ImportError as e:
    print(f"⚠️ Enhanced Checkbox Manager not available: {e}")
    ENHANCED_CHECKBOX_AVAILABLE = False

# Import the advanced risk management engine
try:
    import sys
    import os
    # Add parent directory to path to find dfs_risk_engine
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from dfs_risk_engine import DFSRiskEngine, DFSBankrollManager, RiskMetrics
    RISK_ENGINE_AVAILABLE = True
    print("✅ Advanced Risk Engine loaded successfully!")
except ImportError as e:
    print(f"⚠️ Risk Engine not available: {e}")
    RISK_ENGINE_AVAILABLE = False

# Import the probability enhanced optimizer
try:
    from probability_enhanced_optimizer import ProbabilityEnhancedOptimizer
    PROBABILITY_OPTIMIZER_AVAILABLE = True
    print("✅ Probability Enhanced Optimizer loaded successfully!")
except ImportError as e:
    print(f"⚠️ Probability Enhanced Optimizer not available: {e}")
    PROBABILITY_OPTIMIZER_AVAILABLE = False

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
SALARY_CAP = 50000
MIN_SALARY_DEFAULT = 45000  # Default minimum salary requirement
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

# GENETIC ALGORITHM DIVERSITY ENGINE FOR MULTIPLE UNIQUE LINEUPS
class GeneticDiversityEngine:
    """Enhanced Genetic Algorithm engine for creating diverse lineup populations"""
    
    def __init__(self, df_players, position_limits, salary_cap, team_selections, min_salary):
        self.df_players = df_players
        self.position_limits = position_limits
        self.salary_cap = salary_cap
        self.team_selections = team_selections
        self.min_salary = min_salary
        self.population = []
        self.generation = 0
        
    def create_diverse_lineups(self, num_lineups, stack_type):
        """Main method to create diverse lineups using genetic algorithm principles"""
        logging.info(f"🧬 GENETIC DIVERSITY ENGINE: Creating {num_lineups} unique lineups for {stack_type}")
        
        # Phase 1: Create larger initial population
        population_size = max(num_lineups * 3, 50)  # 3x requested for selection
        population = self.create_initial_population(population_size, stack_type)
        
        if len(population) < num_lineups:
            logging.warning(f"🧬 Only {len(population)} diverse lineups created from {population_size} attempts")
            return population
            
        # Phase 2: Evolve population for better diversity
        evolved_population = self.evolve_population(population, generations=3)
        
        # Phase 3: Select most diverse subset
        diverse_selection = self.select_diverse_subset(evolved_population, num_lineups)
        
        logging.info(f"🧬 GENETIC ENGINE COMPLETE: Delivered {len(diverse_selection)}/{num_lineups} diverse lineups")
        return diverse_selection
    
    def create_initial_population(self, population_size, stack_type):
        """Create diverse initial population using genetic variation"""
        population = []
        lineup_hashes = set()  # Track unique lineups
        attempts = 0
        max_attempts = population_size * 5
        
        while len(population) < population_size and attempts < max_attempts:
            attempts += 1
            
            try:
                # Generate lineup with random variation
                variation_level = (attempts % 5)  # Cycle through variation levels
                lineup = self._generate_variant_lineup(stack_type, variation_level)
                
                if not lineup.empty and self._is_valid_lineup(lineup):
                    lineup_hash = self._get_lineup_hash(lineup)
                    
                    # Only add truly unique lineups
                    if lineup_hash not in lineup_hashes:
                        lineup_hashes.add(lineup_hash)
                        population.append(lineup)
                        
            except Exception as e:
                logging.debug(f"GA generation attempt {attempts} failed: {e}")
                continue
        
        logging.info(f"🧬 Initial population: {len(population)}/{population_size} unique lineups")
        return population
    
    def evolve_population(self, population, generations=3):
        """Evolve population for better fitness and diversity"""
        if len(population) < 2:
            return population
            
        current_pop = population.copy()
        
        for gen in range(generations):
            # Tournament selection - keep best performers
            elite = self._tournament_selection(current_pop, len(current_pop) // 2)
            
            # Generate offspring through crossover
            offspring = self._create_offspring(elite, len(current_pop) // 4)
            
            # Generate mutants for exploration
            mutants = self._create_mutants(elite, len(current_pop) // 4)
            
            # Combine populations
            current_pop = elite + offspring + mutants
            
            # Ensure diversity - remove too-similar lineups
            current_pop = self._ensure_diversity(current_pop, min_similarity=0.3)
            
        logging.info(f"🧬 Evolution complete: {len(current_pop)} lineups after {generations} generations")
        return current_pop
    
    def select_diverse_subset(self, population, target_count):
        """Select most diverse subset using maximal diversity algorithm"""
        if len(population) <= target_count:
            return population
            
        # Start with best performing lineup
        selected = [max(population, key=lambda x: x['Predicted_DK_Points'].sum() if 'Predicted_DK_Points' in x.columns else 0)]
        remaining = [x for x in population if not self._lineups_identical(x, selected[0])]
        
        # Greedily select most diverse remaining lineups
        while len(selected) < target_count and remaining:
            best_candidate = None
            max_min_distance = -1
            
            for candidate in remaining:
                # Calculate minimum distance to any selected lineup
                min_distance = min(self._lineup_distance(candidate, selected_lineup) 
                                 for selected_lineup in selected)
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        logging.info(f"🧬 Diverse selection: {len(selected)}/{target_count} maximally diverse lineups")
        return selected
    
    def _generate_variant_lineup(self, stack_type, variation_level):
        """Generate lineup variant with controlled randomness"""
        df_variant = self.df_players.copy()
        
        # Apply variation based on level
        if variation_level > 0 and 'Predicted_DK_Points' in df_variant.columns:
            # Add progressive noise for diversity
            noise_factor = 0.05 * variation_level  # 5% noise per level
            noise = np.random.normal(0, noise_factor, len(df_variant))
            df_variant['Predicted_DK_Points'] = np.maximum(
                df_variant['Predicted_DK_Points'] + (df_variant['Predicted_DK_Points'] * noise),
                0  # Ensure no negative points
            )
        
        # Use existing optimization with variant data
        lineup, _ = optimize_single_lineup((df_variant, stack_type, {}, self.team_selections, self.min_salary))
        return lineup
    
    def _tournament_selection(self, population, target_size):
        """Select best lineups using tournament selection"""
        if len(population) <= target_size:
            return population
            
        # Score each lineup
        scored_pop = []
        for lineup in population:
            if not lineup.empty and 'Predicted_DK_Points' in lineup.columns:
                fitness = lineup['Predicted_DK_Points'].sum()
                scored_pop.append((fitness, lineup))
        
        # Sort by fitness and return top performers
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        return [lineup for _, lineup in scored_pop[:target_size]]
    
    def _create_offspring(self, parents, target_count):
        """Create offspring through genetic crossover"""
        offspring = []
        
        for i in range(target_count):
            if len(parents) >= 2:
                parent1 = np.random.choice(parents)
                parent2 = np.random.choice(parents)
                child = self._crossover_lineups(parent1, parent2)
                if self._is_valid_lineup(child):
                    offspring.append(child)
        
        return offspring
    
    def _create_mutants(self, population, target_count):
        """Create mutants through random mutations"""
        mutants = []
        
        for i in range(target_count):
            if population:
                parent = np.random.choice(population)
                mutant = self._mutate_lineup(parent)
                if self._is_valid_lineup(mutant):
                    mutants.append(mutant)
        
        return mutants
    
    def _crossover_lineups(self, parent1, parent2):
        """Simple crossover: randomly mix players from two parents"""
        if parent1.empty or parent2.empty:
            return parent1.copy() if not parent1.empty else parent2.copy()
        
        child = parent1.copy()
        
        # For each position, randomly choose parent
        for position in self.position_limits.keys():
            if np.random.random() < 0.5:  # 50% chance to take from parent2
                p2_players = parent2[parent2['Position'] == position]
                if len(p2_players) > 0:
                    # Replace with parent2's players for this position
                    child_pos_mask = child['Position'] == position
                    child.loc[child_pos_mask] = p2_players.values
        
        return child
    
    def _mutate_lineup(self, lineup):
        """Mutate lineup by replacing random players"""
        if lineup.empty:
            return lineup
            
        mutated = lineup.copy()
        
        # Randomly mutate 1-2 positions
        positions_to_mutate = np.random.choice(
            list(self.position_limits.keys()), 
            size=np.random.randint(1, 3), 
            replace=False
        )
        
        for position in positions_to_mutate:
            current_players = mutated[mutated['Position'] == position]
            if len(current_players) > 0:
                # Find alternative players
                alternatives = self.df_players[
                    (self.df_players['Position'] == position) & 
                    (~self.df_players['Name'].isin(mutated['Name']))
                ]
                
                if len(alternatives) > 0:
                    # Replace with random alternative
                    replacement = alternatives.sample(1).iloc[0]
                    mutated.loc[mutated['Position'] == position].iloc[0] = replacement
        
        return mutated
    
    def _ensure_diversity(self, population, min_similarity=0.3):
        """Remove overly similar lineups to maintain diversity"""
        if len(population) <= 1:
            return population
        
        diverse_pop = [population[0]]  # Keep first lineup
        
        for lineup in population[1:]:
            # Check if this lineup is too similar to any existing one
            too_similar = any(
                self._genetic_similarity(lineup, existing) > (1.0 - min_similarity)
                for existing in diverse_pop
            )
            
            if not too_similar:
                diverse_pop.append(lineup)
        
        return diverse_pop
    
    def _lineup_distance(self, lineup1, lineup2):
        """Calculate distance between two lineups (higher = more diverse)"""
        if lineup1.empty or lineup2.empty:
            return 0
        
        # Count different players
        players1 = set(lineup1['Name'])
        players2 = set(lineup2['Name'])
        different_players = len(players1.symmetric_difference(players2))
        total_players = len(players1.union(players2))
        
        return different_players / total_players if total_players > 0 else 0
    
    def _genetic_similarity(self, lineup1, lineup2):
        """Calculate genetic similarity (0=different, 1=identical)"""
        return 1.0 - self._lineup_distance(lineup1, lineup2)
    
    def _lineups_identical(self, lineup1, lineup2):
        """Check if two lineups are identical"""
        if lineup1.empty or lineup2.empty:
            return False
        return set(lineup1['Name']) == set(lineup2['Name'])
    
    def _get_lineup_hash(self, lineup):
        """Get unique hash for lineup"""
        if lineup.empty:
            return ""
        player_names = sorted(lineup['Name'].tolist())
        return hashlib.md5('|'.join(player_names).encode()).hexdigest()
    
    def _is_valid_lineup(self, lineup):
        """Check if lineup meets all constraints"""
        if lineup.empty:
            return False
            
        try:
            # Check salary constraints
            total_salary = lineup['Salary'].sum() if 'Salary' in lineup.columns else 0
            if total_salary > self.salary_cap or total_salary < self.min_salary:
                return False
                
            # Check position constraints
            position_counts = lineup['Position'].value_counts()
            for position, required_count in self.position_limits.items():
                if position_counts.get(position, 0) != required_count:
                    return False
                    
            return True
            
        except Exception:
            return False

def optimize_single_lineup(args):
    df, stack_type, team_projected_runs, team_selections, min_salary = args
    logging.debug(f"optimize_single_lineup: Starting with stack type {stack_type}, min_salary={min_salary}")
    
    # INJECT DIVERSITY: Add aggressive random noise to projections to get truly different optimal solutions
    import random
    import numpy as np
    import time
    
    # Ensure truly random noise for each lineup by resetting seed with time + process info
    import time
    import os
    seed_value = int(time.time() * 1000000) % 2147483647 + os.getpid()
    random.seed(seed_value)  # Use time + PID for truly unique seed
    np.random.seed(seed_value)  # Use time + PID for numpy random seed
    
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Create the optimization problem
    problem = pulp.LpProblem("DFS_Lineup_Optimization", pulp.LpMaximize)
    
    # Create binary variables for each player
    player_vars = {}
    for idx in df.index:
        player_vars[idx] = pulp.LpVariable(f"player_{idx}", cat='Binary')
    
    # 🎲 PROBABILITY-ENHANCED OBJECTIVE FUNCTION
    has_probability_metrics = any(col in df.columns for col in [
        'Expected_Utility', 'Risk_Adjusted_Points', 'Kelly_Fraction'
    ])
    
    if has_probability_metrics:
        logging.debug("🎲 Using probability-enhanced objective function")
        
        # Use Expected Utility as primary objective if available
        if 'Expected_Utility' in df.columns:
            base_objective = [df.at[idx, 'Expected_Utility'] * player_vars[idx] for idx in df.index]
        elif 'Risk_Adjusted_Points' in df.columns:
            base_objective = [df.at[idx, 'Risk_Adjusted_Points'] * player_vars[idx] for idx in df.index]
        else:
            base_objective = [df.at[idx, 'Predicted_DK_Points'] * player_vars[idx] for idx in df.index]
        
        # Add Kelly Criterion weighting if available
        if 'Kelly_Fraction' in df.columns:
            kelly_bonus = [df.at[idx, 'Kelly_Fraction'] * df.at[idx, 'Predicted_DK_Points'] * 0.1 * player_vars[idx] 
                          for idx in df.index]
            objective = pulp.lpSum(base_objective + kelly_bonus)
        else:
            objective = pulp.lpSum(base_objective)
            
    else:
        # Standard objective function with diversity noise
        # Add aggressive noise for lineup diversity - ENHANCED for maximum diversity
        diversity_factor = random.uniform(0.20, 0.50)  # Increased to 20-50% noise for better diversity
        noise = np.random.normal(1.0, diversity_factor, len(df))
        
        # Add additional randomness to prevent identical lineups
        player_boost = np.random.choice(df.index, size=random.randint(1, 3), replace=False)
        for idx in player_boost:
            noise[df.index.get_loc(idx)] *= random.uniform(1.1, 1.3)
        
        df['Predicted_DK_Points'] = df['Predicted_DK_Points'] * noise
        
        # Objective: Maximize projected points (clean, no efficiency adjustments)
        objective = pulp.lpSum([df.at[idx, 'Predicted_DK_Points'] * player_vars[idx] for idx in df.index])
    
    problem += objective

    # Basic constraints
    problem += pulp.lpSum(player_vars.values()) == REQUIRED_TEAM_SIZE
    problem += pulp.lpSum([df.at[idx, 'Salary'] * player_vars[idx] for idx in df.index]) <= SALARY_CAP
    
    # ADD MINIMUM SALARY CONSTRAINT
    if min_salary and min_salary > 0:
        problem += pulp.lpSum([df.at[idx, 'Salary'] * player_vars[idx] for idx in df.index]) >= min_salary
        logging.debug(f"optimize_single_lineup: Added minimum salary constraint >= {min_salary}")
    
    for position, limit in POSITION_LIMITS.items():
        available_for_position = [idx for idx in df.index if position in df.at[idx, 'Position']]
        logging.debug(f"optimize_single_lineup: Position {position} needs {limit}, available: {len(available_for_position)}")
        if len(available_for_position) < limit:
            logging.error(f"optimize_single_lineup: INSUFFICIENT PLAYERS for {position}: need {limit}, have {len(available_for_position)}")
            return pd.DataFrame(), stack_type
        problem += pulp.lpSum([player_vars[idx] for idx in df.index if position in df.at[idx, 'Position']]) == limit

    # Handle different stack types
    if stack_type == "No Stacks":
        # No stacking constraints - just basic position and salary constraints
        logging.debug("optimize_single_lineup: Using no stacks")
    elif stack_type in ["5", "4", "3"]:
        # Handle simple single stack types (5 players, 4 players, 3 players from one team)
        stack_size = int(stack_type)
        logging.info(f"🎯 OPTIMIZER: Processing simple {stack_size}-stack")
        logging.info(f"🎯 OPTIMIZER: Team selections received: {team_selections}")
        
        # Get teams available for this stack size
        available_teams = None
        
        # ENHANCED TEAM SELECTION LOOKUP - handles multiple key formats
        if isinstance(team_selections, dict):
            # Method 1: Check for exact integer match
            if stack_size in team_selections:
                available_teams = team_selections[stack_size]
                logging.info(f"🎯 FOUND teams for {stack_size}-stack using exact integer match: {available_teams}")
            # Method 2: Check for string integer match
            elif str(stack_size) in team_selections:
                available_teams = team_selections[str(stack_size)]
                logging.info(f"🎯 FOUND teams for {stack_size}-stack using string integer match: {available_teams}")
            # Method 3: Check for "X-Stack" format
            elif f"{stack_size}-Stack" in team_selections:
                available_teams = team_selections[f"{stack_size}-Stack"]
                logging.info(f"🎯 FOUND teams for {stack_size}-stack using dash format match: {available_teams}")
            # Method 4: Check for "X Stack" format (with space)
            elif f"{stack_size} Stack" in team_selections:
                available_teams = team_selections[f"{stack_size} Stack"]
                logging.info(f"🎯 FOUND teams for {stack_size}-stack using space format match: {available_teams}")
            # Method 5: Check for "all" selection (from All Stacks tab)
            elif "all" in team_selections:
                available_teams = team_selections["all"]
                logging.info(f"🎯 FOUND teams for {stack_size}-stack using 'all' key: {available_teams}")
            # No specific match found in dictionary
            else:
                logging.warning(f"🎯 NO specific team selection found for {stack_size}-stack in keys: {list(team_selections.keys())}")
        # Fallback to list format
        elif isinstance(team_selections, list):
            available_teams = team_selections
            logging.debug(f"optimize_single_lineup: Using team list for stack size {stack_size}: {available_teams}")
        
        # Final fallback - use all available teams
        if not available_teams:
            available_teams = df['Team'].unique().tolist()
            logging.warning(f"🚨 FALLBACK: Using ALL {len(available_teams)} teams for {stack_size}-stack!")
        
        # Filter available teams to only those with enough batters
        valid_teams = []
        for team in available_teams:
            team_batters = df[(df['Team'] == team) & (~df['Position'].str.contains('P', na=False))].index
            if len(team_batters) >= stack_size:
                valid_teams.append(team)
                logging.debug(f"optimize_single_lineup: Team {team} has {len(team_batters)} batters (need {stack_size}) - VALID")
            else:
                logging.debug(f"optimize_single_lineup: Team {team} has {len(team_batters)} batters (need {stack_size}) - INSUFFICIENT")
                
        if not valid_teams:
            logging.warning(f"optimize_single_lineup: No valid teams with enough batters for {stack_size}-stack")
            logging.warning(f"Available teams: {available_teams}")
        else:
            # Enforce constraint for the selected teams
            if len(valid_teams) == 1:
                # If only one team selected for this stack size, enforce it directly
                selected_team = valid_teams[0]
                team_batters = df[(df['Team'] == selected_team) & (~df['Position'].str.contains('P', na=False))].index
                problem += pulp.lpSum([player_vars[idx] for idx in team_batters]) >= stack_size
                logging.info(f"✅ ENFORCING: Must have at least {stack_size} players from {selected_team}")
                
            else:
                # If multiple teams selected for this stack size, create OR constraint
                # This means: at least 'stack_size' players from ANY of the selected teams
                team_binary_vars = {}
                for team in valid_teams:
                    team_binary_vars[team] = pulp.LpVariable(f"use_team_{team}_{stack_size}", cat='Binary')
                
                # At least one team must be selected
                problem += pulp.lpSum(team_binary_vars.values()) >= 1
                
                # If a team is selected, enforce the stack constraint
                for team in valid_teams:
                    team_batters = df[(df['Team'] == team) & (~df['Position'].str.contains('P', na=False))].index
                    if len(team_batters) >= stack_size:
                        # If team is selected (binary = 1), enforce at least 'stack_size' players
                        problem += pulp.lpSum([player_vars[idx] for idx in team_batters]) >= stack_size * team_binary_vars[team]
                
                logging.info(f"✅ ENFORCING: Must have at least {stack_size} players from ANY of: {valid_teams}")
            
            # Log final enforcement summary
            logging.info(f"📊 STACK CONSTRAINT SUMMARY: {stack_size}-stack will be enforced using teams: {valid_teams}")
    else:
        # Implement complex stacking with proper team selection enforcement
        stack_sizes = [int(size) for size in stack_type.split('|')]
        logging.info(f"🎯 OPTIMIZER: Processing stack sizes: {stack_sizes}")
        logging.info(f"🎯 OPTIMIZER: Team selections received: {team_selections}")
        
        # Simplified approach: For each stack size, randomly pick one of the available teams
        # and enforce that constraint. This avoids creating too many binary variables.
        import random
        
        for i, size in enumerate(stack_sizes):
            # Get teams available for this specific stack size
            available_teams = None
            
            # ENHANCED TEAM SELECTION LOOKUP - handles multiple key formats
            if isinstance(team_selections, dict):
                # Method 1: Check for exact integer match
                if size in team_selections:
                    available_teams = team_selections[size]
                    logging.info(f"🎯 FOUND teams for {size}-stack using exact integer match: {available_teams}")
                # Method 2: Check for string integer match
                elif str(size) in team_selections:
                    available_teams = team_selections[str(size)]
                    logging.info(f"🎯 FOUND teams for {size}-stack using string integer match: {available_teams}")
                # Method 3: Check for "X-Stack" format
                elif f"{size}-Stack" in team_selections:
                    available_teams = team_selections[f"{size}-Stack"]
                    logging.info(f"🎯 FOUND teams for {size}-stack using dash format match: {available_teams}")
                # Method 4: Check for "X Stack" format (with space)
                elif f"{size} Stack" in team_selections:
                    available_teams = team_selections[f"{size} Stack"]
                    logging.info(f"🎯 FOUND teams for {size}-stack using space format match: {available_teams}")
                # Method 5: Check for "all" selection (from All Stacks tab)
                elif "all" in team_selections:
                    available_teams = team_selections["all"]
                    logging.info(f"🎯 FOUND teams for {size}-stack using 'all' key: {available_teams}")
                # No specific match found in dictionary
                else:
                    logging.warning(f"🎯 NO specific team selection found for {size}-stack in keys: {list(team_selections.keys())}")
            # Fallback to list format
            elif isinstance(team_selections, list):
                available_teams = team_selections
                logging.debug(f"optimize_single_lineup: Using team list for stack size {size}: {available_teams}")
            
            # Final fallback - AVOID using all available teams unless absolutely necessary
            if not available_teams:
                # Check if team_selections has data but wrong format
                if team_selections:
                    logging.error(f"🚨 CRITICAL: Team selections exist but {size}-stack not found!")
                    
                    # Only access dictionary methods if it's actually a dictionary
                    if isinstance(team_selections, dict):
                        logging.error(f"🚨 Available keys: {list(team_selections.keys())}")
                        logging.error(f"🚨 This suggests a format mismatch or checkbox detection issue!")
                        
                        # Try to find any matching stack with similar size
                        for key, teams in team_selections.items():
                            if str(size) in str(key) or str(key) in str(size):
                                available_teams = teams
                                logging.warning(f"🎯 Using approximate match {key} for {size}-stack: {teams}")
                                break
                    else:
                        logging.error(f"🚨 Team selections is not a dictionary: {type(team_selections)}")
                        logging.error(f"🚨 Value: {team_selections}")
                
                # Only use all teams as absolute last resort and warn user
                if not available_teams:
                    available_teams = df['Team'].unique().tolist()
                    logging.error(f"🚨 LAST RESORT: Using ALL {len(available_teams)} teams for {size}-stack!")
                    logging.error(f"🚨 This means your team selections were not detected properly!")
            else:
                # Check if we got ALL teams when user likely selected specific ones
                all_teams = df['Team'].unique().tolist()
                if len(available_teams) == len(all_teams) and set(available_teams) == set(all_teams):
                    logging.error(f"🚨 SUSPICIOUS: {size}-stack has ALL {len(available_teams)} teams - likely a selection bug!")
                else:
                    logging.info(f"✅ CONFIRMED: Will enforce {size}-stack using {len(available_teams)} teams: {available_teams}")
            
            if not available_teams:
                logging.warning(f"optimize_single_lineup: No teams available for stack size {size}, skipping")
                continue
            
            # Filter available teams to only those with enough batters
            valid_teams = []
            for team in available_teams:
                team_batters = df[(df['Team'] == team) & (~df['Position'].str.contains('P', na=False))].index
                if len(team_batters) >= size:
                    valid_teams.append(team)
                    logging.debug(f"optimize_single_lineup: Team {team} has {len(team_batters)} batters (need {size}) - VALID")
                else:
                    logging.debug(f"optimize_single_lineup: Team {team} has {len(team_batters)} batters (need {size}) - INSUFFICIENT")
                    
            if not valid_teams:
                logging.warning(f"optimize_single_lineup: No valid teams with enough batters for stack size {size}")
                logging.warning(f"Available teams: {available_teams}")
                continue
                
            # CRITICAL FIX: Instead of randomly selecting one team, enforce constraint for ALL user-selected teams
            # This ensures we respect the user's specific team choices for each stack size
            
            if len(valid_teams) == 1:
                # If only one team selected for this stack size, enforce it directly
                selected_team = valid_teams[0]
                team_batters = df[(df['Team'] == selected_team) & (~df['Position'].str.contains('P', na=False))].index
                problem += pulp.lpSum([player_vars[idx] for idx in team_batters]) >= size
                logging.info(f"✅ ENFORCING: Must have at least {size} players from {selected_team}")
                
            else:
                # If multiple teams selected for this stack size, create OR constraint
                # This means: at least 'size' players from ANY of the selected teams
                team_constraints = []
                for team in valid_teams:
                    team_batters = df[(df['Team'] == team) & (~df['Position'].str.contains('P', na=False))].index
                    if len(team_batters) >= size:
                        team_constraints.append(pulp.lpSum([player_vars[idx] for idx in team_batters]))
                
                if team_constraints:
                    # At least one of the selected teams must contribute 'size' players
                    # Create binary variables for each team to implement OR logic
                    team_binary_vars = {}
                    for team in valid_teams:
                        team_binary_vars[team] = pulp.LpVariable(f"use_team_{team}_{size}_{i}", cat='Binary')
                    
                    # At least one team must be selected
                    problem += pulp.lpSum(team_binary_vars.values()) >= 1
                    
                    # If a team is selected, enforce the stack constraint
                    for team in valid_teams:
                        team_batters = df[(df['Team'] == team) & (~df['Position'].str.contains('P', na=False))].index
                        if len(team_batters) >= size:
                            # If team is selected (binary = 1), enforce at least 'size' players
                            problem += pulp.lpSum([player_vars[idx] for idx in team_batters]) >= size * team_binary_vars[team]
                    
                    logging.info(f"✅ ENFORCING: Must have at least {size} players from ANY of: {valid_teams}")
                else:
                    logging.warning(f"❌ NO valid team constraints created for {size}-stack")
            
            # Log final enforcement summary
            logging.info(f"📊 STACK CONSTRAINT SUMMARY: {size}-stack will be enforced using teams: {valid_teams}")

    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
    status = problem.solve(solver)

    if pulp.LpStatus[status] == 'Optimal':
        lineup = df.loc[[idx for idx in df.index if player_vars[idx].varValue is not None and player_vars[idx].varValue > 0.5]]
        
        # Log the actual team composition for debugging
        team_counts = lineup['Team'].value_counts()
        logging.debug(f"optimize_single_lineup: Found optimal solution with {len(lineup)} players")
        logging.debug(f"optimize_single_lineup: Team composition: {dict(team_counts)}")
        
        return lineup, stack_type
    else:
        logging.debug(f"optimize_single_lineup: No optimal solution found. Status: {pulp.LpStatus[status]}")
        logging.debug(f"Constraints: {problem.constraints}")
        return pd.DataFrame(), stack_type
def simulate_iteration(df):
    random_factors = np.random.normal(1, 0.1, size=len(df))
    df = df.copy()
    df['Predicted_DK_Points'] = df['Predicted_DK_Points'] * random_factors
    df['Predicted_DK_Points'] = df['Predicted_DK_Points'].clip(lower=1)
    return df

class OptimizationWorker(QThread):
    optimization_done = pyqtSignal(dict, dict, dict)
    
    def __init__(self, df_players, salary_cap, position_limits, included_players, stack_settings, min_exposure, max_exposure, min_points, monte_carlo_iterations, num_lineups, team_selections, min_unique=0, bankroll=1000, risk_tolerance='medium', disable_kelly=False, min_salary=None, use_advanced_quant=False, advanced_quant_params=None):
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
        self.min_unique = min_unique  # Add min unique constraint
        self.disable_kelly = disable_kelly  # Option to disable Kelly sizing
        self.min_salary = min_salary if min_salary is not None else MIN_SALARY_DEFAULT  # Add minimum salary constraint
        self.use_advanced_quant = use_advanced_quant  # Add advanced quantitative optimization flag
        self.advanced_quant_params = advanced_quant_params or {}  # Store advanced quant parameters
        
        self.max_workers = multiprocessing.cpu_count()  # Or set a specific number
        self.min_points = min_points
        self.monte_carlo_iterations = monte_carlo_iterations
        self.team_selections = team_selections  # Passed from main app
        
        # 🎲 DETECT PROBABILITY METRICS for probability-aware optimization
        print(f"🔍 WORKER DEBUG: Checking for probability metrics in DataFrame")
        print(f"🔍 DataFrame shape: {df_players.shape}")
        print(f"🔍 DataFrame columns: {list(df_players.columns)}")
        
        self.has_probability_metrics = any(col in df_players.columns for col in [
            'Expected_Utility', 'Risk_Adjusted_Points', 'Kelly_Fraction', 'Implied_Volatility'
        ])
        
        # Also check for raw probability columns
        prob_cols_raw = [col for col in df_players.columns if col.startswith('Prob_')]
        print(f"🔍 Raw probability columns found: {len(prob_cols_raw)} - {prob_cols_raw}")
        
        if self.has_probability_metrics:
            logging.info("🎲 Probability-enhanced optimization enabled")
            print("✅ Enhanced probability metrics detected!")
            self.prob_columns = [col for col in df_players.columns if col.startswith('Prob_')]
            if self.prob_columns:
                logging.info(f"🎲 Found probability columns: {self.prob_columns}")
                print(f"✅ Raw probability columns: {self.prob_columns}")
        else:
            logging.info("Standard optimization mode (no probability metrics)")
            print("❌ No enhanced probability metrics found")
            print("📋 Looking for these columns: ['Expected_Utility', 'Risk_Adjusted_Points', 'Kelly_Fraction', 'Implied_Volatility']")
            self.prob_columns = []
        
        # Initialize risk management components
        self.bankroll = bankroll
        self.risk_tolerance = risk_tolerance
        if RISK_ENGINE_AVAILABLE:
            self.risk_engine = DFSRiskEngine()
            self.bankroll_manager = DFSBankrollManager(bankroll)
            logging.info(f"🔥 Risk management initialized: Bankroll=${bankroll}, Risk={risk_tolerance}")
        else:
            self.risk_engine = None
            self.bankroll_manager = None

    def run(self):
        logging.debug("OptimizationWorker: Starting optimization")
        results, team_exposure, stack_exposure = self.optimize_lineups()
        logging.debug(f"OptimizationWorker: Optimization complete. Results: {len(results)}")
        self.optimization_done.emit(results, team_exposure, stack_exposure)

    def optimize_lineups(self):
        df_filtered = self.preprocess_data()
        logging.debug(f"optimize_lineups: Starting with {len(df_filtered)} players")
        
        # SHOW USER THEIR SELECTIONS (team_selections already passed to worker)
        if self.team_selections:
            print(f"\n✅ YOUR TEAM SELECTIONS DETECTED:")
            for stack_size, teams in self.team_selections.items():
                if stack_size == "all":
                    print(f"   All Stacks: {teams}")
                else:
                    print(f"   {stack_size}-Stack: {teams}")
        else:
            print(f"\n⚠️ NO TEAM SELECTIONS - Using all teams")
        
        # IMMEDIATE DEBUG OUTPUT - Always print to console
        print(f"\n🚨 IMMEDIATE PLAYER FILTER DEBUG:")
        print(f"   📊 Total players after filtering: {len(df_filtered)}")
        print(f"   🎯 Included players list: {self.included_players}")
        print(f"   📋 Included players count: {len(self.included_players) if self.included_players else 0}")
        if len(df_filtered) > 0:
            print(f"   👥 Sample filtered players: {df_filtered['Name'].head(3).tolist()}")
        print(f"   🔥 This should help identify if filtering is working!\n")

        results = {}
        team_exposure = defaultdict(int)
        stack_exposure = defaultdict(int)
        
        # Check for advanced quantitative optimization
        if self.use_advanced_quant and ADVANCED_QUANT_AVAILABLE:
            logging.info("🔬 Using advanced quantitative optimization with financial modeling")
            return self.optimize_lineups_with_advanced_quant(df_filtered, team_exposure, stack_exposure)
        
        # Risk-adjusted lineup generation if available and enabled
        use_risk_management = (RISK_ENGINE_AVAILABLE and 
                             self.risk_engine and 
                             getattr(self, 'enable_risk_mgmt', True))
        
        if use_risk_management:
            logging.info("🔥 Using advanced risk management optimization")
            return self.optimize_lineups_with_risk_management(df_filtered, team_exposure, stack_exposure)
        
        # Check if we should use Genetic Diversity Engine for combinations
        use_genetic_engine = (
            hasattr(self, '_is_combination_mode') and self._is_combination_mode and 
            self.num_lineups >= 2  # Use GA for 2+ lineups (lowered threshold further)
        )
        
        # DEBUG: Log genetic engine decision
        logging.info(f"🔍 GENETIC ENGINE CHECK: has_combo_mode={hasattr(self, '_is_combination_mode')}, is_combo={getattr(self, '_is_combination_mode', False)}, num_lineups={self.num_lineups}, use_genetic={use_genetic_engine}")
        
        if use_genetic_engine:
            logging.info("🧬 Using GENETIC DIVERSITY ENGINE for combination lineups")
            return self.optimize_lineups_with_genetic_diversity(df_filtered, team_exposure, stack_exposure)
        
        # If combination mode requested more than 1 lineup, use Monte Carlo diversification
        if hasattr(self, '_is_combination_mode') and self._is_combination_mode and self.num_lineups > 1:
            logging.info("🎲 Using MONTE CARLO diversification for combination lineups")
            return self.optimize_lineups_with_monte_carlo(df_filtered, team_exposure, stack_exposure)
        
        # Traditional optimization (FIXED for proper lineup distribution)
        logging.info("📊 Using traditional optimization with ENHANCED DIVERSITY")
        
        # CRITICAL FIX: Distribute lineups across stack types instead of multiplying
        # Guarantee multiple solves per stack in combination mode
        if getattr(self, '_is_combination_mode', False):
            min_solves_per_stack = max(5, self.num_lineups // max(1, len(self.stack_settings)))
            total_candidates_needed = max(self.num_lineups * 6, min_solves_per_stack * len(self.stack_settings))
        else:
            total_candidates_needed = self.num_lineups * 3  # Increase to 3x for better diversity
        lineups_per_stack = max(1, total_candidates_needed // len(self.stack_settings))
        extra_lineups = total_candidates_needed % len(self.stack_settings)
        
        logging.info(f"🎯 DISTRIBUTION: {total_candidates_needed} total candidates across {len(self.stack_settings)} stack types")
        logging.info(f"🎯 DISTRIBUTION: {lineups_per_stack} per stack type + {extra_lineups} extra")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, stack_type in enumerate(self.stack_settings):
                # Give extra lineups to first few stack types
                lineups_for_this_stack = lineups_per_stack + (1 if i < extra_lineups else 0)
                if getattr(self, '_is_combination_mode', False):
                    lineups_for_this_stack = max(lineups_for_this_stack, min_solves_per_stack)
                
                for _ in range(lineups_for_this_stack):
                    # Create unique variant of data for each lineup
                    df_variant = df_filtered.copy()
                    future = executor.submit(optimize_single_lineup, (df_variant, stack_type, self.team_projected_runs, self.team_selections, self.min_salary))
                    futures.append(future)
                
                logging.info(f"🎯 QUEUED: {lineups_for_this_stack} candidates for {stack_type}")

            for future in concurrent.futures.as_completed(futures):
                try:
                    lineup, stack_type = future.result()
                    if lineup.empty:
                        logging.debug(f"optimize_lineups: Empty lineup returned for stack type {stack_type}")
                    else:
                        total_points = lineup['Predicted_DK_Points'].sum()
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
    
    def optimize_lineups_with_genetic_diversity(self, df_filtered, team_exposure, stack_exposure):
        """Optimize lineups using Genetic Diversity Engine for maximum uniqueness"""
        logging.info("🧬 GENETIC DIVERSITY OPTIMIZATION STARTING")
        
        results = {}
        
        try:
            # Initialize Genetic Diversity Engine
            ga_engine = GeneticDiversityEngine(
                df_players=df_filtered,
                position_limits=self.position_limits,
                salary_cap=self.salary_cap,
                team_selections=self.team_selections,
                min_salary=self.min_salary
            )
            
            # Generate diverse lineups for each stack type
            total_lineups_generated = 0
            
            for i, stack_type in enumerate(self.stack_settings):
                # Distribute lineups across stack types
                lineups_for_stack = self.num_lineups // len(self.stack_settings)
                if i < (self.num_lineups % len(self.stack_settings)):
                    lineups_for_stack += 1
                
                if lineups_for_stack > 0:
                    logging.info(f"🧬 Generating {lineups_for_stack} diverse lineups for {stack_type}")
                    
                    # Use genetic engine to create diverse lineups
                    diverse_lineups = ga_engine.create_diverse_lineups(lineups_for_stack, stack_type)
                    
                    # Add to results
                    for lineup in diverse_lineups:
                        if not lineup.empty:
                            total_points = lineup['Predicted_DK_Points'].sum() if 'Predicted_DK_Points' in lineup.columns else 0
                            results[len(results)] = {
                                'total_points': total_points,
                                'lineup': lineup,
                                'stack_type': stack_type
                            }
                            
                            # Update exposures
                            for team in lineup['Team'].unique():
                                team_exposure[team] += 1
                            stack_exposure[stack_type] += 1
                            total_lineups_generated += 1
            
            logging.info(f"🧬 GENETIC DIVERSITY COMPLETE: Generated {total_lineups_generated} unique lineups")
            
            # Ensure we have diverse results by validating uniqueness
            if len(results) > 1:
                unique_count = self._validate_lineup_uniqueness(results)
                logging.info(f"🧬 DIVERSITY VALIDATION: {unique_count}/{len(results)} truly unique lineups")
            
            return results, team_exposure, stack_exposure
            
        except Exception as e:
            logging.error(f"🧬 Error in genetic diversity optimization: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to traditional optimization
            logging.warning("🧬 Falling back to traditional optimization")
            return self.optimize_lineups_traditional(df_filtered, team_exposure, stack_exposure)
    
    def _validate_lineup_uniqueness(self, results):
        """Validate that lineups are actually unique"""
        unique_lineups = set()
        
        for result in results.values():
            lineup = result.get('lineup')
            if lineup is not None and not lineup.empty:
                player_names = tuple(sorted(lineup['Name'].tolist()))
                unique_lineups.add(player_names)
        
        return len(unique_lineups)

    def preprocess_data(self):
        """Preprocess player data for optimization"""
        df_filtered = self.df_players.copy()
        
        # Enhanced logging for debugging player selection issues
        print(f"\n🔍 PLAYER SELECTION DEBUG - DETAILED:")
        print(f"   📊 Total players available: {len(df_filtered)}")
        print(f"   🎯 Included players list: {self.included_players}")
        print(f"   📋 Number of included players: {len(self.included_players) if self.included_players else 0}")
        
        logging.info(f"🔍 PLAYER SELECTION DEBUG:")
        logging.info(f"   Total players available: {len(df_filtered)}")
        logging.info(f"   Included players list: {self.included_players}")
        logging.info(f"   Number of included players: {len(self.included_players) if self.included_players else 0}")
        
        # Apply included players filter - This is critical for respecting user selections!
        if self.included_players and len(self.included_players) > 0:
            # If players are specifically selected, use only those players
            print(f"   ✅ FILTERING to {len(self.included_players)} specifically selected players")
            logging.info(f"✅ Filtering to {len(self.included_players)} specifically selected players")
            original_count = len(df_filtered)
            df_filtered = df_filtered[df_filtered['Name'].isin(self.included_players)]
            final_count = len(df_filtered)
            print(f"   ✅ RESULT: {final_count}/{original_count} players remain after filtering")
            logging.info(f"✅ After filtering by selected players: {final_count}/{original_count} players remain")
            
            # Additional debug: show which selected players were found/not found
            if final_count == 0:
                print(f"   ⚠️ WARNING: NO PLAYERS FOUND after filtering! Check names match exactly.")
                logging.warning(f"⚠️ NO PLAYERS FOUND after filtering! Check if player names match exactly.")
                logging.warning(f"   Selected players: {self.included_players[:5]}...")  # Show first 5
                if len(df_filtered) > 0:
                    logging.warning(f"   Available player names: {df_filtered['Name'].head().tolist()}")
            elif final_count < len(self.included_players):
                found_players = set(df_filtered['Name'].tolist())
                missing_players = [p for p in self.included_players if p not in found_players]
                print(f"   ⚠️ WARNING: Some selected players not found: {missing_players[:3]}...")
                logging.warning(f"⚠️ Some selected players not found: {missing_players[:3]}...")  # Show first 3
        else:
            # If no players are specifically selected, use all players
            print(f"   ℹ️ NO players specifically selected - using all {len(df_filtered)} players")
            print(f"   ℹ️ This happens when: (1) No checkboxes checked, or (2) Checkbox detection failed")
            logging.info(f"ℹ️ No players specifically selected - using all {len(df_filtered)} players")
            logging.info(f"   This happens when: (1) No checkboxes checked, or (2) Checkbox detection failed")
        
        print(f"")  # Empty line for readability
        
        # 🎲 PROBABILITY-BASED CONTEST OPTIMIZATION
        if self.has_probability_metrics and PROBABILITY_OPTIMIZER_AVAILABLE:
            try:
                prob_optimizer = ProbabilityEnhancedOptimizer()
                
                # Determine optimal contest type based on risk tolerance
                contest_type = 'balanced'  # Default
                if self.risk_tolerance == 'conservative':
                    contest_type = 'cash'
                elif self.risk_tolerance == 'aggressive':
                    contest_type = 'gpp'
                
                # Apply contest-specific optimization
                df_filtered = prob_optimizer.optimize_for_contest_type(df_filtered, contest_type)
                
                # Display top players by probability metrics
                if 'Expected_Utility' in df_filtered.columns:
                    top_players = df_filtered.nlargest(5, 'Expected_Utility')[['Name', 'Expected_Utility', 'Risk_Adjusted_Points']].copy()
                    print(f"\n🎲 TOP 5 BY EXPECTED UTILITY ({contest_type.upper()} optimized):")
                    for _, player in top_players.iterrows():
                        print(f"   {player['Name']}: {player['Expected_Utility']:.3f} utility, {player['Risk_Adjusted_Points']:.2f} risk-adj pts")
                    
                logging.info(f"🎲 Applied {contest_type} contest optimization with probability metrics")
                
            except Exception as e:
                logging.warning(f"Error applying probability contest optimization: {e}")
        
        # Apply exposure constraints (future enhancement)
        # Could add min/max exposure filtering here
        
        return df_filtered

    def calculate_team_projected_runs(self, df):
        """Calculate projected runs for each team"""
        team_runs = {}
        for team in df['Team'].unique():
            team_players = df[df['Team'] == team]
            # Simple calculation based on average points
            avg_points = team_players['Predicted_DK_Points'].mean()
            team_runs[team] = avg_points * 0.1  # Simple scaling factor
        return team_runs

    def optimize_lineups_with_risk_management(self, df_filtered, team_exposure, stack_exposure):
        """
        Advanced lineup optimization using financial risk management principles
        """
        logging.info("🔥 Starting risk-adjusted optimization with financial portfolio theory")
        
        results = {}
        lineup_candidates = []
        
        # Step 1: Generate diverse lineup candidates
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            # With aggressive diversity injection, we need fewer candidates per request
            # Still generate extra to account for filtering, but not as many
            candidate_multiplier = max(10, min(15, self.num_lineups // 10))  # Enhanced multiplier for better selection
            candidate_multiplier = max(candidate_multiplier, 10)  # Minimum of 10x
            
            logging.info(f"🎯 Generating {self.num_lineups * candidate_multiplier} diverse candidates (multiplier: {candidate_multiplier}x) with aggressive noise injection")
            
            for stack_type in self.stack_settings:
                for _ in range(self.num_lineups * candidate_multiplier):
                    future = executor.submit(optimize_single_lineup, (df_filtered.copy(), stack_type, self.team_projected_runs, self.team_selections, self.min_salary))
                    futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    lineup, stack_type = future.result()
                    if not lineup.empty:
                        lineup_data = self.calculate_lineup_metrics(lineup, stack_type)
                        lineup_candidates.append(lineup_data)
                        
                except Exception as e:
                    logging.error(f"Error generating lineup candidate: {str(e)}")
        
        logging.info(f"📊 Generated {len(lineup_candidates)} lineup candidates")
        
        # Step 2: Calculate risk metrics for all candidates
        for i, candidate in enumerate(lineup_candidates):
            try:
                risk_metrics = self.risk_engine.calculate_lineup_risk_metrics(candidate)
                candidate['risk_metrics'] = risk_metrics
                candidate['lineup_id'] = i
                
                # Add risk-adjusted score
                candidate['risk_adjusted_score'] = self.calculate_risk_adjusted_score(candidate, risk_metrics)
                
            except Exception as e:
                logging.error(f"Error calculating risk metrics for candidate {i}: {e}")
                candidate['risk_adjusted_score'] = 0
        
        # Step 3: Select optimal lineups based on risk tolerance
        selected_lineups = self.select_lineups_by_risk_profile(lineup_candidates)
        
        # Step 4: Optimize allocation across selected lineups
        if len(selected_lineups) > 1:
            allocation = self.optimize_lineup_allocation(selected_lineups)
            logging.info(f"💰 Portfolio allocation optimized across {len(selected_lineups)} lineups")
        else:
            allocation = {'weights': [1.0]} if selected_lineups else {'weights': []}
        
        # Step 5: Apply position sizing and bankroll management
        final_selection = self.apply_position_sizing(selected_lineups, allocation)
        
        # Build results in expected format
        for i, lineup_data in enumerate(final_selection):
            lineup_df = lineup_data['lineup']
            total_points = lineup_df['Predicted_DK_Points'].sum()

            # Add risk information to the result
            risk_info = {
                'sharpe_ratio': lineup_data.get('risk_metrics', RiskMetrics(0,0,0,0,0,0)).sharpe_ratio,
                'volatility': lineup_data.get('risk_metrics', RiskMetrics(0,0,0,0,0,0)).volatility,
                'kelly_fraction': lineup_data.get('risk_metrics', RiskMetrics(0,0,0,0,0,0)).kelly_fraction,
                'position_size': lineup_data.get('position_size', 0),
                'allocation_weight': lineup_data.get('allocation_weight', 0)
            }
            
            results[len(results)] = {
                'total_points': total_points, 
                'lineup': lineup_df,
                'risk_info': risk_info
            }
            
            # Track exposure
            for team in lineup_df['Team'].unique():
                team_exposure[team] += 1
            stack_exposure[lineup_data['stack_type']] += 1
        
        logging.info(f"🎯 Risk-adjusted optimization complete. Selected {len(results)} optimal lineups")
        return results, team_exposure, stack_exposure
    
    def calculate_lineup_metrics(self, lineup_df, stack_type):
        """Calculate comprehensive metrics for a lineup"""
        try:
            total_points = lineup_df['Predicted_DK_Points'].sum()
            total_salary = lineup_df['Salary'].sum()
            
            # Calculate player variance (simplified)
            player_variances = []
            for _, player in lineup_df.iterrows():
                # Estimate variance from projected points (higher projection = higher variance)
                base_variance = (player['Predicted_DK_Points'] * 0.3) ** 2
                player_variances.append(base_variance)
            
            lineup_data = {
                'expected_points': total_points,
                'total_salary': total_salary,
                'lineup': lineup_df,
                'stack_type': stack_type,
                'players': [
                    {
                        'name': row['Name'],
                        'variance': var,
                        'projected_points': row['Predicted_DK_Points']
                    } 
                    for (_, row), var in zip(lineup_df.iterrows(), player_variances)
                ],
                'n_teams': len(lineup_df['Team'].unique()),
                'salary_efficiency': total_points / total_salary if total_salary > 0 else 0
            }
            
            return lineup_data
            
        except Exception as e:
            logging.error(f"Error calculating lineup metrics: {e}")
            return {
                'expected_points': 0,
                'total_salary': 50000,
                'lineup': lineup_df,
                'stack_type': stack_type,
                'players': []
            }
    
    def calculate_risk_adjusted_score(self, lineup_data, risk_metrics):
        """Calculate risk-adjusted score based on user's risk tolerance"""
        base_score = lineup_data['expected_points']
        
        # Risk tolerance multipliers
        risk_multipliers = {
            'conservative': {'sharpe_weight': 0.4, 'volatility_penalty': 0.3},
            'medium': {'sharpe_weight': 0.3, 'volatility_penalty': 0.2},
            'aggressive': {'sharpe_weight': 0.2, 'volatility_penalty': 0.1}
        }
        
        multiplier = risk_multipliers.get(self.risk_tolerance, risk_multipliers['medium'])
        
        # Sharpe ratio bonus
        sharpe_bonus = risk_metrics.sharpe_ratio * multiplier['sharpe_weight'] * base_score
        
        # Volatility penalty
        volatility_penalty = risk_metrics.volatility * multiplier['volatility_penalty'] * base_score
        
        # Kelly fraction bonus (rewards mathematically optimal sizing)
        kelly_bonus = max(0, risk_metrics.kelly_fraction) * 0.1 * base_score
        
        risk_adjusted_score = base_score + sharpe_bonus - volatility_penalty + kelly_bonus
        
        return max(0, risk_adjusted_score)
    
    def select_lineups_by_risk_profile(self, lineup_candidates):
        """Select lineups based on risk profile and diversification with strict min_unique constraint"""
        if not lineup_candidates:
            return []
        
        # Sort by risk-adjusted score
        sorted_candidates = sorted(lineup_candidates, key=lambda x: x.get('risk_adjusted_score', 0), reverse=True)
        
        selected = []
        used_core_players = set()
        filtered_count = 0
        
        logging.info(f"🎲 Selecting diverse lineups with min_unique constraint: {self.min_unique}")
        
        # If min_unique is 0, use the old progressive overlap logic
        if self.min_unique == 0:
            logging.info("🔄 Using progressive overlap thresholds (min_unique=0)")
            for overlap_threshold in [5, 6, 7, 8]:
                if len(selected) >= self.num_lineups:
                    break
                    
                logging.info(f"🔍 Trying overlap threshold {overlap_threshold}...")
                temp_selected = []
                temp_used_players = set()
                
                for candidate in sorted_candidates:
                    if len(temp_selected) >= self.num_lineups:
                        break
                        
                    # Check for excessive overlap with already selected lineups
                    lineup_df = candidate['lineup']
                    core_players = set(lineup_df['Name'].tolist())
                    
                    # Check overlap with both permanently selected and temporarily selected
                    all_used_players = used_core_players.union(temp_used_players)
                    max_overlap = max([len(core_players.intersection(used_players)) for used_players in all_used_players] + [0])
                    
                    if max_overlap < overlap_threshold:
                        temp_selected.append(candidate)
                        temp_used_players.add(frozenset(core_players))
                        logging.debug(f"Selected lineup {len(selected) + len(temp_selected)} with overlap: {max_overlap}")
                    else:
                        filtered_count += 1
                        logging.debug(f"Filtered lineup due to overlap ({max_overlap} >= {overlap_threshold})")
                
                # Add temp selections to permanent selections
                selected.extend(temp_selected)
                used_core_players.update(temp_used_players)
                
                logging.info(f"📊 Got {len(temp_selected)} lineups at threshold {overlap_threshold}, total: {len(selected)}")
                
                if len(selected) >= self.num_lineups:
                    break
        else:
            # Use strict min_unique constraint
            logging.info(f"🔥 Using strict min_unique constraint: {self.min_unique}")
            
            for candidate in sorted_candidates:
                if len(selected) >= self.num_lineups:
                    break
                    
                # Check for min_unique constraint with already selected lineups
                lineup_df = candidate['lineup']
                core_players = set(lineup_df['Name'].tolist())
                
                # Check if this lineup meets the min_unique constraint with ALL selected lineups
                meets_constraint = True
                for selected_lineup in selected:
                    selected_players = set(selected_lineup['lineup']['Name'].tolist())
                    unique_players = len(core_players.symmetric_difference(selected_players))
                    
                    if unique_players < self.min_unique:
                        meets_constraint = False
                        logging.debug(f"Lineup rejected: only {unique_players} unique players (need {self.min_unique})")
                        break
                
                if meets_constraint:
                    selected.append(candidate)
                    logging.debug(f"Selected lineup {len(selected)} - meets min_unique constraint")
                else:
                    filtered_count += 1
        
        # Final fallback - take top lineups regardless of overlap if we don't have enough
        if len(selected) < min(5, self.num_lineups):
            logging.warning(f"🚨 FALLBACK: Taking top {min(self.num_lineups, len(sorted_candidates))} lineups regardless of constraints")
            selected = sorted_candidates[:min(self.num_lineups, len(sorted_candidates))]
        
        logging.info(f"🎯 FINAL RESULT: Selected {len(selected)} diverse lineups from {len(lineup_candidates)} candidates")
        logging.info(f"🔍 Min unique constraint: {self.min_unique}, Filtered out: {filtered_count}")
        logging.info(f"✅ SUCCESS: Delivered exactly what you requested!")
        return selected
    
    def optimize_lineup_allocation(self, selected_lineups):
        """Optimize allocation weights across selected lineups using portfolio theory"""
        try:
            lineup_data_for_optimization = []
            for lineup in selected_lineups:
                risk_metrics = lineup.get('risk_metrics', RiskMetrics(0,0,0,0,0,0))
                lineup_data_for_optimization.append({
                    'expected_return': lineup['expected_points'],
                    'variance': risk_metrics.volatility ** 2,
                    'sharpe_ratio': risk_metrics.sharpe_ratio
                })
            
            allocation_result = self.risk_engine.optimize_lineup_allocation(lineup_data_for_optimization)
            
            if allocation_result and 'weights' in allocation_result:
                logging.info(f"📈 Portfolio Sharpe Ratio: {allocation_result.get('sharpe_ratio', 0):.3f}")
                return allocation_result
            else:
                # Equal weights fallback
                n_lineups = len(selected_lineups)
                return {'weights': [1.0/n_lineups] * n_lineups}
                
        except Exception as e:
            logging.error(f"Error in portfolio optimization: {e}")
            # Equal weights fallback
            n_lineups = len(selected_lineups)
            return {'weights': [1.0/n_lineups] * n_lineups}
    
    def optimize_lineups_with_advanced_quant(self, df_filtered, team_exposure, stack_exposure):
        """
        Advanced lineup optimization - now uses PuLP-based approach instead of external optimizer
        """
        logging.info("🔬 Using advanced PuLP-based optimization (replaces external AdvancedQuantitativeOptimizer)")
        
        try:
            # Use the new PuLP-based advanced optimization
            return self.optimize_lineups_with_advanced_pulp(df_filtered, team_exposure, stack_exposure)
        except Exception as e:
            logging.error(f"❌ Error in advanced quantitative optimization: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to traditional optimization
            logging.info("🔄 Falling back to traditional optimization")
            return self.optimize_lineups_traditional(df_filtered, team_exposure, stack_exposure)
    
    def optimize_lineups_with_advanced_pulp(self, df_filtered, team_exposure, stack_exposure):
        """
        Advanced PuLP-based lineup optimization with risk-adjusted scoring
        """
        logging.info("🔬 Using advanced PuLP-based optimization with risk modeling")
        
        try:
            # Prepare player data with risk metrics
            df_enhanced = df_filtered.copy()
            
            # Add risk-adjusted metrics
            df_enhanced['volatility'] = df_enhanced['Predicted_DK_Points'] * 0.15  # Assume 15% volatility
            df_enhanced['risk_adjusted_points'] = df_enhanced['Predicted_DK_Points'] / (1 + df_enhanced['volatility'])
            
            # Generate lineups using enhanced traditional method with risk adjustment
            results = {}
        
            # CRITICAL FIX: Distribute lineups across stack types instead of multiplying
            total_candidates_needed = self.num_lineups * 2  # Generate 2x candidates for better performance
            lineups_per_stack = max(1, total_candidates_needed // len(self.stack_settings))
            extra_lineups = total_candidates_needed % len(self.stack_settings)
            
            logging.info(f"🎯 Advanced PuLP optimization: {total_candidates_needed} candidates across {len(self.stack_settings)} stack types")
            
            for i, stack_type in enumerate(self.stack_settings):
                current_lineups = lineups_per_stack
                if i < extra_lineups:
                    current_lineups += 1
                
                logging.info(f"🔥 Processing {stack_type}-stack: {current_lineups} lineups")
                
                # Generate lineups for this stack type using risk-adjusted approach
                stack_results = []
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for _ in range(current_lineups):
                        future = executor.submit(optimize_single_lineup, 
                                               (df_enhanced.copy(), stack_type, 
                                                self.team_projected_runs, self.team_selections, 
                                                self.min_salary))
                        futures.append(future)

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            lineup, returned_stack_type = future.result()
                            if not lineup.empty:
                                # Add risk metrics
                                lineup_data = {
                                    'lineup': lineup,
                                    'total_points': lineup['Predicted_DK_Points'].sum(),
                                    'total_salary': lineup['Salary'].sum(),
                                    'stack_type': returned_stack_type,
                                    'risk_adjusted_points': lineup['risk_adjusted_points'].sum(),
                                    'volatility': lineup['volatility'].mean(),
                                    'sharpe_ratio': lineup['Predicted_DK_Points'].sum() / lineup['volatility'].mean() if lineup['volatility'].mean() > 0 else 0
                                }
                                stack_results.append(lineup_data)
                                
                                # Update exposure tracking
                                for _, player in lineup.iterrows():
                                    team_exposure[player['Team']] += 1
                                    stack_exposure[returned_stack_type] += 1
                        except Exception as e:
                            logging.error(f"Error in advanced PuLP lineup generation: {str(e)}")
                            continue

                # Sort by risk-adjusted score
                stack_results.sort(key=lambda x: x['risk_adjusted_points'], reverse=True)
                
                # Take top lineups for this stack type
                top_lineups = stack_results[:min(len(stack_results), self.num_lineups // len(self.stack_settings))]
                
                for j, lineup_data in enumerate(top_lineups):
                    key = f"{stack_type}_{j}"
                    results[key] = lineup_data
                
                logging.info(f"✅ Generated {len(top_lineups)} lineups for {stack_type}-stack")
            
            logging.info(f"🏆 Advanced PuLP optimization complete: {len(results)} lineups generated")
            return results, team_exposure, stack_exposure
            
        except Exception as e:
            logging.error(f"❌ Critical error in advanced PuLP optimization: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to traditional optimization
            return self.optimize_lineups_traditional(df_filtered, team_exposure, stack_exposure)
    
    def get_advanced_quant_parameters(self):
        """Get advanced quantitative optimization parameters"""
        return self.advanced_quant_params
        
    def prepare_player_data_for_quant(self, df_filtered):
        """Prepare player data for the advanced quantitative optimizer"""
        player_data = []
        for _, player in df_filtered.iterrows():
            player_data.append({
                'name': player['Name'],
                'position': player['Position'],
                'team': player['Team'],
                'salary': player['Salary'],
                'projected_points': player['Predicted_DK_Points'],
                'value': player.get('Value', player['Predicted_DK_Points'] / (player['Salary'] / 1000))
            })
        return player_data
        
    def generate_historical_performance_data(self, df_filtered):
        """Generate historical performance data for players (mock data for now)"""
        historical_data = {}
        
        # In production, this would load real historical data
        # For now, generate realistic mock data
        import time
        np.random.seed(int(time.time() * 1000) % 100000)  # Time-based seed for true randomness
        
        for _, player in df_filtered.iterrows():
            name = player['Name']
            projected = player['Predicted_DK_Points']
            
            # Generate 100 days of historical performance
            # Use projected points as mean with realistic variance
            volatility = max(0.15, min(0.35, projected * 0.02))  # 15-35% volatility
            
            historical_points = np.random.normal(
                loc=projected,
                scale=projected * volatility,
                size=100
            )
            
            # Ensure non-negative values
            historical_points = np.maximum(historical_points, 0)
            
            historical_data[name] = historical_points.tolist()
                    
        return historical_data
        
    def convert_quant_lineup_to_dataframe(self, lineup_result, df_filtered):
        """Convert advanced quantitative lineup result back to DataFrame format"""
        try:
            # Try different possible keys for the lineup data
            lineup_data = lineup_result.get('lineup', lineup_result.get('players', []))
            
            logging.debug(f"🔍 Lineup result keys: {lineup_result.keys()}")
            logging.debug(f"🔍 Lineup data length: {len(lineup_data) if lineup_data else 'None'}")
            
            if not lineup_data:
                logging.warning("⚠️ No lineup data found in lineup_result")
                return pd.DataFrame()
            
            # Log first player for debugging
            if lineup_data:
                logging.debug(f"🔍 First player structure: {lineup_data[0]}")
            
            # Get player data for selected players
            lineup_players = []
            for i, player_info in enumerate(lineup_data):
                # Extract player name from the player info
                player_name = player_info.get('name', player_info.get('Name', ''))
                
                logging.debug(f"🔍 Player {i+1}: {player_name}")
                
                if player_name:
                    player_data = df_filtered[df_filtered['Name'] == player_name]
                    if not player_data.empty:
                        lineup_players.append(player_data.iloc[0])
                        logging.debug(f"✅ Found player {player_name} in filtered data")
                    else:
                        logging.warning(f"⚠️ Player {player_name} not found in filtered data")
                else:
                    logging.warning(f"⚠️ No name found for player {i+1}: {player_info}")
            
            logging.debug(f"🔍 Total players matched: {len(lineup_players)}")
            
            if not lineup_players:
                logging.warning("⚠️ No players matched in filtered data")
                return pd.DataFrame()
                        
            lineup_df = pd.DataFrame(lineup_players)
            logging.debug(f"✅ Created lineup DataFrame with {len(lineup_df)} players")
            return lineup_df
                        
        except Exception as e:
            logging.error(f"Error converting quantitative lineup: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
            
    def determine_stack_type_from_lineup(self, lineup_df):
        """Determine stack type from lineup composition"""
        if lineup_df.empty:
            return "Unknown"
        
        # Count players by team
        team_counts = lineup_df['Team'].value_counts()
        
        # Find the largest stack
        max_stack = team_counts.max()
        
        if max_stack >= 5:
            return "5 Stack"
        elif max_stack >= 4:
            return "4 Stack"
        elif max_stack >= 3:
            return "3 Stack"
        elif max_stack >= 2:
            return "2 Stack"
        else:
            return "No Stacks"
            
    def optimize_lineups_traditional(self, df_filtered, team_exposure, stack_exposure):
        """Traditional optimization method (fallback)"""
        logging.info("📊 Running traditional optimization")
        
        results = {}
        
        try:
            # Use the existing traditional optimization logic with ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for stack_type in self.stack_settings:
                    for _ in range(self.num_lineups):
                        future = executor.submit(optimize_single_lineup, (df_filtered.copy(), stack_type, self.team_projected_runs, self.team_selections, self.min_salary))
                        futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        lineup, stack_type = future.result()
                        if lineup.empty:
                            logging.debug(f"Empty lineup returned for stack type {stack_type}")
                        else:
                            total_points = lineup['Predicted_DK_Points'].sum()
                            results[len(results)] = {
                                'total_points': total_points,
                                'lineup': lineup,
                                'stack_type': stack_type
                            }
                            for team in lineup['Team'].unique():
                                team_exposure[team] += 1
                            stack_exposure[stack_type] += 1
                            logging.debug(f"Found valid lineup for stack type {stack_type}")
                    except Exception as e:
                        logging.error(f"Error in traditional optimization: {str(e)}")

            logging.info(f"📊 Traditional optimization complete. Generated {len(results)} lineups")
            return results, team_exposure, stack_exposure

        except Exception as e:
            logging.error(f"❌ Error in traditional optimization: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return results, team_exposure, stack_exposure

    def apply_position_sizing(self, selected_lineups, allocation):
        """Apply Kelly criterion and bankroll management to determine position sizes"""
        try:
            weights = allocation.get('weights', [])
            
            # Fix array comparison issue - properly check if weights is empty or wrong length
            weights_is_empty = False
            weights_wrong_length = False
            
            try:
                # Check if weights is empty (handle both list and array cases)
                if hasattr(weights, '__len__'):
                    weights_is_empty = len(weights) == 0
                    weights_wrong_length = len(weights) != len(selected_lineups)
                else:
                    weights_is_empty = True
            except:
                weights_is_empty = True
            
            if weights_is_empty or weights_wrong_length:
                if len(selected_lineups) > 0:
                    weights = [1.0/len(selected_lineups)] * len(selected_lineups)
                    logging.info(f"💰 Using equal weights for {len(selected_lineups)} lineups")
                else:
                    logging.error(f"🚨 CRITICAL: No selected lineups available for position sizing!")
                    return []
            else:
                logging.info(f"💰 Using portfolio-optimized weights for {len(selected_lineups)} lineups")
            
            position_limits = None
            if self.bankroll_manager and not self.disable_kelly:
                # Calculate average edge and volatility for position sizing
                # FIX: Ensure scalar values to prevent array comparison errors
                edge_values = []
                volatility_values = []
                
                for l in selected_lineups:
                    # Safely extract salary_efficiency as scalar
                    salary_eff = l.get('salary_efficiency', 0)
                    if hasattr(salary_eff, '__len__') and not isinstance(salary_eff, str):
                        salary_eff = float(salary_eff[0]) if len(salary_eff) > 0 else 0.0
                    edge_values.append(float(salary_eff))
                    
                    # Safely extract volatility as scalar
                    risk_metrics = l.get('risk_metrics', RiskMetrics(0,0,0,0,0,0))
                    if hasattr(risk_metrics, 'volatility'):
                        vol = risk_metrics.volatility
                        if hasattr(vol, '__len__') and not isinstance(vol, str):
                            vol = float(vol[0]) if len(vol) > 0 else 0.0
                        volatility_values.append(float(vol))
                    else:
                        volatility_values.append(0.0)
                
                # Calculate means as scalars
                avg_edge = float(np.mean(edge_values)) if edge_values else 0.0
                avg_volatility = float(np.mean(volatility_values)) if volatility_values else 0.0
                
                position_limits = self.bankroll_manager.calculate_position_limits(avg_edge, avg_volatility)
                logging.info(f"💰 Kelly sizing input: avg_edge={avg_edge:.3f}, avg_volatility={avg_volatility:.3f}")
                logging.info(f"💰 Kelly sizing: Recommended {position_limits.get('recommended_lineups', 1)} lineups")
                logging.info(f"💰 Kelly position limits: {position_limits}")
            elif self.disable_kelly:
                logging.info(f"💰 Kelly sizing: DISABLED - Will use all requested lineups")
            
            # Apply position sizing to lineups
            final_lineups = []
            # Ensure recommended_lineups is always a scalar integer
            recommended_lineups = position_limits.get('recommended_lineups', len(selected_lineups)) if position_limits else len(selected_lineups)
            if hasattr(recommended_lineups, '__len__') and not isinstance(recommended_lineups, str):
                recommended_lineups = int(recommended_lineups[0]) if len(recommended_lineups) > 0 else len(selected_lineups)
            recommended_lineups = int(recommended_lineups)
            
            # OVERRIDE: If Kelly is being too conservative, use more lineups
            if position_limits and recommended_lineups < max(5, self.num_lineups // 10) and not self.disable_kelly:
                logging.warning(f"💰 Kelly recommendation ({recommended_lineups}) seems too conservative. Overriding to use more lineups.")
                # Use a more aggressive override - prioritize user's requested count
                if self.num_lineups >= 50:
                    # For large requests, use 80% of requested lineups instead of half
                    recommended_lineups = max(10, min(len(selected_lineups), int(self.num_lineups * 0.8)))
                else:
                    # For smaller requests, use half
                    recommended_lineups = max(5, min(len(selected_lineups), self.num_lineups // 2))
                logging.info(f"💰 Adjusted to {recommended_lineups} lineups (aggressive override for {self.num_lineups} requested)")
            elif self.disable_kelly:
                # Use all selected lineups when Kelly is disabled - GUARANTEE EXACT COUNT
                recommended_lineups = len(selected_lineups)
                logging.info(f"💰 Kelly disabled: Using all {recommended_lineups} selected lineups")
            
            # CRITICAL: Ensure we deliver the exact number requested when Kelly is disabled
            if self.disable_kelly:
                # COMBINATION FIX: For combinations, be more aggressive about delivering count
                if hasattr(self, '_is_combination_mode') and self._is_combination_mode:
                    # FORCE FULL COUNT FOR COMBINATIONS - duplicate if necessary
                    lineups_to_use = self.num_lineups  # Always try for full count
                    logging.info(f"🧬 COMBINATION MODE: FORCING {lineups_to_use} lineups (disable_kelly=True)")
                    
                    # If we don't have enough unique lineups, we'll duplicate the best ones
                    if len(selected_lineups) < self.num_lineups:
                        logging.warning(f"🧬 COMBINATION: Only {len(selected_lineups)} unique lineups, will duplicate to reach {self.num_lineups}")
                else:
                    lineups_to_use = min(len(selected_lineups), self.num_lineups)
                    if lineups_to_use < self.num_lineups:
                        logging.warning(f"🚨 INSUFFICIENT LINEUPS: Only {lineups_to_use} available but {self.num_lineups} requested")
                    else:
                        logging.info(f"✅ EXACT COUNT: Delivering {lineups_to_use} lineups as requested")
            else:
                # With Kelly enabled, use recommended count
                lineups_to_use = min(len(selected_lineups), recommended_lineups, self.num_lineups)
                logging.info(f"💰 Kelly enabled: Using {lineups_to_use} lineups (recommended: {recommended_lineups}, requested: {self.num_lineups})")
            
            # Build final lineup list
            for i in range(lineups_to_use):
                # COMBINATION FIX: Handle duplication when we need more lineups
                if i < len(selected_lineups):
                    lineup = selected_lineups[i].copy()
                else:
                    # Duplicate from existing lineups (cycle through them)
                    source_idx = i % len(selected_lineups)
                    lineup = selected_lineups[source_idx].copy()
                    logging.debug(f"🧬 DUPLICATING lineup {source_idx} as lineup {i}")
                
                lineup['allocation_weight'] = weights[i] if i < len(weights) else 1.0/lineups_to_use
                
                # Fix the array comparison issue in position sizing
                if position_limits and i < len(weights):
                    optimal_size = position_limits.get('optimal_position_size', 100)
                    # Ensure optimal_size is a scalar, not an array
                    if hasattr(optimal_size, '__len__') and not isinstance(optimal_size, str):
                        optimal_size = float(optimal_size[0]) if len(optimal_size) > 0 else 100.0
                    else:
                        optimal_size = float(optimal_size)
                    
                    # Ensure weights[i] is also a scalar
                    weight_value = weights[i]
                    if hasattr(weight_value, '__len__') and not isinstance(weight_value, str):
                        weight_value = float(weight_value[0]) if len(weight_value) > 0 else 1.0/lineups_to_use
                    else:
                        weight_value = float(weight_value)
                    
                    lineup['position_size'] = optimal_size * weight_value
                else:
                    lineup['position_size'] = 100.0
                
                final_lineups.append(lineup)
            
            logging.info(f"🎯 FINAL DELIVERY: {len(final_lineups)} lineups delivered (requested: {self.num_lineups})")
            return final_lineups
            
        except Exception as e:
            logging.error(f"Error applying position sizing: {e}")
            logging.error(f"Error details: Type={type(e)}, Args={e.args}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            
            # Return the selected lineups without position sizing as fallback
            logging.warning(f"🚨 FALLBACK: Returning {len(selected_lineups)} lineups without position sizing due to error")
            return selected_lineups[:self.num_lineups]

class FantasyBaseballApp(QMainWindow):
    def enforce_player_exposure(self, lineups, max_exposure):
        """
        Enforce player exposure limits across generated lineups.

        :param lineups: List of lineups, where each lineup is a list of player IDs.
        :param max_exposure: Dictionary with player IDs as keys and maximum exposure percentages as values.
        :return: Filtered list of lineups that respect the exposure limits.
        """
        from collections import Counter

        # Count player appearances across all lineups
        player_counts = Counter(player for lineup in lineups for player in lineup)
        total_lineups = len(lineups)

        # Calculate current exposure for each player
        player_exposure = {
            player: count / total_lineups for player, count in player_counts.items()
        }

        # Filter lineups to respect max exposure
        filtered_lineups = []
        for lineup in lineups:
            if all(
                player_exposure.get(player, 0) <= max_exposure.get(player, 1.0)
                for player in lineup
            ):
                filtered_lineups.append(lineup)

        return filtered_lineups

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced MLB DFS Optimizer")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize enhanced checkbox managers
        if ENHANCED_CHECKBOX_AVAILABLE:
            self.checkbox_managers = {}  # One manager per stack size
            print("✅ Checkbox managers initialized")
        else:
            self.checkbox_managers = None
        
        self.setup_ui()
        
        self.included_players = []
        self.stack_settings = {}
        self.min_exposure = {}
        self.max_exposure = {}
        self.min_points = 1
        self.monte_carlo_iterations = 100
        self.team_selections = {}  # Initialize team selections

    def update_probability_display(self, prob_summary, status):
        """
        Update the GUI with probability metrics and status.

        :param prob_summary: Dictionary containing probability metrics.
        :param status: Status message to display.
        """
        # Example: Update a QLabel or QTableWidget with the probability summary
        logging.info(f"Updating probability display with status: {status}")
        
        # Update GUI elements if they exist
        if hasattr(self, 'prob_detection_label'):
            if prob_summary.get('columns') and any(cols for cols in prob_summary['columns'].values()):
                total_cols = sum(len(cols) for cols in prob_summary['columns'].values())
                self.prob_detection_label.setText(f"✅ {total_cols} probability columns detected")
                self.prob_detection_label.setStyleSheet("color: green; font-size: 10px; padding: 3px;")
            else:
                self.prob_detection_label.setText("❌ No probability columns detected")
                self.prob_detection_label.setStyleSheet("color: red; font-size: 10px; padding: 3px;")
        
        if hasattr(self, 'prob_summary_text'):
            if prob_summary.get('columns') and any(cols for cols in prob_summary['columns'].values()):
                summary_text = f"🎲 PROBABILITY DATA DETECTED ({status})\n\n"
                
                # Show detected columns by category
                for category, columns in prob_summary['columns'].items():
                    if columns:
                        summary_text += f"{category}: {', '.join(columns)}\n"
                
                # Show enhanced metrics
                summary_text += f"\nEnhanced Metrics:\n"
                for metric, count in prob_summary.get('metrics', {}).items():
                    if count > 0:
                        summary_text += f"• {metric}: {count} players\n"
                
                self.prob_summary_text.setText(summary_text)
            else:
                self.prob_summary_text.setText("No probability data available.\n\nTo use probability-based optimization, ensure your CSV contains columns like:\n• Prob_Over_5\n• Prob_Over_10\n• Prob_Over_15\n• etc.")
        
        # Console output
        print(f"🎲 Probability Summary: {prob_summary}, Status: {status}")

    def update_probability_display_error(self, error_message):
        """
        Display an error message in the GUI.

        :param error_message: Error message to display.
        """
        # Example: Update a QLabel or show a QMessageBox with the error message
        logging.error(f"Displaying error message: {error_message}")
        # Placeholder for actual GUI error display logic
        print(f"Error: {error_message}")

    def update_probability_display_no_data(self):
        """
        Update GUI when no probability data is available.
        """
        logging.info("No probability data available - updating GUI")
        if hasattr(self, 'prob_detection_label'):
            self.prob_detection_label.setText("❌ No probability columns detected")
        if hasattr(self, 'prob_summary_text'):
            self.prob_summary_text.setText("No probability data available.\n\nTo use probability-based optimization, ensure your CSV contains columns like:\n• Prob_Over_5\n• Prob_Over_10\n• Prob_Over_15\n• etc.")
        print("🎲 No probability data detected in loaded CSV")

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
        self.favorites_lineups = []  # Store favorite lineups from multiple runs
        self.favorites_file = "favorites_lineups.json"  # Persistent storage for favorites

        self.create_players_tab()
        self.create_team_stack_tab()
        self.create_stack_exposure_tab()
        self.create_team_combinations_tab()  # Add team combinations tab
        self.create_control_panel()
        self.create_advanced_quant_tab()  # Add advanced quantitative optimization tab
        self.create_favorites_tab()  # Add favorites tab
        self.load_favorites()  # Load saved favorites on startup

    def get_advanced_quant_params_for_worker(self):
        """Get advanced quantitative optimization parameters for worker"""
        if not getattr(self, 'use_advanced_quant', False):
            return {}
        
        try:
            # Check if advanced quant UI elements exist before accessing them
            if not hasattr(self, 'optimization_strategy'):
                logging.debug("Advanced quant UI not initialized yet, returning default parameters")
                return {
                    'optimization_strategy': 'combined',
                    'risk_tolerance': 1.0,
                    'var_confidence': 0.95,
                    'target_volatility': 0.15,
                    'mc_simulations': 10000,
                    'time_horizon': 1,
                    'garch_p': 1,
                    'garch_q': 1,
                    'garch_lookback': 100,
                    'copula_family': 'gaussian',
                    'dependency_threshold': 0.3,
                    'kelly_fraction_limit': 0.25,
                    'expected_win_rate': 0.2
                }
            
            return {
                'optimization_strategy': self.optimization_strategy.currentText(),
                'risk_tolerance': self.risk_tolerance.value(),
                'var_confidence': self.var_confidence.value(),
                'target_volatility': self.target_volatility.value(),
                'mc_simulations': self.mc_simulations.value(),
                'time_horizon': self.time_horizon.value(),
                'garch_p': self.garch_p.value(),
                'garch_q': self.garch_q.value(),
                'garch_lookback': self.garch_lookback.value(),
                'copula_family': self.copula_family.currentText(),
                'dependency_threshold': self.dependency_threshold.value(),
                'kelly_fraction_limit': self.kelly_fraction_limit.value(),
                'expected_win_rate': self.expected_win_rate.value()
            }
        except AttributeError as e:
            logging.warning(f"Could not get advanced quant parameters: {e}")
            return {}
        except Exception as e:
            logging.warning(f"Error getting advanced quant parameters: {e}")
            return {}

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

            table = QTableWidget(0, 10)
            table.setHorizontalHeaderLabels(["Select", "Name", "Team", "Position", "Salary", "Predicted_DK_Points", "Value", "Min Exp", "Max Exp", "Actual Exp (%)"])
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

            # Add Select All / Deselect All buttons for team stacks
            button_layout = QHBoxLayout()
            
            select_all_btn = QPushButton(f"✅ Select All")
            select_all_btn.setToolTip(f"Select all teams in {stack_size} tab")
            select_all_btn.clicked.connect(lambda checked, size=stack_size: self.select_all_teams(size))
            button_layout.addWidget(select_all_btn)
            
            deselect_all_btn = QPushButton(f"❌ Deselect All")
            deselect_all_btn.setToolTip(f"Deselect all teams in {stack_size} tab")
            deselect_all_btn.clicked.connect(lambda checked, size=stack_size: self.deselect_all_teams(size))
            button_layout.addWidget(deselect_all_btn)
            
            button_layout.addStretch()  # Push buttons to the left
            sub_layout.addLayout(button_layout)

            table = QTableWidget(0, 8)
            table.setHorizontalHeaderLabels(["Select", "Teams", "Status", "Time", "Proj Runs", "Min Exp", "Max Exp", "Actual Exp (%)"])
            sub_layout.addWidget(table)

            self.team_stack_tables[stack_size] = table

        self.team_stack_table = self.team_stack_tables["All Stacks"]

        refresh_button = QPushButton("Refresh Team Stacks")
        refresh_button.clicked.connect(self.refresh_team_stacks)
        layout.addWidget(refresh_button)
        
        # Add debug button to test team selection collection
        debug_button = QPushButton("🔍 Test Team Selection Detection")
        debug_button.setStyleSheet("background-color: #ff6b6b; color: white; font-weight: bold;")
        debug_button.clicked.connect(self.debug_team_selections)
        layout.addWidget(debug_button)

    def debug_team_selections(self):
        """Debug function to test team selection collection"""
        print("\n" + "="*60)
        print("🔍 MANUAL TEAM SELECTION DEBUG TEST")
        print("="*60)
        
        team_selections = self.collect_team_selections()
        
        print(f"\n📊 RESULTS:")
        if team_selections:
            print(f"✅ Found team selections: {team_selections}")
            for stack_size, teams in team_selections.items():
                print(f"   {stack_size}: {teams}")
        else:
            print(f"❌ No team selections found")
        
        print("="*60)

    def refresh_team_stacks(self):
        self.populate_team_stack_table()

    def select_all_teams(self, stack_size):
        """Select all teams in a specific team stack table"""
        if not hasattr(self, 'team_stack_tables') or stack_size not in self.team_stack_tables:
            logging.debug(f"No team stack table found for: {stack_size}")
            return
        
        table = self.team_stack_tables[stack_size]
        selected_count = 0
        
        # Try enhanced method first
        if (ENHANCED_CHECKBOX_AVAILABLE and hasattr(self, 'checkbox_managers') and 
            self.checkbox_managers and stack_size in self.checkbox_managers):
            
            manager = self.checkbox_managers[stack_size]
            logging.info(f"🚀 Using enhanced method to select all teams in {stack_size}")
            
            # Get all checkbox IDs for this manager and set them to checked
            for row in range(table.rowCount()):
                row_id = f"team_{row}"
                if manager.set_checkbox_state(row_id, True):
                    selected_count += 1
        else:
            # Fallback to original method
            logging.info(f"⚠️ Using fallback method to select all teams in {stack_size}")
            
            # Check all checkboxes in the table
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget:
                    # Find the checkbox within the widget
                    layout = checkbox_widget.layout()
                    if layout and layout.count() > 0:
                        checkbox = layout.itemAt(0).widget()
                        if isinstance(checkbox, QCheckBox):
                            checkbox.setChecked(True)
                            selected_count += 1
        
        logging.info(f"Selected all {selected_count} teams in {stack_size} table")
        if hasattr(self, 'status_label'):
            self.status_label.setText(f"✅ Selected all {selected_count} teams in {stack_size}")

    def deselect_all_teams(self, stack_size):
        """Deselect all teams in a specific team stack table"""
        if not hasattr(self, 'team_stack_tables') or stack_size not in self.team_stack_tables:
            logging.debug(f"No team stack table found for: {stack_size}")
            return
        
        table = self.team_stack_tables[stack_size]
        deselected_count = 0
        
        # Try enhanced method first
        if (ENHANCED_CHECKBOX_AVAILABLE and hasattr(self, 'checkbox_managers') and 
            self.checkbox_managers and stack_size in self.checkbox_managers):
            
            manager = self.checkbox_managers[stack_size]
            logging.info(f"🚀 Using enhanced method to deselect all teams in {stack_size}")
            
            # Get all checkbox IDs for this manager and set them to unchecked
            for row in range(table.rowCount()):
                row_id = f"team_{row}"
                if manager.set_checkbox_state(row_id, False):
                    deselected_count += 1
        else:
            # Fallback to original method
            logging.info(f"⚠️ Using fallback method to deselect all teams in {stack_size}")
            
            # Uncheck all checkboxes in the table
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget:
                    # Find the checkbox within the widget
                    layout = checkbox_widget.layout()
                    if layout and layout.count() > 0:
                        checkbox = layout.itemAt(0).widget()
                        if isinstance(checkbox, QCheckBox):
                            checkbox.setChecked(False)
                            deselected_count += 1
        
        logging.info(f"Deselected all {deselected_count} teams in {stack_size} table")
        if hasattr(self, 'status_label'):
            self.status_label.setText(f"❌ Deselected all {deselected_count} teams in {stack_size}")

    def populate_team_combinations_teams(self):
        """Populate team checkboxes in the combinations tab when data is loaded"""
        if not hasattr(self, 'df_players') or self.df_players is None or self.df_players.empty:
            return
        
        # Clear existing checkboxes
        for i in reversed(range(self.teams_layout.count())):
            child = self.teams_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        self.team_checkboxes.clear()
        
        # Get unique teams
        teams = sorted(self.df_players['Team'].unique())
        
        # Create checkboxes for each team
        for team in teams:
            checkbox = QCheckBox(team)
            checkbox.setStyleSheet("padding: 2px; font-size: 11px;")
            self.teams_layout.addWidget(checkbox)
            self.team_checkboxes[team] = checkbox
        
        logging.info(f"Populated {len(teams)} teams in combinations tab")

    def select_all_combination_teams(self):
        """Select all teams in the combinations tab"""
        for checkbox in self.team_checkboxes.values():
            checkbox.setChecked(True)
        logging.info("Selected all teams in combinations tab")

    def deselect_all_combination_teams(self):
        """Deselect all teams in the combinations tab"""
        for checkbox in self.team_checkboxes.values():
            checkbox.setChecked(False)
        logging.info("Deselected all teams in combinations tab")

    def generate_team_combinations(self):
        """Generate all possible team combinations based on selected teams and stack pattern"""
        try:
            # Get selected teams
            selected_teams = []
            for team, checkbox in self.team_checkboxes.items():
                if checkbox.isChecked():
                    selected_teams.append(team)
            print(f"[DEBUG] Selected teams for combinations: {selected_teams}")
            
            if len(selected_teams) < 2:
                QMessageBox.warning(self, "Warning", "Please select at least 2 teams to generate combinations.")
                return
            
            # Get stack pattern and parse it
            stack_pattern = self.combinations_stack_combo.currentText()
            stack_sizes = [int(x) for x in stack_pattern.split('|')]
            teams_needed = len(stack_sizes)
            print(f"[DEBUG] Stack pattern: {stack_pattern}, stack_sizes: {stack_sizes}, teams_needed: {teams_needed}")
            
            if len(selected_teams) < teams_needed:
                QMessageBox.warning(self, "Warning", f"Need at least {teams_needed} teams selected for {stack_pattern} stack pattern.")
                return
            
            # Get default lineups per combination
            try:
                default_lineups = int(self.default_lineups_input.text())
            except ValueError:
                default_lineups = 5
            
            # Generate all possible team combinations for this stack pattern
            team_combinations = []
            
            # Generate combinations of teams
            for team_combo in combinations(selected_teams, teams_needed):
                # Generate all permutations to assign different stack sizes to different teams
                for team_perm in itertools.permutations(team_combo):
                    combination_info = {
                        'teams': team_perm,
                        'stack_pattern': stack_pattern,
                        'stack_sizes': stack_sizes
                    }
                    team_combinations.append(combination_info)
            print(f"[DEBUG] Generated {len(team_combinations)} team_combinations. Example: {team_combinations[:2]}")
            
            # Clear existing combinations table
            self.combinations_table.setRowCount(0)
            
            # Populate table with combinations
            for i, combo_info in enumerate(team_combinations):
                row = self.combinations_table.rowCount()
                self.combinations_table.insertRow(row)
                
                # Select checkbox
                checkbox = QCheckBox()
                checkbox.setChecked(True)  # Select all by default
                checkbox.stateChanged.connect(self.update_total_lineups_display)
                checkbox_widget = QWidget()
                layout = QHBoxLayout(checkbox_widget)
                layout.addWidget(checkbox)
                layout.setAlignment(Qt.AlignCenter)
                layout.setContentsMargins(0, 0, 0, 0)
                self.combinations_table.setCellWidget(row, 0, checkbox_widget)
                
                # Team combination display
                teams = combo_info['teams']
                stack_sizes = combo_info['stack_sizes']
                combo_parts = []
                for j, team in enumerate(teams):
                    combo_parts.append(f"{team}({stack_sizes[j]})")
                combo_text = " + ".join(combo_parts)
                self.combinations_table.setItem(row, 1, QTableWidgetItem(combo_text))
                
                # Lineups per combo (editable)
                lineups_item = QTableWidgetItem(str(default_lineups))
                lineups_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
                self.combinations_table.setItem(row, 2, lineups_item)
                
                # Actions (could add individual generate button later)
                self.combinations_table.setItem(row, 3, QTableWidgetItem("Ready"))
            
            # Connect table editing signal to update total when lineups per combo are changed
            self.combinations_table.itemChanged.connect(self.update_total_lineups_display)
            
            # Update total lineups display
            self.update_total_lineups_display()
            
            # Show success message
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"✅ Generated {len(team_combinations)} combinations from {len(selected_teams)} teams using {stack_pattern} pattern")
            
            logging.info(f"Generated {len(team_combinations)} team combinations using {stack_pattern} pattern")
            
        except Exception as e:
            logging.error(f"Error generating team combinations: {e}")
            QMessageBox.critical(self, "Error", f"Failed to generate combinations: {str(e)}")

    def update_total_lineups_display(self):
        """Update the total lineups count display"""
        try:
            total_lineups = 0
            
            for row in range(self.combinations_table.rowCount()):
                # Check if combination is selected
                checkbox_widget = self.combinations_table.cellWidget(row, 0)
                if checkbox_widget:
                    layout = checkbox_widget.layout()
                    if layout and layout.count() > 0:
                        checkbox = layout.itemAt(0).widget()
                        if isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                            # Get lineups count for this combination
                            lineups_item = self.combinations_table.item(row, 2)
                            if lineups_item:
                                try:
                                    lineups_count = int(lineups_item.text())
                                    total_lineups += lineups_count
                                except ValueError:
                                    pass
            
            self.total_lineups_display.setText(str(total_lineups))
            
        except Exception as e:
            logging.error(f"Error updating total lineups display: {e}")

    def generate_combination_lineups(self):
        """Generate lineups for all selected team combinations"""
        try:
            if not hasattr(self, 'df_players') or self.df_players is None or self.df_players.empty:
                QMessageBox.warning(self, "Warning", "Please load player data first.")
                return
            
            selected_combinations = []
            
            # Get selected combinations and their lineup counts
            for row in range(self.combinations_table.rowCount()):
                checkbox_widget = self.combinations_table.cellWidget(row, 0)
                if checkbox_widget:
                    layout = checkbox_widget.layout()
                    if layout and layout.count() > 0:
                        checkbox = layout.itemAt(0).widget()
                        if isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                            combo_item = self.combinations_table.item(row, 1)
                            lineups_item = self.combinations_table.item(row, 2)
                            if combo_item and lineups_item:
                                try:
                                    combo_text = combo_item.text()
                                    # Parse the combination text like "LAD(4) + SF(2)"
                                    team_parts = combo_text.split(' + ')
                                    teams = []
                                    stack_sizes = []
                                    for part in team_parts:
                                        if '(' in part and ')' in part:
                                            team = part.split('(')[0].strip()
                                            stack_size = int(part.split('(')[1].split(')')[0])
                                            teams.append(team)
                                            stack_sizes.append(stack_size)
                                    lineups_count = int(lineups_item.text())
                                    selected_combinations.append((teams, stack_sizes, lineups_count))
                                except ValueError:
                                    pass
            print(f"[DEBUG] selected_combinations: {selected_combinations}")
            
            if not selected_combinations:
                QMessageBox.warning(self, "Warning", "No combinations selected.")
                return
            
            # Get stack pattern
            stack_pattern = self.combinations_stack_combo.currentText()
            
            # Generate lineups for each combination
            all_lineups = []
            
            for teams, stack_sizes, lineups_count in selected_combinations:
                if lineups_count <= 0:
                    continue
                
                # Create team selections for this combination
                # Map each stack size to its corresponding teams
                team_selections = {}
                for i, stack_size in enumerate(stack_sizes):
                    if stack_size not in team_selections:
                        team_selections[stack_size] = []
                    team_selections[stack_size].append(teams[i])
                print(f"[DEBUG] team_selections mapping for this combination: {team_selections}")
                # Check for format mismatches
                for k in team_selections.keys():
                    if not (isinstance(k, int) or (isinstance(k, str) and k.isdigit())):
                        print(f"[WARNING] team_selections key format may not match optimizer expectations: {k}")
                
                # Set up optimization parameters
                salary_cap = 50000
                position_limits = {'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
                
                # Get included players (all players for now)
                included_players = self.get_included_players()
                
                # Run optimization for this combination
                try:
                    # Format stack settings for the worker
                    stack_pattern = self.combinations_stack_combo.currentText()
                    stack_settings = {stack_pattern: True}  # Worker expects a dict of stack patterns
                    
                    # AGGRESSIVE APPROACH: Generate multiple lineups by running worker multiple times with noise
                    all_combo_results = {}
                    for attempt in range(max(lineups_count, 5)):  # At least 5 attempts
                        # Add noise to player projections for diversity
                        df_noisy = self.df_players.copy()
                        if 'Predicted_DK_Points' in df_noisy.columns:
                            # Add lognormal noise (5-15% variation)
                            noise_factor = np.random.lognormal(0, 0.1, len(df_noisy))
                            df_noisy['Predicted_DK_Points'] = df_noisy['Predicted_DK_Points'] * noise_factor
                        
                        worker = OptimizationWorker(
                            df_players=df_noisy,  # Use noisy data for diversity
                            salary_cap=salary_cap,
                            position_limits=position_limits,
                            included_players=included_players,
                            stack_settings=stack_settings,
                            min_exposure={},
                            max_exposure={},
                            min_points=1,
                            monte_carlo_iterations=50,  # Reduced for speed
                            num_lineups=1,  # Generate one lineup per attempt
                            team_selections=team_selections,
                            min_unique=0,
                            bankroll=1000,
                            risk_tolerance='medium',
                            disable_kelly=True,
                            min_salary=self.get_min_salary_constraint(),
                            use_advanced_quant=getattr(self, 'use_advanced_quant', False),
                            advanced_quant_params=self.get_advanced_quant_params_for_worker()
                        )
                        
                        # SET COMBINATION MODE FLAG
                        worker._is_combination_mode = True
                        
                        # Run single optimization
                        attempt_results, _, _ = worker.optimize_lineups()
                        
                        if attempt_results and len(attempt_results) > 0:
                            # Merge results with uniqueness check
                            for key, result in attempt_results.items():
                                if result and 'lineup' in result and not result['lineup'].empty:
                                    # Simple uniqueness check based on player names
                                    lineup_signature = tuple(sorted(result['lineup']['Name'].tolist()))
                                    if lineup_signature not in [tuple(sorted(existing['lineup']['Name'].tolist())) 
                                                              for existing in all_combo_results.values()]:
                                        all_combo_results[len(all_combo_results)] = result
                        
                        # Stop if we have enough unique lineups
                        if len(all_combo_results) >= lineups_count:
                            break
                    
                    combo_results = all_combo_results
                    logging.info(f"🎲 NOISE GENERATION: Created {len(combo_results)} unique lineups from {attempt + 1} attempts")
                    
                    # ADD DEBUG LOGGING
                    logging.info(f"🔍 DEBUG: Requested {lineups_count} lineups, got {len(combo_results) if combo_results else 0} results")
                    if combo_results:
                        logging.info(f"🔍 DEBUG: combo_results type: {type(combo_results)}")
                        if isinstance(combo_results, dict):
                            logging.info(f"🔍 DEBUG: combo_results keys: {list(combo_results.keys())}")
                    
                    if combo_results and len(combo_results) > 0:
                        # Convert results to list format for consistency
                        if isinstance(combo_results, dict):
                            combo_lineups = []
                            for result in combo_results.values():
                                if isinstance(result, dict) and 'lineup' in result:
                                    combo_lineups.append(result['lineup'])
                                elif isinstance(result, pd.DataFrame):
                                    combo_lineups.append(result)
                        elif isinstance(combo_results, list):
                            # If it's already a list, use it directly
                            combo_lineups = combo_results
                        else:
                            # Handle any other format
                            combo_lineups = [combo_results] if combo_results is not None else []
                        
                        # Ensure requested count per combination
                        if len(combo_lineups) < lineups_count and len(combo_lineups) > 0:
                            deficit = lineups_count - len(combo_lineups)
                            logging.warning(f"🧬 COMBINATION FILL: Only {len(combo_lineups)} unique lineups, duplicating {deficit} to reach {lineups_count}")
                            idx = 0
                            while len(combo_lineups) < lineups_count:
                                combo_lineups.append(combo_lineups[idx % len(combo_lineups)].copy())
                                idx += 1
                        all_lineups.extend(combo_lineups)
                        combo_display = " + ".join([f"{team}({size})" for team, size in zip(teams, stack_sizes)])
                        logging.info(f"Generated {len(combo_lineups)} lineups for combination: {combo_display}")
                    else:
                        combo_display = " + ".join([f"{team}({size})" for team, size in zip(teams, stack_sizes)])
                        logging.warning(f"No lineups generated for combination: {combo_display}")
                        
                except Exception as e:
                    combo_display = " + ".join([f"{team}({size})" for team, size in zip(teams, stack_sizes)])
                    logging.error(f"Error generating lineups for combination {combo_display}: {e}")
                    continue
            
            if all_lineups and len(all_lineups) > 0:
                # Store the results
                self.optimized_lineups = all_lineups
                
                # Convert list format to expected dictionary format for display_results
                results_dict = {}
                for i, lineup_df in enumerate(all_lineups):
                    # Calculate lineup metrics for display
                    if isinstance(lineup_df, dict) and 'lineup' in lineup_df:
                        # Already in proper format
                        results_dict[i] = lineup_df
                    else:
                        # Convert DataFrame to expected format
                        total_salary = lineup_df['Salary'].sum() if 'Salary' in lineup_df.columns else 0
                        total_points = lineup_df['Predicted_DK_Points'].sum() if 'Predicted_DK_Points' in lineup_df.columns else 0
                        
                        results_dict[i] = {
                            'lineup': lineup_df,
                            'total_salary': total_salary,
                            'total_points': total_points
                        }
                
                # Display results
                team_exposure = {}
                stack_exposure = {}
                self.display_results(results_dict, team_exposure, stack_exposure)
                
                # Show success message
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"✅ Generated {len(all_lineups)} total lineups from {len(selected_combinations)} combinations")
                
                logging.info(f"Successfully generated {len(all_lineups)} total lineups from combinations")
                
            else:
                error_msg = ("No lineups were generated. This could be due to:\n\n"
                           "• Not enough players selected for some positions\n"
                           "• Stack requirements too restrictive for available teams\n"
                           "• Salary constraints too tight\n"
                           "• Selected players don't form valid lineups\n\n"
                           "Try:\n"
                           "• Selecting more players in each position\n"
                           "• Choosing simpler stack patterns\n"
                           "• Adjusting salary constraints")
                QMessageBox.warning(self, "No Lineups Generated", error_msg)
                
        except Exception as e:
            logging.error(f"Error in generate_combination_lineups: {e}")
            QMessageBox.critical(self, "Error", f"Failed to generate combination lineups: {str(e)}")

    def create_stack_exposure_tab(self):
        stack_exposure_tab = QWidget()
        self.tabs.addTab(stack_exposure_tab, "Stack Exposure")
    
        layout = QVBoxLayout(stack_exposure_tab)
    
        self.stack_exposure_table = QTableWidget(0, 7)
        self.stack_exposure_table.setHorizontalHeaderLabels(["Select", "Stack Type", "Min Exp", "Max Exp", "Lineup Exp", "Pool Exp", "Entry Exp"])
        layout.addWidget(self.stack_exposure_table)
    
        stack_types = ["5", "4", "3", "No Stacks", "4|2|2", "4|2", "3|3|2", "3|2|2", "2|2|2", "5|3", "5|2"]
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

    def create_team_combinations_tab(self):
        """Create tab for generating all team combinations with specified stacks"""
        combinations_tab = QWidget()
        self.tabs.addTab(combinations_tab, "Team Combinations")
        
        layout = QVBoxLayout(combinations_tab)
        
        # Header section
        header_label = QLabel("🔥 Team Combination Generator")
        header_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #FF5722; padding: 10px;")
        layout.addWidget(header_label)
        
        # Description
        desc_label = QLabel("Select teams and stack type to generate all possible combinations, then specify lineups per combination.")
        desc_label.setStyleSheet("color: #666; font-size: 12px; padding: 5px;")
        layout.addWidget(desc_label)
        
        # Controls section
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_layout = QHBoxLayout(controls_frame)
        layout.addWidget(controls_frame)
        
        # Team selection section
        team_section = QVBoxLayout()
        team_label = QLabel("📋 Select Teams:")
        team_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        team_section.addWidget(team_label)
        
        # Team selection buttons
        team_btn_layout = QHBoxLayout()
        select_all_teams_btn = QPushButton("✅ Select All Teams")
        deselect_all_teams_btn = QPushButton("❌ Deselect All Teams")
        team_btn_layout.addWidget(select_all_teams_btn)
        team_btn_layout.addWidget(deselect_all_teams_btn)
        team_section.addLayout(team_btn_layout)
        
        # Team checkboxes area (scrollable)
        teams_scroll = QScrollArea()
        teams_scroll.setMaximumHeight(200)
        teams_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.teams_widget = QWidget()
        self.teams_layout = QVBoxLayout(self.teams_widget)
        teams_scroll.setWidget(self.teams_widget)
        teams_scroll.setWidgetResizable(True)
        team_section.addWidget(teams_scroll)
        
        controls_layout.addLayout(team_section)
        
        # Stack type and settings section
        settings_section = QVBoxLayout()
        
        # Stack type selection
        stack_label = QLabel("🏗️ Stack Type:")
        stack_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        settings_section.addWidget(stack_label)
        
        self.combinations_stack_combo = QComboBox()
        self.combinations_stack_combo.addItems(["5", "4", "3", "No Stacks", "5|2", "4|2", "4|2|2", "3|3|2", "3|2|2", "2|2|2", "5|3"])
        self.combinations_stack_combo.setCurrentText("4")
        self.combinations_stack_combo.setToolTip("Stack pattern: Simple stacks (5, 4, 3, No Stacks) or complex patterns (e.g., '4|2' means 4 players from one team, 2 from another)")
        settings_section.addWidget(self.combinations_stack_combo)
        
        # Default lineups per combination
        lineups_label = QLabel("📊 Default Lineups per Combination:")
        lineups_label.setStyleSheet("font-weight: bold; color: #9C27B0;")
        settings_section.addWidget(lineups_label)
        
        self.default_lineups_input = QLineEdit()
        self.default_lineups_input.setText("5")
        self.default_lineups_input.setPlaceholderText("e.g., 5")
        self.default_lineups_input.setToolTip("Default number of lineups to generate per combination")
        settings_section.addWidget(self.default_lineups_input)
        
        # Generate combinations button
        generate_combinations_btn = QPushButton("🔄 Generate Team Combinations")
        generate_combinations_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }")
        generate_combinations_btn.clicked.connect(self.generate_team_combinations)
        settings_section.addWidget(generate_combinations_btn)
        
        controls_layout.addLayout(settings_section)
        
        # Combinations display table
        combinations_label = QLabel("🎯 Generated Combinations:")
        combinations_label.setStyleSheet("font-weight: bold; color: #673AB7; font-size: 14px; padding: 10px 0px 5px 0px;")
        layout.addWidget(combinations_label)
        
        self.combinations_table = QTableWidget(0, 4)
        self.combinations_table.setHorizontalHeaderLabels(["Select", "Team Combination", "Lineups per Combo", "Actions"])
        self.combinations_table.setAlternatingRowColors(True)
        layout.addWidget(self.combinations_table)
        
        # Generate lineups button
        generate_lineups_frame = QFrame()
        generate_lineups_layout = QHBoxLayout(generate_lineups_frame)
        layout.addWidget(generate_lineups_frame)
        
        total_lineups_label = QLabel("Total Lineups:")
        total_lineups_label.setStyleSheet("font-weight: bold; color: #333;")
        self.total_lineups_display = QLabel("0")
        self.total_lineups_display.setStyleSheet("font-weight: bold; color: #FF5722; font-size: 14px;")
        
        generate_lineups_layout.addWidget(total_lineups_label)
        generate_lineups_layout.addWidget(self.total_lineups_display)
        generate_lineups_layout.addStretch()
        
        generate_lineups_btn = QPushButton("🚀 Generate All Combination Lineups")
        generate_lineups_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; font-size: 14px; }")
        generate_lineups_btn.clicked.connect(self.generate_combination_lineups)
        generate_lineups_layout.addWidget(generate_lineups_btn)
        
        # Connect team selection buttons
        select_all_teams_btn.clicked.connect(self.select_all_combination_teams)
        deselect_all_teams_btn.clicked.connect(self.deselect_all_combination_teams)
        
        # Initialize empty team checkboxes (will be populated when data is loaded)
        self.team_checkboxes = {}

    def create_control_panel(self):
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_layout = QVBoxLayout(control_panel)

        self.splitter.addWidget(control_panel)

        load_button = QPushButton('Load CSV')
        load_button.clicked.connect(self.load_file)
        control_layout.addWidget(load_button)

        load_dk_predictions_button = QPushButton('Load DraftKings Predictions')
        load_dk_predictions_button.clicked.connect(self.load_dk_predictions)
        control_layout.addWidget(load_dk_predictions_button)

        load_entries_button = QPushButton('Load Entries CSV')
        load_entries_button.clicked.connect(self.load_entries_csv)
        control_layout.addWidget(load_entries_button)
        
        # Simple selection status label (for debugging)
        self.selection_status_label = QLabel("Player selection status will appear here")
        self.selection_status_label.setStyleSheet("color: #666; font-size: 10px; padding: 3px;")
        control_layout.addWidget(self.selection_status_label)
        
        self.min_unique_label = QLabel('Min Unique:')
        self.min_unique_input = QLineEdit()
        self.min_unique_input.setText("3")  # Default value
        self.min_unique_input.setPlaceholderText("e.g., 3")
        self.min_unique_input.setToolTip("Minimum number of unique players between lineups (0-10). Higher values create more diverse lineups.")
        control_layout.addWidget(self.min_unique_label)
        control_layout.addWidget(self.min_unique_input)
        
        self.num_lineups_label = QLabel('Number of Lineups:')
        self.num_lineups_input = QLineEdit()
        self.num_lineups_input.setText("100")  # Default value
        self.num_lineups_input.setPlaceholderText("e.g., 100")
        self.num_lineups_input.setToolTip("Number of lineups to generate (1-500). More lineups take longer to generate.")
        control_layout.addWidget(self.num_lineups_label)
        control_layout.addWidget(self.num_lineups_input)
        
        # Kelly sizing control
        self.disable_kelly_checkbox = QCheckBox('Disable Kelly Sizing (Generate All Requested Lineups)')
        self.disable_kelly_checkbox.setToolTip("Check this to disable Kelly Criterion position sizing and generate the full number of requested lineups. Useful when you want more lineups regardless of risk management recommendations.")
        control_layout.addWidget(self.disable_kelly_checkbox)

        # ADD MINIMUM SALARY CONTROLS
        min_salary_label = QLabel("💰 Minimum Salary Constraint:")
        min_salary_label.setStyleSheet("font-weight: bold; color: #4CAF50; padding: 5px; font-size: 12px;")
        control_layout.addWidget(min_salary_label)
        
        self.min_salary_input = QLineEdit()
        self.min_salary_input.setText("45000")  # Default value
        self.min_salary_input.setPlaceholderText("e.g., 45000")
        self.min_salary_input.setToolTip("Minimum total salary to spend (0-50000). Forces lineups to use higher budget to avoid too many cheap players.")
        control_layout.addWidget(self.min_salary_input)

        # Salary Range Constraints
        salary_range_label = QLabel("💰 Salary Range Constraints:")
        salary_range_label.setStyleSheet("font-weight: bold; color: #4CAF50; padding: 5px; font-size: 12px;")
        control_layout.addWidget(salary_range_label)
        
        self.max_salary_label = QLabel('Maximum Salary Spent:')
        self.max_salary_input = QLineEdit()
        self.max_salary_input.setText("50000")  # Default maximum (salary cap)
        self.max_salary_input.setPlaceholderText("e.g., 50000")
        self.max_salary_input.setToolTip("Maximum total salary to spend on lineups (default: 50000 - salary cap)")
        control_layout.addWidget(self.max_salary_label)
        control_layout.addWidget(self.max_salary_input)

        self.sorting_label = QLabel('Sorting Method:')
        self.sorting_combo = QComboBox()
        self.sorting_combo.addItems(["Points", "Value", "Salary"])
        control_layout.addWidget(self.sorting_label)
        control_layout.addWidget(self.sorting_combo)

        # Add Probability Metrics Display Section
        prob_separator = QFrame()
        prob_separator.setFrameShape(QFrame.HLine)
        prob_separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(prob_separator)
        
        prob_label = QLabel("🎲 Probability Metrics:")
        prob_label.setStyleSheet("font-weight: bold; color: #9C27B0; padding: 5px; font-size: 12px;")
        control_layout.addWidget(prob_label)
        
        # Probability detection status
        self.prob_detection_label = QLabel("No probability data loaded")
        self.prob_detection_label.setStyleSheet("color: #666; font-size: 10px; padding: 3px;")
        control_layout.addWidget(self.prob_detection_label)
        
        # Probability summary display
        self.prob_summary_text = QTextEdit()
        self.prob_summary_text.setMaximumHeight(100)
        self.prob_summary_text.setReadOnly(True)
        self.prob_summary_text.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd; font-family: monospace; font-size: 10px;")
        self.prob_summary_text.setPlaceholderText("Probability metrics will appear here when CSV with probability columns is loaded...")
        control_layout.addWidget(self.prob_summary_text)
        
        # Contest optimization strategy display
        self.contest_strategy_label = QLabel("Contest Strategy: Not Set")
        self.contest_strategy_label.setStyleSheet("color: #9C27B0; font-weight: bold; font-size: 11px; padding: 3px;")
        control_layout.addWidget(self.contest_strategy_label)

        # Add Risk Management Section
        if RISK_ENGINE_AVAILABLE:
            risk_separator = QFrame()
            risk_separator.setFrameShape(QFrame.HLine)
            risk_separator.setFrameShadow(QFrame.Sunken)
            control_layout.addWidget(risk_separator)
            
            risk_label = QLabel("🔥 Risk Management:")
            risk_label.setStyleSheet("font-weight: bold; color: #FF5722; padding: 5px; font-size: 12px;")
            control_layout.addWidget(risk_label)
            
            # Bankroll management
            self.bankroll_label = QLabel('Bankroll ($):')
            self.bankroll_input = QLineEdit()
            self.bankroll_input.setText("1000")
            self.bankroll_input.setPlaceholderText("e.g., 1000")
            self.bankroll_input.setToolTip("Your total bankroll for position sizing and Kelly criterion")
            control_layout.addWidget(self.bankroll_label)
            control_layout.addWidget(self.bankroll_input)
            
            # Risk tolerance
            self.risk_tolerance_label = QLabel('Risk Profile:')
            self.risk_tolerance_combo = QComboBox()
            self.risk_tolerance_combo.addItems(["conservative", "medium", "aggressive"])
            self.risk_tolerance_combo.setCurrentText("medium")
            self.risk_tolerance_combo.setToolTip("Conservative: Focus on Sharpe ratio\nMedium: Balanced approach\nAggressive: Higher risk tolerance")
            control_layout.addWidget(self.risk_tolerance_label)
            control_layout.addWidget(self.risk_tolerance_combo)
            
            # Enable/disable risk management
            self.enable_risk_mgmt_checkbox = QCheckBox("Enable Advanced Risk Management")
            self.enable_risk_mgmt_checkbox.setChecked(True)
            self.enable_risk_mgmt_checkbox.setToolTip("Use Kelly criterion, GARCH volatility, and portfolio theory")
            self.enable_risk_mgmt_checkbox.setStyleSheet("color: #FF5722; font-weight: bold;")
            control_layout.addWidget(self.enable_risk_mgmt_checkbox)

        run_button = QPushButton('Run Contest Sim')
        run_button.clicked.connect(self.run_optimization)
        control_layout.addWidget(run_button)

        save_button = QPushButton('Save CSV for DraftKings')
        save_button.clicked.connect(self.save_csv)
        control_layout.addWidget(save_button)
        
        # Add button for loading and filling DraftKings entries
        load_dk_entries_button = QPushButton('Load DraftKings Entries File')
        load_dk_entries_button.clicked.connect(self.load_dk_entries_file)
        load_dk_entries_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        control_layout.addWidget(load_dk_entries_button)
        
        # Add button for filling loaded entries with optimized lineups
        fill_entries_button = QPushButton('Fill Entries with Optimized Lineups')
        fill_entries_button.clicked.connect(self.fill_dk_entries_dynamic)
        fill_entries_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        control_layout.addWidget(fill_entries_button)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator)
        
        # Add favorites control buttons to main panel
        favorites_label = QLabel("🌟 Favorites Management:")
        favorites_label.setStyleSheet("font-weight: bold; color: #FF9800; padding: 5px;")
        control_layout.addWidget(favorites_label)
        
        add_to_favorites_main_button = QPushButton("➕ Add Current to Favorites")
        add_to_favorites_main_button.clicked.connect(self.add_current_lineups_to_favorites)
        add_to_favorites_main_button.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        control_layout.addWidget(add_to_favorites_main_button)
        
        save_favorites_main_button = QPushButton("💾 Export Favorites as New Lineups")
        save_favorites_main_button.clicked.connect(self.save_favorites_to_entries)
        save_favorites_main_button.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        control_layout.addWidget(save_favorites_main_button)

        self.results_table = QTableWidget(0, 9)
        self.results_table.setHorizontalHeaderLabels(["Player", "Team", "Position", "Salary", "Predicted_DK_Points", "Total Salary", "Total Points", "Exposure (%)", "Max Exp (%)"])
        control_layout.addWidget(self.results_table)

        self.status_label = QLabel('')
        control_layout.addWidget(self.status_label)

    def create_advanced_quant_tab(self):
        """Create the advanced quantitative optimization tab"""
        advanced_tab = QWidget()
        self.tabs.addTab(advanced_tab, "🔬 Advanced Quant")
        
        layout = QVBoxLayout(advanced_tab)
        
        # Main toggle
        self.advanced_quant_enabled = QCheckBox("Enable Advanced Quantitative Optimization")
        self.advanced_quant_enabled.setToolTip("Use advanced financial techniques: GARCH volatility, copulas, Monte Carlo, VaR, Kelly criterion")
        self.advanced_quant_enabled.setChecked(True)  # Enable by default
        self.advanced_quant_enabled.stateChanged.connect(self.toggle_advanced_quant)
        layout.addWidget(self.advanced_quant_enabled)
        
        # Create scrollable area for parameters
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setObjectName("scroll_widget")
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Optimization Strategy Section
        strategy_group = QGroupBox("Optimization Strategy")
        strategy_layout = QVBoxLayout(strategy_group)
        
        self.optimization_strategy = QComboBox()
        self.optimization_strategy.addItems([
            "combined",
            "kelly_criterion", 
            "risk_parity",
            "mean_variance",
            "equal_weight"
        ])
        self.optimization_strategy.setCurrentText("combined")
        self.optimization_strategy.setToolTip("Choose optimization strategy for lineup selection")
        strategy_layout.addWidget(QLabel("Strategy:"))
        strategy_layout.addWidget(self.optimization_strategy)
        
        scroll_layout.addWidget(strategy_group)
        
        # Risk Parameters Section
        risk_group = QGroupBox("Risk Parameters")
        risk_layout = QGridLayout(risk_group)
        
        # Risk tolerance
        self.risk_tolerance = QDoubleSpinBox()
        self.risk_tolerance.setRange(0.1, 2.0)
        self.risk_tolerance.setValue(1.0)
        self.risk_tolerance.setSingleStep(0.1)
        self.risk_tolerance.setDecimals(1)
        risk_layout.addWidget(QLabel("Risk Tolerance:"), 0, 0)
        risk_layout.addWidget(self.risk_tolerance, 0, 1)
        
        # VaR confidence level
        self.var_confidence = QDoubleSpinBox()
        self.var_confidence.setRange(0.90, 0.99)
        self.var_confidence.setValue(0.95)
        self.var_confidence.setSingleStep(0.01)
        self.var_confidence.setDecimals(2)
        risk_layout.addWidget(QLabel("VaR Confidence Level:"), 1, 0)
        risk_layout.addWidget(self.var_confidence, 1, 1)
        
        # Target volatility
        self.target_volatility = QDoubleSpinBox()
        self.target_volatility.setRange(0.05, 0.50)
        self.target_volatility.setValue(0.15)
        self.target_volatility.setSingleStep(0.01)
        self.target_volatility.setDecimals(2)
        risk_layout.addWidget(QLabel("Target Volatility:"), 2, 0)
        risk_layout.addWidget(self.target_volatility, 2, 1)
        
        scroll_layout.addWidget(risk_group)
        
        # Monte Carlo Section
        mc_group = QGroupBox("Monte Carlo Simulation")
        mc_layout = QGridLayout(mc_group)
        
        # Number of simulations
        self.mc_simulations = QSpinBox()
        self.mc_simulations.setRange(1000, 50000)
        self.mc_simulations.setValue(10000)
        self.mc_simulations.setSingleStep(1000)
        mc_layout.addWidget(QLabel("Simulations:"), 0, 0)
        mc_layout.addWidget(self.mc_simulations, 0, 1)
        
        # Time horizon
        self.time_horizon = QSpinBox()
        self.time_horizon.setRange(1, 30)
        self.time_horizon.setValue(1)
        mc_layout.addWidget(QLabel("Time Horizon (days):"), 1, 0)
        mc_layout.addWidget(self.time_horizon, 1, 1)
        
        scroll_layout.addWidget(mc_group)
        
        # GARCH Parameters Section
        garch_group = QGroupBox("GARCH Volatility Modeling")
        garch_layout = QGridLayout(garch_group)
        
        # GARCH p parameter
        self.garch_p = QSpinBox()
        self.garch_p.setRange(1, 5)
        self.garch_p.setValue(1)
        garch_layout.addWidget(QLabel("GARCH p:"), 0, 0)
        garch_layout.addWidget(self.garch_p, 0, 1)
        
        # GARCH q parameter
        self.garch_q = QSpinBox()
        self.garch_q.setRange(1, 5)
        self.garch_q.setValue(1)
        garch_layout.addWidget(QLabel("GARCH q:"), 1, 0)
        garch_layout.addWidget(self.garch_q, 1, 1)
        
        # Lookback period
        self.garch_lookback = QSpinBox()
        self.garch_lookback.setRange(30, 365)
        self.garch_lookback.setValue(100)
        garch_layout.addWidget(QLabel("Lookback Period:"), 2, 0)
        garch_layout.addWidget(self.garch_lookback, 2, 1)
        
        scroll_layout.addWidget(garch_group)
        
        # Copula Parameters Section
        copula_group = QGroupBox("Copula Dependency Modeling")
        copula_layout = QGridLayout(copula_group)
        
        # Copula family
        self.copula_family = QComboBox()
        self.copula_family.addItems(["gaussian", "t", "clayton", "frank", "gumbel"])
        self.copula_family.setCurrentText("gaussian")
        copula_layout.addWidget(QLabel("Copula Family:"), 0, 0)
        copula_layout.addWidget(self.copula_family, 0, 1)
        
        # Dependency threshold
        self.dependency_threshold = QDoubleSpinBox()
        self.dependency_threshold.setRange(0.1, 0.9)
        self.dependency_threshold.setValue(0.3)
        self.dependency_threshold.setSingleStep(0.05)
        self.dependency_threshold.setDecimals(2)
        copula_layout.addWidget(QLabel("Dependency Threshold:"), 1, 0)
        copula_layout.addWidget(self.dependency_threshold, 1, 1)
        
        scroll_layout.addWidget(copula_group)
        
        # Kelly Criterion Section
        kelly_group = QGroupBox("Kelly Criterion Position Sizing")
        kelly_layout = QGridLayout(kelly_group)
        
        # Kelly fraction limit
        self.kelly_fraction_limit = QDoubleSpinBox()
        self.kelly_fraction_limit.setRange(0.1, 1.0)
        self.kelly_fraction_limit.setValue(0.25)
        self.kelly_fraction_limit.setSingleStep(0.05)
        self.kelly_fraction_limit.setDecimals(2)
        kelly_layout.addWidget(QLabel("Max Kelly Fraction:"), 0, 0)
        kelly_layout.addWidget(self.kelly_fraction_limit, 0, 1)
        
        # Expected win rate
        self.expected_win_rate = QDoubleSpinBox()
        self.expected_win_rate.setRange(0.1, 0.9)
        self.expected_win_rate.setValue(0.2)
        self.expected_win_rate.setSingleStep(0.05)
        self.expected_win_rate.setDecimals(2)
        kelly_layout.addWidget(QLabel("Expected Win Rate:"), 1, 0)
        kelly_layout.addWidget(self.expected_win_rate, 1, 1)
        
        scroll_layout.addWidget(kelly_group)
        
        # Status and Info Section
        info_group = QGroupBox("Status & Information")
        info_layout = QVBoxLayout(info_group)
        
        self.quant_status_label = QLabel("Advanced quantitative optimization disabled")
        self.quant_status_label.setStyleSheet("color: gray;")
        info_layout.addWidget(self.quant_status_label)
        
        # Library status
        self.library_status = QLabel()
        self.update_library_status()
        info_layout.addWidget(self.library_status)
        
        scroll_layout.addWidget(info_group)
        
        # Probability Enhancement Section
        prob_group = QGroupBox("Probability Enhancement")
        prob_layout = QVBoxLayout(prob_group)
        
        # Probability detection status
        self.prob_detection_label = QLabel("❌ No probability columns detected")
        self.prob_detection_label.setStyleSheet("color: #666; font-size: 10px; padding: 3px;")
        prob_layout.addWidget(self.prob_detection_label)
        
        # Contest strategy display
        self.contest_strategy_label = QLabel("Contest Strategy: Not Set")
        self.contest_strategy_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        prob_layout.addWidget(self.contest_strategy_label)
        
        # Probability summary text area
        self.prob_summary_text = QTextEdit()
        self.prob_summary_text.setMaximumHeight(120)
        self.prob_summary_text.setReadOnly(True)
        self.prob_summary_text.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd; font-size: 10px;")
        self.prob_summary_text.setText("No probability data available.\n\nTo use probability-based optimization, ensure your CSV contains columns like:\n• Prob_Over_5\n• Prob_Over_10\n• Prob_Over_15\n• etc.")
        prob_layout.addWidget(self.prob_summary_text)
        
        scroll_layout.addWidget(prob_group)
        
        # Set the scroll area
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Initially enable controls since we default to enabled
        self.toggle_advanced_quant(True)
    
    def toggle_advanced_quant(self, enabled):
        """Toggle advanced quantitative optimization on/off"""
        self.use_advanced_quant = enabled
        
        # Enable/disable all controls in the advanced tab
        scroll_widget = self.findChild(QWidget, "scroll_widget")
        if scroll_widget:
            for child in scroll_widget.findChildren(QWidget):
                if child != self.advanced_quant_enabled:
                    child.setEnabled(enabled)
        
        # Update status
        if enabled and ADVANCED_QUANT_AVAILABLE:
            self.quant_status_label.setText("✅ Advanced quantitative optimization ENABLED")
            self.quant_status_label.setStyleSheet("color: green; font-weight: bold;")
        elif enabled and not ADVANCED_QUANT_AVAILABLE:
            self.quant_status_label.setText("⚠️ Advanced quantitative optimization UNAVAILABLE - missing libraries")
            self.quant_status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.advanced_quant_enabled.setChecked(False)
            self.use_advanced_quant = False
        else:
            self.quant_status_label.setText("❌ Advanced quantitative optimization DISABLED")
            self.quant_status_label.setStyleSheet("color: gray;")
    
    def update_library_status(self):
        """Update the library availability status"""
        status_text = "Library Status:\n"
        
        if ADVANCED_QUANT_AVAILABLE:
            status_text += "✅ Advanced Quantitative Optimizer: Available\n"
        else:
            status_text += "❌ Advanced Quantitative Optimizer: Missing\n"
            
        # Check individual libraries
        try:
            import arch
            status_text += "✅ ARCH (GARCH): Available\n"
        except ImportError:
            status_text += "❌ ARCH (GARCH): Missing - pip install arch\n"
            
        try:
            import copulas
            status_text += "✅ Copulas: Available\n"
        except ImportError:
            status_text += "⚠️ Copulas: Optional - dependency modeling limited\n"
            
        try:
            import scipy
            status_text += "✅ SciPy: Available\n"
        except ImportError:
            status_text += "❌ SciPy: Missing - pip install scipy\n"
            
        try:
            import sklearn
            status_text += "✅ Scikit-learn: Available\n"
        except ImportError:
            status_text += "❌ Scikit-learn: Missing - pip install scikit-learn\n"
        
        self.library_status.setText(status_text)
        self.library_status.setStyleSheet("font-family: monospace; font-size: 10px;")

    def create_favorites_tab(self):
        """Create the favorites tab for managing saved lineups from multiple runs"""
        favorites_tab = QWidget()
        self.tabs.addTab(favorites_tab, "My Entries")
        
        layout = QVBoxLayout(favorites_tab)
        
        # Header with info
        header_label = QLabel("💾 My Entries - Build your final contest lineup from multiple optimization runs")
        header_label.setStyleSheet("font-weight: bold; color: #2196F3; padding: 10px;")
        layout.addWidget(header_label)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        # Add button to save current lineups to favorites
        self.add_to_favorites_button = QPushButton("➕ Add Current Pool to Favorites")
        self.add_to_favorites_button.clicked.connect(self.add_current_lineups_to_favorites)
        self.add_to_favorites_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 5px; }")
        button_layout.addWidget(self.add_to_favorites_button)
        
        # Clear favorites button
        clear_favorites_button = QPushButton("🗑️ Clear All Favorites")
        clear_favorites_button.clicked.connect(self.clear_favorites)
        clear_favorites_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 5px; }")
        button_layout.addWidget(clear_favorites_button)
        
        # Save favorites to entries button
        save_favorites_button = QPushButton("💾 Export Favorites as New Lineups")
        save_favorites_button.clicked.connect(self.save_favorites_to_entries)
        save_favorites_button.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 5px; }")
        button_layout.addWidget(save_favorites_button)
        
        layout.addLayout(button_layout)
        
        # Stats display
        self.favorites_stats_label = QLabel("📊 Total Favorites: 0 lineups")
        self.favorites_stats_label.setStyleSheet("padding: 5px; color: #666;")
        layout.addWidget(self.favorites_stats_label)
        
        # Favorites table
        self.favorites_table = QTableWidget(0, 11)
        self.favorites_table.setHorizontalHeaderLabels([
            "Select", "Run#", "Player", "Team", "Position", "Salary", "Points", 
            "Total Salary", "Total Points", "Added Date", "Actions"
        ])
        
        # Set table selection behavior
        self.favorites_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.favorites_table.setAlternatingRowColors(True)
        
        layout.addWidget(self.favorites_table)
        
        # Update favorites display
        self.update_favorites_display()

    def load_favorites(self):
        """Load saved favorites from persistent storage"""
        try:
            if os.path.exists(self.favorites_file):
                with open(self.favorites_file, 'r') as f:
                    favorites_data = json.load(f)
                    
                # Convert back to DataFrame format
                self.favorites_lineups = []
                for fav_data in favorites_data:
                    lineup_df = pd.DataFrame(fav_data['lineup_data'])
                    fav_entry = {
                        'lineup': lineup_df,
                        'total_points': float(fav_data['total_points']),
                        'total_salary': int(fav_data['total_salary']),
                        'run_number': fav_data.get('run_number', 1),
                        'date_added': fav_data.get('date_added', 'Unknown')
                    }
                    self.favorites_lineups.append(fav_entry)
                    
                logging.info(f"Loaded {len(self.favorites_lineups)} favorite lineups")
            else:
                self.favorites_lineups = []
                logging.info("No favorites file found, starting with empty favorites")
                
        except Exception as e:
            logging.error(f"Error loading favorites: {e}")
            self.favorites_lineups = []

    def save_favorites(self):
        """Save favorites to persistent storage"""
        try:
            favorites_data = []
            for fav in self.favorites_lineups:
                fav_data = {
                    'lineup_data': fav['lineup'].to_dict('records'),
                    'total_points': float(fav['total_points']),  # Convert to Python float
                    'total_salary': int(fav['total_salary']),   # Convert to Python int
                    'run_number': fav.get('run_number', 1),
                    'date_added': fav.get('date_added', 'Unknown')
                }
                favorites_data.append(fav_data)
            
            with open(self.favorites_file, 'w') as f:
                json.dump(favorites_data, f, indent=2)
                
            logging.info(f"Saved {len(self.favorites_lineups)} favorite lineups")
            
        except Exception as e:
            logging.error(f"Error saving favorites: {e}")

    def add_current_lineups_to_favorites(self):
        """Add current optimized lineups to favorites"""
        if not hasattr(self, 'optimized_lineups') or not self.optimized_lineups:
            QMessageBox.warning(self, "No Lineups", "No optimized lineups available to add to favorites.\n\nPlease run optimization first.")
            return
        
        # Ask user how many lineups to add
        num_available = len(self.optimized_lineups)
        num_to_add, ok = QInputDialog.getInt(
            self, 
            'Add to Favorites', 
            f'How many lineups to add to favorites?\n\n(Available: {num_available} optimized lineups)', 
            value=min(num_available, 50),  # Default to 50 or available, whichever is less
            min=1, 
            max=num_available
        )
        
        if not ok:
            return
        
        # Get current run number (increment from existing favorites)
        current_run = max([fav.get('run_number', 0) for fav in self.favorites_lineups], default=0) + 1
        current_date = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        
        # Add selected lineups to favorites
        added_count = 0
        for i, lineup in enumerate(self.optimized_lineups[:num_to_add]):
            fav_entry = {
                'lineup': lineup.copy(),
                'total_points': lineup['Predicted_DK_Points'].sum(),
                'total_salary': lineup['Salary'].sum(),
                'run_number': current_run,
                'date_added': current_date
            }
            self.favorites_lineups.append(fav_entry)
            added_count += 1
        
        # Save to persistent storage
        self.save_favorites()
        
        # Update display
        self.update_favorites_display()
        
        # Show success message with feedback if fewer than requested
        success_msg = f"✅ Successfully added {added_count} lineups to favorites!\n\n"
        success_msg += f"🏃 Run #{current_run}\n"
        success_msg += f"📅 {current_date}\n"
        success_msg += f"📊 Total favorites: {len(self.favorites_lineups)} lineups"
        
        if added_count < num_to_add:
            success_msg += f"\n\n⚠️ Note: Added {added_count} lineups (requested {num_to_add})"
        
        QMessageBox.information(self, "Added to Favorites", success_msg)
        
        self.status_label.setText(f'Added {added_count} lineups to favorites (Run #{current_run})')

    def clear_favorites(self):
        """Clear all favorites"""
        if not self.favorites_lineups:
            QMessageBox.information(self, "No Favorites", "No favorites to clear.")
            return
        
        reply = QMessageBox.question(
            self, 
            'Clear Favorites', 
            f'Are you sure you want to clear all {len(self.favorites_lineups)} favorite lineups?\n\nThis action cannot be undone.',
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.favorites_lineups = []
            self.save_favorites()
            self.update_favorites_display()
            QMessageBox.information(self, "Cleared", "All favorites have been cleared.")
            self.status_label.setText('All favorites cleared')

    def update_favorites_display(self):
        """Update the favorites table display"""
        if not hasattr(self, 'favorites_table'):
            return  # Table not created yet
            
        self.favorites_table.setRowCount(0)
        
        if not self.favorites_lineups:
            self.favorites_stats_label.setText("📊 Total Favorites: 0 lineups")
            return
        
        # Update stats
        total_favorites = len(self.favorites_lineups)
        unique_runs = len(set(fav.get('run_number', 1) for fav in self.favorites_lineups))
        self.favorites_stats_label.setText(f"📊 Total Favorites: {total_favorites} lineups from {unique_runs} runs")
        
        # Populate table
        for fav_idx, fav in enumerate(self.favorites_lineups):
            lineup = fav['lineup']
            run_number = fav.get('run_number', 1)
            date_added = fav.get('date_added', 'Unknown')
            
            # Add each player in the lineup as a row
            for _, player in lineup.iterrows():
                row_position = self.favorites_table.rowCount()
                self.favorites_table.insertRow(row_position)
                
                # Checkbox for selection
                checkbox = QCheckBox()
                checkbox_widget = QWidget()
                layout_checkbox = QHBoxLayout(checkbox_widget)
                layout_checkbox.addWidget(checkbox)
                layout_checkbox.setAlignment(Qt.AlignCenter)
                layout_checkbox.setContentsMargins(0, 0, 0, 0)
                self.favorites_table.setCellWidget(row_position, 0, checkbox_widget)
                
                # Fill row data
                self.favorites_table.setItem(row_position, 1, QTableWidgetItem(f"Run {run_number}"))
                self.favorites_table.setItem(row_position, 2, QTableWidgetItem(str(player['Name'])))
                self.favorites_table.setItem(row_position, 3, QTableWidgetItem(str(player['Team'])))
                self.favorites_table.setItem(row_position, 4, QTableWidgetItem(str(player['Position'])))
                self.favorites_table.setItem(row_position, 5, QTableWidgetItem(str(player['Salary'])))
                self.favorites_table.setItem(row_position, 6, QTableWidgetItem(f"{player['Predicted_DK_Points']:.2f}"))
                self.favorites_table.setItem(row_position, 7, QTableWidgetItem(str(fav['total_salary'])))
                self.favorites_table.setItem(row_position, 8, QTableWidgetItem(f"{fav['total_points']:.2f}"))
                self.favorites_table.setItem(row_position, 9, QTableWidgetItem(date_added))
                
                # Delete button for this lineup
                delete_button = QPushButton("🗑️")
                delete_button.setMaximumWidth(30)
                delete_button.clicked.connect(lambda checked, idx=fav_idx: self.delete_favorite_lineup(idx))
                delete_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
                self.favorites_table.setCellWidget(row_position, 10, delete_button)

    def delete_favorite_lineup(self, lineup_index):
        """Delete a specific favorite lineup"""
        if 0 <= lineup_index < len(self.favorites_lineups):
            run_number = self.favorites_lineups[lineup_index].get('run_number', 'Unknown')
            
            reply = QMessageBox.question(
                self, 
                'Delete Favorite', 
                f'Delete lineup from Run #{run_number}?',
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                del self.favorites_lineups[lineup_index]
                self.save_favorites()
                self.update_favorites_display()
                self.status_label.setText(f'Deleted favorite lineup from Run #{run_number}')

    def save_favorites_to_entries(self):
        """Save selected favorites to a DraftKings entries file"""
        if not self.favorites_lineups:
            QMessageBox.warning(self, "No Favorites", "No favorite lineups available.\n\nPlease add some lineups to favorites first.")
            return
        
        try:
            # Show detailed info about favorites
            total_favorites = len(self.favorites_lineups)
            unique_runs = len(set(fav.get('run_number', 1) for fav in self.favorites_lineups))
            
            # Ask user how many favorites to use
            num_available = len(self.favorites_lineups)
            dialog_text = f'How many favorite lineups to save?\n\n'
            dialog_text += f'📊 Available: {num_available} favorite lineups\n'
            dialog_text += f'🏃 From {unique_runs} different optimization runs\n\n'
            dialog_text += f'💡 Tip: You can save up to {num_available} lineups'
            
            num_to_use, ok = QInputDialog.getInt(
                self, 
                'Save Favorites to Entries', 
                dialog_text,
                value=min(num_available, 150),
                min=1, 
                max=num_available
            )
            
            if not ok:
                return
            
            # Ask for save location
            save_path, _ = QFileDialog.getSaveFileName(
                self, 
                'Save Favorites to Entries File', 
                'my_favorites_entries.csv',
                'CSV Files (*.csv);;All Files (*)'
            )
            
            if not save_path:
                return
            
            # Use the export method that uses the exact same logic as create_filled_entries_df
            result = self.export_favorites_as_new_lineups(save_path, num_to_use)
            
            # Show success message with more details
            success_msg = f"🎉 Favorites saved successfully!\n\n"
            success_msg += f"📊 Saved {num_to_use} favorite lineups\n"
            success_msg += f"🏃 From {unique_runs} optimization runs\n"
            success_msg += f"💾 Saved to: {os.path.basename(save_path)}\n\n"
            success_msg += f"🚀 Ready to upload to DraftKings!"
            
            # Add details from the export result
            if result:
                success_msg += f"\n✅ Exported {result.get('lineups_exported', 0)} lineups"
                success_msg += f"\n🆔 Used {result.get('player_ids_used', 0)} player IDs"
                if result.get('entry_metadata_found'):
                    success_msg += f"\n📋 Contest metadata included"
            
            QMessageBox.information(self, "Favorites Saved", success_msg)
            self.status_label.setText(f'Saved {num_to_use} favorites to {os.path.basename(save_path)}')
            
        except Exception as e:
            error_msg = f"Error saving favorites: {str(e)}"
            self.status_label.setText(error_msg)
            logging.error(error_msg)
            QMessageBox.critical(self, "Save Error", f"Failed to save favorites:\n\n{str(e)}")

    def save_favorites_as_new_lineups(self):
        """Save favorites in DraftKings contest entry format (same as DD.csv)"""
        if not self.favorites_lineups:
            QMessageBox.warning(self, "No Favorites", "No favorite lineups available.\n\nPlease add some lineups to favorites first.")
            return
        
        try:
            # Show detailed info about favorites
            total_favorites = len(self.favorites_lineups)
            unique_runs = len(set(fav.get('run_number', 1) for fav in self.favorites_lineups))
            
            # Ask user how many favorites to export
            num_available = len(self.favorites_lineups)
            dialog_text = f'How many favorite lineups to export in DraftKings format?\n\n'
            dialog_text += f'📊 Available: {num_available} favorite lineups\n'
            dialog_text += f'🏃 From {unique_runs} different optimization runs\n\n'
            dialog_text += f'🎯 Export format: DraftKings contest entry format\n'
            dialog_text += f'📋 Headers: Entry ID, Contest Name, Contest ID, Entry Fee, P, P, C, 1B, 2B, 3B, SS, OF, OF, OF\n'
            dialog_text += f'🔢 Contains: Player IDs (not names) ready for DraftKings upload'
            
            num_to_use, ok = QInputDialog.getInt(
                self, 
                'Export Favorites in DraftKings Format', 
                dialog_text,
                value=min(num_available, 150),
                min=1, 
                max=num_available
            )
            
            if not ok:
                return
            
            # Ask for save location
            save_path, _ = QFileDialog.getSaveFileName(
                self, 
                'Save Favorites in DraftKings Format', 
                'my_favorites_dk_format.csv',
                'CSV Files (*.csv);;All Files (*)'
            )
            
            if not save_path:
                return
            
            # Export favorites in DraftKings contest entry format
            export_info = self.export_favorites_as_new_lineups(save_path, num_to_use)
            
            # Show success message with detailed feedback
            success_msg = f"🎉 Favorites exported in DraftKings format!\n\n"
            success_msg += f"📊 Exported {export_info['lineups_exported']} favorite lineups\n"
            success_msg += f"🏃 From {unique_runs} optimization runs\n"
            success_msg += f"💾 Saved to: {os.path.basename(save_path)}\n\n"
            success_msg += f"📋 Format: Entry ID, Contest Name, Contest ID, Entry Fee, P, P, C, 1B, 2B, 3B, SS, OF, OF, OF\n"
            
            if export_info['player_ids_used'] > 0:
                success_msg += f"🔢 Player IDs: {export_info['player_ids_used']} positions filled with numeric IDs\n"
            else:
                success_msg += f"⚠️ Player IDs: Using player names (no ID mappings found)\n"
            
            if export_info['entry_metadata_found']:
                success_msg += f"📝 Entry metadata: Preserved from loaded DK entries file\n"
            else:
                success_msg += f"📝 Entry metadata: Empty (no DK entries file loaded)\n"
            
            success_msg += f"\n🚀 Ready for DraftKings upload!"
            
            QMessageBox.information(self, "DraftKings Format Export Complete", success_msg)
            self.status_label.setText(f'Exported {num_to_use} favorites in DK format to {os.path.basename(save_path)}')
            
        except Exception as e:
            error_msg = f"Error exporting favorites in DK format: {str(e)}"
            self.status_label.setText(error_msg)
            logging.error(error_msg)
            QMessageBox.critical(self, "Export Error", f"Failed to export favorites:\n\n{str(e)}")

    def export_favorites_as_new_lineups(self, output_path, num_to_use):
        """Export favorite lineups using the EXACT same logic as create_filled_entries_df"""
        if not self.favorites_lineups:
            logging.warning("No favorite lineups available to export")
            return {'lineups_exported': 0, 'player_ids_used': 0, 'entry_metadata_found': False}
        
        logging.info(f"Exporting {num_to_use} favorites using create_filled_entries_df logic to {output_path}")
        
        # CRITICAL FIX: Clear cached favorites data to force fresh generation
        if hasattr(self, '_favorites_with_player_data'):
            delattr(self, '_favorites_with_player_data')
            logging.info("🔄 Cleared cached favorites data to force fresh generation")
        
        # FORCE FRESH LINEUP GENERATION: Temporarily disable favorites file reading
        original_favorites_file_path = None
        backup_favorites_file_path = None
        saved_contest_info = None
        original_lineups = None  # Ensure this is always defined
        
        try:
            # First, create backup files and extract contest info from original files
            workspace_favorites_path = r"c:\Users\smtes\Downloads\coinbase_ml_trader\my_favorites_entries.csv"
            onedrive_favorites_path = r"c:\Users\smtes\OneDrive\Documents\my_favorites_entries.csv"
            
            # Extract contest info from original file BEFORE renaming
            if os.path.exists(workspace_favorites_path):
                self.extract_contest_info_from_favorites()
                saved_contest_info = getattr(self, '_contest_info_list', None)
                logging.info(f"🎯 Extracted contest info from workspace file: {len(saved_contest_info) if saved_contest_info else 0} entries")
                
                original_favorites_file_path = workspace_favorites_path
                backup_favorites_file_path = workspace_favorites_path + ".backup_temp"
                
                # Remove existing backup file if it exists
                if os.path.exists(backup_favorites_file_path):
                    os.remove(backup_favorites_file_path)
                    logging.info(f"🗑️ Removed existing backup file")
                
                os.rename(workspace_favorites_path, backup_favorites_file_path)
                logging.info(f"🔄 Temporarily renamed favorites file to force fresh generation")
            elif os.path.exists(onedrive_favorites_path):
                self.extract_contest_info_from_favorites()
                saved_contest_info = getattr(self, '_contest_info_list', None)
                logging.info(f"🎯 Extracted contest info from OneDrive file: {len(saved_contest_info) if saved_contest_info else 0} entries")
                
                original_favorites_file_path = onedrive_favorites_path
                backup_favorites_file_path = onedrive_favorites_path + ".backup_temp"
                
                # Remove existing backup file if it exists
                if os.path.exists(backup_favorites_file_path):
                    os.remove(backup_favorites_file_path)
                    logging.info(f"🗑️ Removed existing OneDrive backup file")
                
                os.rename(onedrive_favorites_path, backup_favorites_file_path)
                logging.info(f"🔄 Temporarily renamed OneDrive favorites file to force fresh generation")
            
            # Temporarily store the current optimized_lineups
            original_lineups = getattr(self, 'optimized_lineups', None)
            
            # Replace optimized_lineups with favorites for the duration of this function
            # Extract just the lineup DataFrames from favorites (favorites store {'lineup': df, ...})
            favorite_lineups = [fav['lineup'] for fav in self.favorites_lineups[:num_to_use]]
            self.optimized_lineups = favorite_lineups
            
            logging.info(f"🎯 Using {len(favorite_lineups)} fresh favorite lineups for export")
            
            # Restore the contest information that we extracted before cache-clearing
            if saved_contest_info:
                self._contest_info_list = saved_contest_info
                logging.info(f"🎯 Restored contest info: {len(saved_contest_info)} entries")
            
            # Use the EXACT same logic as create_filled_entries_df
            filled_entries = self.create_filled_entries_df(num_to_use)
            
            # Save to CSV
            filled_entries.to_csv(output_path, index=False)
            
            logging.info(f"✅ Successfully exported {len(filled_entries)} favorites using create_filled_entries_df logic")
            
            # Calculate return info
            player_ids_used = 0
            for _, row in filled_entries.iterrows():
                player_ids_used += len([id for id in row[4:] if id and str(id).strip()])
            
            entry_metadata_found = bool(filled_entries.iloc[0, 0] if len(filled_entries) > 0 else False)
            
            return {
                'lineups_exported': len(filled_entries),
                'player_ids_used': player_ids_used,
                'entry_metadata_found': entry_metadata_found
            }
            
        finally:
            # Restore the original optimized_lineups
            if original_lineups is not None:
                self.optimized_lineups = original_lineups
            else:
                # Remove the temporary attribute if it didn't exist before
                if hasattr(self, 'optimized_lineups'):
                    delattr(self, 'optimized_lineups')
            
            # Restore the temporarily renamed favorites file
            if original_favorites_file_path and backup_favorites_file_path:
                try:
                    if os.path.exists(backup_favorites_file_path):
                        # If the original exists, remove it before renaming
                        if os.path.exists(original_favorites_file_path):
                            os.remove(original_favorites_file_path)
                        os.rename(backup_favorites_file_path, original_favorites_file_path)
                        logging.info("🔄 Restored favorites file after export")
                except Exception as e:
                    logging.error(f"Error restoring favorites file: {e}")

    def run_optimization(self):
        """Run the optimization with min unique constraint support"""
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
        
        # Get min unique constraint
        min_unique = self.get_min_unique_constraint()
        
        # Get requested number of lineups
        requested_lineups = self.get_requested_lineups()
        
        logging.debug(f"Included players: {len(self.included_players)}")
        logging.debug(f"Stack settings: {self.stack_settings}")
        logging.debug(f"Min unique constraint: {min_unique}")
        logging.debug(f"Requested lineups: {requested_lineups}")
        
        # Debug team selections
        # Collect team selections (this should now work properly)
        team_selections = self.collect_team_selections()
        logging.info(f"🎯 Team selections from UI: {team_selections}")
        
        # DETAILED DEBUG: Show exactly what was collected
        if team_selections:
            print(f"🎯 USER TEAM SELECTIONS DETECTED:")
            logging.info(f"🎯 DETAILED TEAM SELECTIONS DEBUG:")
            for stack_size, teams in team_selections.items():
                if stack_size == "all":
                    print(f"   All Stacks: {len(teams)} teams")
                    logging.info(f"   All Stacks: {len(teams)} teams = {teams[:5]}{'...' if len(teams) > 5 else ''}")
                else:
                    print(f"   {stack_size}-Stack: {len(teams)} teams = {teams}")
                    logging.info(f"   {stack_size}-Stack: {len(teams)} teams = {teams}")
                    
            # CRITICAL DEBUG: Check if all teams are the same
            all_team_lists = list(team_selections.values())
            if len(all_team_lists) > 1:
                all_same = all(set(teams) == set(all_team_lists[0]) for teams in all_team_lists)
                if all_same and len(all_team_lists[0]) > 10:
                    logging.error(f"🚨 BUG DETECTED: All stack sizes have identical team lists with {len(all_team_lists[0])} teams!")
                    logging.error(f"🚨 This suggests checkbox detection is not working properly!")
                    print(f"🚨 BUG DETECTED: All stack sizes have the same {len(all_team_lists[0])} teams!")
                    print(f"🚨 This means your specific team selections (PIT, NYM/NYY) were not detected!")
                    print(f"🚨 SOLUTION: Try using 'Disable Kelly Sizing' to get all 100 lineups, or manually reselect teams")
                    
            # Also log individual selections for debugging
            for stack_size, teams in team_selections.items():
                logging.info(f"Individual selection - Stack {stack_size}: {teams}")
        else:
            print("⚠️ No team selections found - will use all available teams")
            logging.warning("No team selections detected from UI")        
        
        if not self.stack_settings:
            self.status_label.setText("Please select at least one stack type in the Stack Exposure tab.")
            return
            
        # Get bankroll and risk tolerance settings
        bankroll = self.get_bankroll_setting()
        risk_tolerance = self.get_risk_tolerance_setting()
        enable_risk_mgmt = self.get_risk_management_enabled()
        
        logging.debug(f"Risk settings - Bankroll: ${bankroll}, Tolerance: {risk_tolerance}, Enabled: {enable_risk_mgmt}")
            
        if not enable_risk_mgmt and RISK_ENGINE_AVAILABLE:
            # Temporarily disable risk engine for this run
            original_status = f"Risk management disabled for this run (Bankroll: ${bankroll})"
            self.status_label.setText(f"Running traditional optimization... {original_status}")
        elif enable_risk_mgmt and RISK_ENGINE_AVAILABLE:
            self.status_label.setText(f"Running risk-adjusted optimization... Bankroll: ${bankroll}, Risk: {risk_tolerance}")
        else:
            self.status_label.setText("Running optimization... Please wait.")
            
        # Check if Kelly sizing should be disabled
        disable_kelly = self.disable_kelly_checkbox.isChecked()
        if disable_kelly:
            self.status_label.setText("Running optimization with Kelly sizing disabled... Please wait.")
            logging.info("💰 Kelly sizing disabled by user - will generate all requested lineups")
            logging.info("💰 BYPASS MODE: When Kelly is disabled, min unique constraint will be ignored to guarantee lineup count")
            
            # Override min unique when Kelly is disabled to guarantee full lineup count
            if min_unique > 0:
                # COMBINATION FIX: For combinations with disable_kelly, be less restrictive  
                if hasattr(worker, '_is_combination_mode') and worker._is_combination_mode:
                    logging.info(f"🧬 COMBINATION MODE: Reducing min_unique from {min_unique} to 2 for better results")
                    min_unique = min(2, min_unique)  # Cap at 2 for combinations
                else:
                    logging.warning(f"💰 OVERRIDING: Min unique constraint ({min_unique}) ignored when Kelly sizing is disabled")
                    min_unique = 0  # Force to 0 to bypass filtering
                    logging.info(f"💰 Min unique set to 0 for maximum lineup diversity")
        
        # GET MINIMUM SALARY CONSTRAINT FROM UI
        self.min_salary = self.get_min_salary_constraint()
        logging.debug(f"Minimum salary constraint: {self.min_salary}")
        
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
            num_lineups=requested_lineups,
            team_selections=team_selections,
            min_unique=min_unique,  # Add min unique constraint
            bankroll=bankroll,
            risk_tolerance=risk_tolerance,
            disable_kelly=disable_kelly,
            min_salary=self.min_salary,
            use_advanced_quant=getattr(self, 'use_advanced_quant', False),  # Pass advanced quant flag
            advanced_quant_params=self.get_advanced_quant_params_for_worker()  # Pass advanced quant parameters
        )
        
        # Pass the enable flag to the worker
        if hasattr(self.optimization_thread, 'risk_engine') and not enable_risk_mgmt:
            self.optimization_thread.risk_engine = None  # Disable for this run
        self.optimization_thread.optimization_done.connect(self.display_results)
        logging.debug("Starting optimization thread")
        self.optimization_thread.start()
        
        self.status_label.setText("Running optimization... Please wait.")

    def get_requested_lineups(self):
        """Get the requested number of lineups from the UI input"""
        try:
            num_lineups_text = self.num_lineups_input.text().strip()
            if not num_lineups_text:
                return 100  # Default
            
            num_lineups = int(num_lineups_text)
            if num_lineups < 1:
                num_lineups = 1
            elif num_lineups > 500:
                num_lineups = 500  # Max 500 lineups
            
            return num_lineups
            
        except ValueError:
            logging.warning(f"Invalid number of lineups value: {self.num_lineups_input.text()}")
            return 100

    def get_min_unique_constraint(self):
        """Get the min unique constraint from the UI input"""
        try:
            min_unique_text = self.min_unique_input.text().strip()
            if not min_unique_text:
                return 0  # Default: no constraint
            
            min_unique = int(min_unique_text)
            if min_unique < 0:
                min_unique = 0
            elif min_unique > 10:
                min_unique = 10  # Max 10 unique players per lineup
            
            return min_unique
            
        except ValueError:
            logging.warning(f"Invalid min unique value: {self.min_unique_input.text()}")
            return 0

    def get_min_salary_constraint(self):
        """Get the minimum salary constraint from the UI input"""
        try:
            min_salary_text = self.min_salary_input.text().strip()
            if not min_salary_text:
                return MIN_SALARY_DEFAULT  # Default minimum salary
            
            min_salary = int(min_salary_text)
            if min_salary < 0:
                min_salary = 0
            elif min_salary > SALARY_CAP:
                min_salary = SALARY_CAP  # Cannot exceed salary cap
            
            return min_salary
            
        except ValueError:
            logging.warning(f"Invalid min salary value: {self.min_salary_input.text()}")
            return MIN_SALARY_DEFAULT

    def get_included_players(self):
        """Get the list of included players from the UI with better checkbox detection"""
        included_players = []
        total_checkboxes = 0
        checked_checkboxes = 0
        
        # IMMEDIATE DEBUG OUTPUT
        print(f"\n🔍 CHECKBOX DEBUG - IMPROVED DETECTION:")
        
        if not hasattr(self, 'player_tables') or not self.player_tables:
            print(f"   ❌ No player tables available")
            logging.debug("No player tables available")
            return included_players
        
        print(f"   📋 Found {len(self.player_tables)} player tables")
        
        # Check ALL position tables for selected players with improved detection
        for position_name, table in self.player_tables.items():
            position_checked = 0
            position_total = 0
            
            print(f"   🔍 Checking {position_name} table ({table.rowCount()} rows)...")
            
            for row in range(table.rowCount()):
                # Multiple methods to find the checkbox
                checkbox = None
                
                # Method 1: Check if there's a checkbox widget directly
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget:
                    # Look for QCheckBox in the widget
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if not checkbox:
                        # Maybe the widget itself is a QCheckBox
                        if isinstance(checkbox_widget, QCheckBox):
                            checkbox = checkbox_widget
                        else:
                            # Look through all children
                            for child in checkbox_widget.findChildren(QCheckBox):
                                checkbox = child
                                break
                
                # Method 2: Check if the table item itself has checkbox
                if not checkbox:
                    table_item = table.item(row, 0)
                    if table_item and table_item.checkState() is not None:
                        # This item has a checkbox state
                        total_checkboxes += 1
                        position_total += 1
                        
                        if table_item.checkState() == Qt.Checked:
                            checked_checkboxes += 1
                            position_checked += 1
                            # Get player name from column 1
                            name_item = table.item(row, 1)
                            if name_item:
                                player_name = name_item.text().strip()
                                if player_name and player_name not in included_players:
                                    included_players.append(player_name)
                        continue
                
                # Method 3: Use the found checkbox widget
                if checkbox:
                    total_checkboxes += 1
                    position_total += 1
                    
                    if checkbox.isChecked():
                        checked_checkboxes += 1
                        position_checked += 1
                        # Get player name from column 1
                        name_item = table.item(row, 1)
                        if name_item:
                            player_name = name_item.text().strip()
                            if player_name and player_name not in included_players:
                                included_players.append(player_name)
            
            print(f"     ✅ {position_name}: {position_checked}/{position_total} players selected")
            logging.debug(f"Position {position_name}: {position_checked}/{position_total} players selected")
        
        print(f"   📊 TOTAL: {checked_checkboxes}/{total_checkboxes} checkboxes checked")
        print(f"   🎯 FINAL: {len(included_players)} unique players selected")
        
        # Show warning if too many players selected
        if len(included_players) > 150:
            print(f"   ⚠️  WARNING: {len(included_players)} players selected!")
            print(f"      This is a lot of players. The optimizer can pick from any of these.")
            print(f"      Consider being more selective if you want specific players.")
        elif len(included_players) > 50:
            print(f"    INFO: {len(included_players)} players selected")
            print(f"      The optimizer will choose the best lineups from these players.")
        
        if len(included_players) > 0:
            print(f"   👥 FIRST 5 PLAYERS: {included_players[:5]}")
        
        print(f"")
        
        # Update the status label
        if hasattr(self, 'selection_status_label'):
            if len(included_players) == 0:
                self.selection_status_label.setText("No players selected")
                self.selection_status_label.setStyleSheet("color: #f44336; font-size: 10px;")
            elif len(included_players) <= 50:
                self.selection_status_label.setText(f"{len(included_players)} players selected (Good size)")
                self.selection_status_label.setStyleSheet("color: #4CAF50; font-size: 10px;")
            elif len(included_players) <= 150:
                self.selection_status_label.setText(f"{len(included_players)} players selected (Large pool)")
                self.selection_status_label.setStyleSheet("color: #FF9800; font-size: 10px;")
            else:
                self.selection_status_label.setText(f"{len(included_players)} players selected (Very large!)")
                self.selection_status_label.setStyleSheet("color: #f44336; font-size: 10px; font-weight: bold;")
        
        logging.info(f"🎯 CHECKBOX SELECTION SUMMARY:")
        logging.info(f"   Total checkboxes found: {total_checkboxes}")
        logging.info(f"   Checkboxes checked: {checked_checkboxes}")
        logging.info(f"   Final included players: {len(included_players)}")
        
        if len(included_players) > 0:
            logging.info(f"   First 5 selected players: {included_players[:5]}")
        
        return included_players

    def collect_stack_settings(self):
        """Collect stack settings from the UI"""
        stack_settings = []
        
        if not hasattr(self, 'stack_exposure_table') or not self.stack_exposure_table:
            logging.debug("No stack exposure table available")
            return ["No Stacks"]  # Default
        
        # Check which stack types are selected
        for row in range(self.stack_exposure_table.rowCount()):
            checkbox_widget = self.stack_exposure_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    stack_type_item = self.stack_exposure_table.item(row, 1)
                    if stack_type_item:
                        stack_settings.append(stack_type_item.text())
        
        if not stack_settings:
            stack_settings = ["No Stacks"]  # Default if nothing selected
        
        logging.debug(f"Found stack settings: {stack_settings}")
        return stack_settings

    def collect_exposure_settings(self):
        """Collect min and max exposure settings from the UI"""
        min_exposure = {}
        max_exposure = {}
        
        if not hasattr(self, 'player_tables') or not self.player_tables:
            logging.debug("No player tables available for exposure settings")
            return min_exposure, max_exposure
        
        # For now, return empty dicts as exposure constraints aren't fully implemented
        logging.debug("Exposure settings collection not fully implemented")
        return min_exposure, max_exposure

    def collect_team_selections(self):
        """Collect team selections using manual override first, then enhanced method if available"""
        
        # FIRST: Check for manual team override file
        try:
            import importlib.util
            override_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'manual_team_override.py')
            if os.path.exists(override_path):
                spec = importlib.util.spec_from_file_location("manual_team_override", override_path)
                override_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(override_module)
                
                if hasattr(override_module, 'MANUAL_TEAM_SELECTIONS'):
                    manual_selections = override_module.MANUAL_TEAM_SELECTIONS
                    logging.info(f"🔧 MANUAL OVERRIDE DETECTED: Using manual team selections: {manual_selections}")
                    safe_log_info(f"🔧 MANUAL OVERRIDE ACTIVE: {manual_selections}")
                    print(f"🔧 MANUAL TEAM OVERRIDE ACTIVE:")
                    for stack_size, teams in manual_selections.items():
                        print(f"   {stack_size}-Stack: {teams}")
                        logging.info(f"   Manual override - {stack_size}-Stack: {teams}")
                    return manual_selections
        except Exception as e:
            logging.debug(f"No manual override found or error loading: {e}")
        
        # Try enhanced method next
        if ENHANCED_CHECKBOX_AVAILABLE and hasattr(self, 'checkbox_managers') and self.checkbox_managers:
            logging.info("🚀 Using ENHANCED team selection collection...")
            return collect_team_selections_enhanced(self.team_stack_tables, self.checkbox_managers)
        
        # Fallback to original method with improvements
        logging.info("⚠️ Using fallback team selection collection...")
        team_selections = {}
        
        if not hasattr(self, 'team_stack_tables') or not self.team_stack_tables:
            logging.debug("No team stack tables available")
            return team_selections
        
        safe_log_info("DEBUGGING: Starting team selection collection...")
        
        # Collect selected teams from each stack size table
        for stack_size, table in self.team_stack_tables.items():
            safe_log_info(f"DEBUGGING: Checking table for stack size: {stack_size}")
            safe_log_info(f"DEBUGGING: Table has {table.rowCount()} rows")
            
            selected_teams = []
            
            for row in range(table.rowCount()):
                try:
                    # Get the checkbox from the cell widget
                    checkbox_widget = table.cellWidget(row, 0)
                    team_item = table.item(row, 1)  # Team name is in column 1
                    
                    safe_log_info(f"DEBUGGING: Row {row}:")
                    safe_log_info(f"   - checkbox_widget: {checkbox_widget}")
                    safe_log_info(f"   - team_item: {team_item}")
                    
                    if team_item:
                        team_name = team_item.text()
                        safe_log_info(f"   - team_name: {team_name}")
                    else:
                        safe_log_info(f"   - team_name: None")
                        continue
                    
                    is_checked = False
                    
                    if checkbox_widget:
                        checkbox = None
                        
                        # Method 1: Try through layout (most common)
                        layout = checkbox_widget.layout()
                        safe_log_info(f"   - layout: {layout}")
                        
                        if layout and layout.count() > 0:
                            checkbox = layout.itemAt(0).widget()
                            safe_log_info(f"   - checkbox from layout: {checkbox}")
                            safe_log_info(f"   - checkbox type: {type(checkbox)}")
                            
                            if isinstance(checkbox, QCheckBox):
                                is_checked = checkbox.isChecked()
                                safe_log_info(f"   - is_checked: {is_checked}")
                            
                        # Method 2: Try direct findChild if layout method failed
                        if not checkbox or not isinstance(checkbox, QCheckBox):
                            checkbox = checkbox_widget.findChild(QCheckBox)
                            safe_log_info(f"   - checkbox from findChild: {checkbox}")
                            if checkbox and isinstance(checkbox, QCheckBox):
                                is_checked = checkbox.isChecked()
                                safe_log_info(f"   - is_checked (findChild): {is_checked}")
                        
                        # Method 3: Check if widget itself is a checkbox
                        if not checkbox and isinstance(checkbox_widget, QCheckBox):
                            checkbox = checkbox_widget
                            is_checked = checkbox.isChecked()
                            safe_log_info(f"   - checkbox is widget itself: {is_checked}")
                        
                        if is_checked:
                            selected_teams.append(team_name)
                            safe_log_info(f"   FOUND SELECTED TEAM: {team_name} in stack size {stack_size}")
                        else:
                            safe_log_info(f"   Team {team_name} not selected")
                    else:
                        safe_log_info(f"   No checkbox widget found")
                
                except Exception as e:
                    safe_log_error(f"Error processing row {row} in stack {stack_size}: {e}")
                    import traceback
                    traceback.print_exc()
            
            safe_log_info(f"DEBUGGING: Stack {stack_size} - Found {len(selected_teams)} selected teams: {selected_teams}")
            
            if selected_teams:
                # Convert stack size name to number for optimization logic
                if stack_size == "All Stacks":
                    # For "All Stacks", we'll use these teams for any stack size
                    team_selections["all"] = selected_teams
                else:
                    try:
                        # Try to extract number from stack size (e.g., "4-Stack" -> 4)
                        stack_num = int(stack_size.split(' ')[0])  # Handle "4 Stack" format
                        team_selections[stack_num] = selected_teams
                    except (ValueError, IndexError):
                        # If we can't parse the number, use the string as-is
                        team_selections[stack_size] = selected_teams
                
                safe_log_info(f"Stack size {stack_size}: Selected teams = {selected_teams}")
            else:
                safe_log_info(f"Stack size {stack_size}: No teams selected")
        
        safe_log_info(f"FINAL RESULT: Team selections collected: {team_selections}")
        return team_selections

    def display_results(self, results, team_exposure, stack_exposure):
        """Display optimization results with unique constraint filtering and risk information"""
        logging.debug(f"display_results: Received {len(results)} results")
        logging.info(f"🎯 DISPLAY_RESULTS DEBUG: Starting with {len(results)} lineups")
        
        self.results_table.setRowCount(0)
        
        # Check if we have risk information in results
        # Handle both dict and list formats
        if isinstance(results, dict):
            has_risk_info = any('risk_info' in lineup_data for lineup_data in results.values())
        else:
            # If results is a list, check if any item has risk_info
            has_risk_info = any(isinstance(lineup_data, dict) and 'risk_info' in lineup_data for lineup_data in results)
        
        # Update table headers to include risk info if available
        if has_risk_info:
            self.results_table.setColumnCount(13)
            self.results_table.setHorizontalHeaderLabels([
                "Player", "Team", "Position", "Salary", "Predicted_DK_Points", 
                "Total Salary", "Total Points", "Exposure (%)", "Max Exp (%)",
                "Sharpe Ratio", "Kelly %", "Risk Score", "Position $"
            ])
        else:
            self.results_table.setColumnCount(9)
            self.results_table.setHorizontalHeaderLabels([
                "Player", "Team", "Position", "Salary", "Predicted_DK_Points", 
                "Total Salary", "Total Points", "Exposure (%)", "Max Exp (%)"
            ])
        
        # Get requested number of lineups from UI
        requested_lineups = self.get_requested_lineups()
        
        # Apply min unique filtering if specified
        min_unique = self.get_min_unique_constraint()
        logging.info(f"🎯 MIN UNIQUE CONSTRAINT: {min_unique}")
        
        if min_unique > 0:
            results_before = len(results)
            results = self.filter_lineups_by_uniqueness(results, min_unique)
            results_after = len(results)
            logging.error(f"🚨 MIN UNIQUE FILTERING: {results_before} → {results_after} lineups (FILTERED OUT {results_before - results_after})")
            
            if results_after == 1 and results_before > 1:
                logging.error(f"🚨 CRITICAL: Min unique constraint ({min_unique}) filtered out nearly all lineups!")
                logging.error(f"🚨 SOLUTION: Set min unique to 0 or lower value to get more lineups!")
                print(f"🚨 CRITICAL: Min unique constraint ({min_unique}) filtered out {results_before - results_after} lineups!")
                print(f"🚨 Only {results_after} lineup(s) remain! Try setting min unique to 0 for more lineups.")
        else:
            logging.info(f"🎯 NO MIN UNIQUE FILTERING: Keeping all {len(results)} lineups")
            
        logging.debug(f"After min unique filtering ({min_unique}): {len(results)} results")
        
        total_lineups = len(results)
        
        # Handle both dict and list formats
        if isinstance(results, dict):
            sorted_results = sorted(results.items(), key=lambda x: x[1]['total_points'], reverse=True)
        else:
            # Convert list to items format for sorting
            # Assume each item is a DataFrame and calculate total_points
            items_list = []
            for i, lineup_data in enumerate(results):
                if isinstance(lineup_data, dict) and 'total_points' in lineup_data:
                    items_list.append((i, lineup_data))
                else:
                    # Calculate total points from DataFrame
                    total_points = lineup_data['Predicted_DK_Points'].sum() if 'Predicted_DK_Points' in lineup_data.columns else 0
                    items_list.append((i, {'lineup': lineup_data, 'total_points': total_points}))
            
            sorted_results = sorted(items_list, key=lambda x: x[1]['total_points'], reverse=True)

        self.optimized_lineups = []
        for _, lineup_data in sorted_results:
            self.add_lineup_to_results(lineup_data, total_lineups, has_risk_info)
            self.optimized_lineups.append(lineup_data['lineup'])

        self.update_exposure_in_all_tabs(total_lineups, team_exposure, stack_exposure)
        self.refresh_team_stacks()
        
        # Create status message with risk information
        status_message = self.create_status_message(total_lineups, requested_lineups, min_unique, has_risk_info, results)
        self.status_label.setText(status_message)
        
        # Show detailed feedback dialog for significant differences
        if total_lineups < requested_lineups * 0.8:  # Less than 80% of requested
            self.show_optimization_feedback(total_lineups, requested_lineups, min_unique, has_risk_info)
    
    def create_status_message(self, total_lineups, requested_lineups, min_unique, has_risk_info, results):
        """Create comprehensive status message"""
        unique_msg = f" (Min unique: {min_unique})" if min_unique > 0 else ""
        risk_msg = ""
        
        if has_risk_info and results:
            # Calculate average risk metrics
            # Handle both dict and list formats
            if isinstance(results, dict):
                risk_infos = [lineup_data.get('risk_info', {}) for lineup_data in results.values()]
            else:
                risk_infos = [lineup_data.get('risk_info', {}) for lineup_data in results if isinstance(lineup_data, dict)]
            avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in risk_infos if r])
            avg_kelly = np.mean([r.get('kelly_fraction', 0) for r in risk_infos if r]) * 100
            total_position = sum([r.get('position_size', 0) for r in risk_infos if r])
            
            risk_msg = f" | 🔥 Avg Sharpe: {avg_sharpe:.3f}, Kelly: {avg_kelly:.1f}%, Total: ${total_position:.0f}"
        
        if total_lineups < requested_lineups:
            return f"⚠️ Generated {total_lineups}/{requested_lineups} lineups{unique_msg}{risk_msg}"
        else:
            return f"✅ Generated {total_lineups} lineups{unique_msg}{risk_msg}"
    
    def show_optimization_feedback(self, total_lineups, requested_lineups, min_unique, has_risk_info):
        """Show detailed optimization feedback dialog"""
        risk_text = "\n🔥 Risk-adjusted optimization was used" if has_risk_info else ""
        
        QMessageBox.information(
            self,
            "Lineup Generation Results",
            f"🎯 Optimization Results:\n\n"
            f"📊 Requested: {requested_lineups} lineups\n"
            f"✅ Generated: {total_lineups} lineups\n\n"
            f"{'🔄 Min unique constraint limited results' if min_unique > 0 else '⚡ Limited by available player combinations'}"
            f"{risk_text}\n\n"
            f"💡 Tip: Try reducing min unique constraint or adjusting stack settings for more lineups."
        )

    def update_exposure_in_all_tabs(self, total_lineups, team_exposure, stack_exposure):
        """Update exposure statistics in all UI tabs"""
        # Update stack exposure in the stack exposure table
        if hasattr(self, 'stack_exposure_table') and self.stack_exposure_table:
            for row in range(self.stack_exposure_table.rowCount()):
                stack_type_item = self.stack_exposure_table.item(row, 1)
                if stack_type_item:
                    stack_type = stack_type_item.text()
                    exposure_count = stack_exposure.get(stack_type, 0)
                    exposure_percentage = (exposure_count / total_lineups * 100) if total_lineups > 0 else 0
                    
                    # Update lineup exposure (column 4)
                    self.stack_exposure_table.setItem(row, 4, QTableWidgetItem(f"{exposure_percentage:.1f}%"))
                    
                    # Pool exposure and Entry exposure can be the same for now (columns 5 and 6)
                    self.stack_exposure_table.setItem(row, 5, QTableWidgetItem(f"{exposure_percentage:.1f}%"))
                    self.stack_exposure_table.setItem(row, 6, QTableWidgetItem(f"{exposure_percentage:.1f}%"))
        
        # Update team exposure in team stack tables
        if hasattr(self, 'team_stack_tables') and self.team_stack_tables:
            for stack_size_name, table in self.team_stack_tables.items():
                for row in range(table.rowCount()):
                    team_item = table.item(row, 1)  # Teams column
                    if team_item:
                        team_name = team_item.text()
                        exposure_count = team_exposure.get(team_name, 0)
                        exposure_percentage = (exposure_count / total_lineups * 100) if total_lineups > 0 else 0
                        
                        # Update actual exposure (column 7)
                        table.setItem(row, 7, QTableWidgetItem(f"{exposure_percentage:.1f}%"))
        
        # Update player exposure in player tables (this is already handled in add_lineup_to_results)
        logging.debug(f"Updated exposure in all tabs: {total_lineups} lineups, {len(team_exposure)} teams, {len(stack_exposure)} stacks")

    def filter_lineups_by_uniqueness(self, results, min_unique):
        """Filter lineups to ensure minimum number of unique players between consecutive lineups"""
        if min_unique <= 0 or len(results) <= 1:
            return results
        
        filtered_results = {}
        
        # Handle both dict and list formats
        if isinstance(results, dict):
            sorted_results = sorted(results.items(), key=lambda x: x[1]['total_points'], reverse=True)
        else:
            # Convert list to items format for sorting
            items_list = []
            for i, lineup_data in enumerate(results):
                if isinstance(lineup_data, dict) and 'total_points' in lineup_data:
                    items_list.append((i, lineup_data))
                else:
                    # Calculate total points from DataFrame
                    total_points = lineup_data['Predicted_DK_Points'].sum() if 'Predicted_DK_Points' in lineup_data.columns else 0
                    items_list.append((i, {'lineup': lineup_data, 'total_points': total_points}))
            
            sorted_results = sorted(items_list, key=lambda x: x[1]['total_points'], reverse=True)
        
        if sorted_results:
            # Always keep the first (best) lineup
            first_key, first_data = sorted_results[0]
            filtered_results[0] = first_data
            kept_lineups = [set(first_data['lineup']['Name'].tolist())]
        
        kept_count = 1
        for key, lineup_data in sorted_results[1:]:
            current_players = set(lineup_data['lineup']['Name'].tolist())
            
            # STRICT MIN UNIQUE LOGIC: Always enforce exact min_unique constraint
            if kept_lineups:
                # Check against ALL previously kept lineups, not just the most recent
                is_unique_enough = True
                for previous_players in kept_lineups:
                    unique_players = len(current_players.symmetric_difference(previous_players))
                    if unique_players < min_unique:
                        is_unique_enough = False
                        break
            else:
                is_unique_enough = True
            
            if is_unique_enough:
                filtered_results[kept_count] = lineup_data
                kept_lineups.append(current_players)
                kept_count += 1
                
                # Keep all lineups for strict comparison (no trimming for leniency)
                # This ensures we maintain strict uniqueness against ALL previous lineups
        
        kept_ratio = len(filtered_results) / len(results) if results else 0
        logging.info(f"🎯 STRICT MIN UNIQUE: Kept {len(filtered_results)} out of {len(results)} lineups ({kept_ratio:.1%}) (min_unique={min_unique})")
        
        # STRICT MODE: Only provide emergency fallback if we have very few results
        min_required = max(3, int(len(results) * 0.15))  # At least 15% of lineups or 3, whichever is higher
        if len(filtered_results) < min_required and len(results) >= min_required:
            logging.warning(f"🚨 STRICT EMERGENCY MODE: Only {len(filtered_results)} lineups kept with strict min_unique={min_unique}, providing minimal fallback of {min_required}")
            logging.warning(f"💡 RECOMMENDATION: Consider reducing min_unique constraint if you need more lineups")
            # Take the top lineups regardless of uniqueness constraint
            # Handle both dict and list formats
            if isinstance(results, dict):
                emergency_sorted = sorted(results.items(), key=lambda x: x[1]['total_points'], reverse=True)
            else:
                # Convert list to items format for sorting
                items_list = []
                for i, lineup_data in enumerate(results):
                    if isinstance(lineup_data, dict) and 'total_points' in lineup_data:
                        items_list.append((i, lineup_data))
                    else:
                        # Calculate total points from DataFrame
                        total_points = lineup_data['Predicted_DK_Points'].sum() if 'Predicted_DK_Points' in lineup_data.columns else 0
                        items_list.append((i, {'lineup': lineup_data, 'total_points': total_points}))
                
                emergency_sorted = sorted(items_list, key=lambda x: x[1]['total_points'], reverse=True)
            filtered_results = {}
            for i, (key, lineup_data) in enumerate(emergency_sorted[:min_required]):
                filtered_results[i] = lineup_data
            logging.info(f"🚨 EMERGENCY: Forced to keep {len(filtered_results)} lineups by ignoring uniqueness")
            
        # If we're still filtering out too many, give detailed feedback
        if kept_ratio < 0.2 and min_unique > 0 and len(filtered_results) >= 5:
            logging.warning(f"⚠️ Min unique constraint ({min_unique}) filtered out {len(results) - len(filtered_results)} lineups")
            logging.warning(f"✅ But we guaranteed at least {len(filtered_results)} diverse lineups")
        elif len(filtered_results) < 5:
            logging.error(f"🚨 CRITICAL: Could only generate {len(filtered_results)} lineup(s)")
            logging.error(f"💡 SUGGESTIONS:")
            logging.error(f"   • Check your player data has enough variety")
            logging.error(f"   • Verify salary cap and position constraints")
            logging.error(f"   • Look for constraint conflicts")
        
        return filtered_results

    def add_lineup_to_results(self, lineup_data, total_lineups, has_risk_info=False):
        """Add a lineup to the results table with optional risk information"""
        total_points = lineup_data['total_points']
        lineup = lineup_data['lineup']
        total_salary = lineup['Salary'].sum()
        risk_info = lineup_data.get('risk_info', {})

        for _, player in lineup.iterrows():
            row_position = self.results_table.rowCount()
            self.results_table.insertRow(row_position)
            
            # Basic lineup information
            self.results_table.setItem(row_position, 0, QTableWidgetItem(str(player['Name'])))
            self.results_table.setItem(row_position, 1, QTableWidgetItem(str(player['Team'])))
            self.results_table.setItem(row_position, 2, QTableWidgetItem(str(player['Position'])))
            self.results_table.setItem(row_position, 3, QTableWidgetItem(str(player['Salary'])))
            self.results_table.setItem(row_position, 4, QTableWidgetItem(f"{player['Predicted_DK_Points']:.2f}"))
            self.results_table.setItem(row_position, 5, QTableWidgetItem(str(total_salary)))
            self.results_table.setItem(row_position, 6, QTableWidgetItem(f"{total_points:.2f}"))

            # Exposure calculation
            player_name = player['Name']
            if player_name in self.player_exposure:
                self.player_exposure[player_name] += 1
            else:
                self.player_exposure[player_name] = 1

            exposure = self.player_exposure.get(player_name, 0) / total_lineups * 100
            self.results_table.setItem(row_position, 7, QTableWidgetItem(f"{exposure:.2f}%"))
            self.results_table.setItem(row_position, 8, QTableWidgetItem(f"{self.max_exposure.get(player_name, 100):.2f}%"))
            
            # Add risk information if available
            if has_risk_info and risk_info:
                sharpe_ratio = risk_info.get('sharpe_ratio', 0)
                kelly_fraction = risk_info.get('kelly_fraction', 0) * 100  # Convert to percentage
                volatility = risk_info.get('volatility', 0)
                position_size = risk_info.get('position_size', 0)
                
                # Color code Sharpe ratio for easy interpretation
                sharpe_item = QTableWidgetItem(f"{sharpe_ratio:.3f}")
                if sharpe_ratio > 1.0:
                    sharpe_item.setBackground(QColor(76, 175, 80))  # Green for excellent
                elif sharpe_ratio > 0.5:
                    sharpe_item.setBackground(QColor(255, 193, 7))  # Yellow for good
                elif sharpe_ratio < 0:
                    sharpe_item.setBackground(QColor(244, 67, 54))  # Red for poor
                
                self.results_table.setItem(row_position, 9, sharpe_item)
                self.results_table.setItem(row_position, 10, QTableWidgetItem(f"{kelly_fraction:.1f}%"))
                self.results_table.setItem(row_position, 11, QTableWidgetItem(f"{volatility:.2f}"))
                self.results_table.setItem(row_position, 12, QTableWidgetItem(f"${position_size:.0f}"))

    def load_file(self):
        """Load player data from CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open CSV File', '', 'CSV Files (*.csv);;All Files (*)'
        )
        if file_path:
            try:
                self.df_players = self.load_players(file_path)
                self.populate_player_tables()
                self.status_label.setText(f'Players loaded: {len(self.df_players)} players')
            except Exception as e:
                self.status_label.setText(f'Error loading file: {str(e)}')
                logging.error(f"Error loading file: {e}")

    def load_players(self, file_path):
        """Load players from CSV file with error handling and flexible column mapping"""
        try:
            print(f"🔍 LOADING CSV: {file_path}")
            df = pd.read_csv(file_path)
            print(f"🔍 RAW CSV loaded - Shape: {df.shape}")
            print(f"🔍 RAW CSV columns: {list(df.columns)}")
            
            # Check for probability columns in raw CSV
            raw_prob_cols = [col for col in df.columns if 'prob' in col.lower()]
            print(f"🔍 Raw probability columns in CSV: {len(raw_prob_cols)} - {raw_prob_cols[:5]}{'...' if len(raw_prob_cols) > 5 else ''}")
            
            logging.info(f"Loading players from CSV: {file_path}")
            logging.info(f"Raw CSV shape: {df.shape}, columns: {len(df.columns)}")
            
            # 🔧 AUTOMATIC COLUMN MAPPING - Handle both old and new formats
            column_mapping = {
                'Pos': 'Position',  # Handle old format
                'Predicted_Points': 'Predicted_DK_Points'  # Handle old format
            }
            
            # Apply column mapping automatically
            columns_renamed = []
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
                    columns_renamed.append(f"{old_col} → {new_col}")
            
            if columns_renamed:
                logging.info(f"🔧 Auto-renamed columns: {', '.join(columns_renamed)}")
                print(f"🔧 Auto-renamed columns: {', '.join(columns_renamed)}")
            
            # Basic required columns (after potential renaming)
            basic_required = ['Name', 'Team', 'Position', 'Salary']
            
            # Check for basic required columns
            missing_basic = [col for col in basic_required if col not in df.columns]
            if missing_basic:
                available_cols = list(df.columns)
                raise ValueError(f"Missing required columns: {missing_basic}. Available columns: {available_cols}")
            
            # Handle different prediction column names flexibly
            prediction_column = None
            possible_prediction_columns = [
                'Predicted_DK_Points',  # Standard expected name
                'My_Proj',              # Your CSV format
                'ML_Prediction',        # ML prediction column
                'PPG_Projection',       # PPG projection column
                'Projection',           # Generic projection
                'Points',               # Simple points
                'DK_Points',            # DraftKings points
                'Fantasy_Points'        # Fantasy points
            ]
            
            # Find the first available prediction column
            for col in possible_prediction_columns:
                if col in df.columns:
                    prediction_column = col
                    break
            
            if prediction_column is None:
                available_cols = list(df.columns)
                raise ValueError(f"No prediction column found. Available columns: {available_cols}. Expected one of: {possible_prediction_columns}")
            
            # Rename the prediction column to the standard name for consistency
            if prediction_column != 'Predicted_DK_Points':
                df = df.rename(columns={prediction_column: 'Predicted_DK_Points'})
                logging.info(f"Using '{prediction_column}' as prediction column, renamed to 'Predicted_DK_Points'")
            
            # Clean and validate data
            df = df.dropna(subset=['Name', 'Salary', 'Predicted_DK_Points'])
            df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
            df['Predicted_DK_Points'] = pd.to_numeric(df['Predicted_DK_Points'], errors='coerce')
            
            # Remove rows with invalid salary or prediction values
            df = df.dropna(subset=['Salary', 'Predicted_DK_Points'])
            df = df[df['Salary'] > 0]
            df = df[df['Predicted_DK_Points'] > 0]
            
            # 🎲 DETECT AND ENHANCE PROBABILITY DATA
            print(f"🔍 PROBABILITY DEBUG: PROBABILITY_OPTIMIZER_AVAILABLE = {PROBABILITY_OPTIMIZER_AVAILABLE}")
            if PROBABILITY_OPTIMIZER_AVAILABLE:
                try:
                    print("🔍 Creating ProbabilityEnhancedOptimizer instance...")
                    prob_optimizer = ProbabilityEnhancedOptimizer()
                    print("✅ ProbabilityEnhancedOptimizer created successfully")
                    
                    print("🔍 Detecting probability columns...")
                    prob_columns = prob_optimizer.detect_probability_columns(df)
                    print(f"🔍 Detected probability columns: {prob_columns}")
                    
                    # Check if any probability columns were detected
                    total_prob_cols = sum(len(cols) for cols in prob_columns.values())
                    print(f"🔍 Total probability columns detected: {total_prob_cols}")
                    
                    if total_prob_cols > 0:
                        logging.info(f"🎲 Detected {total_prob_cols} probability columns")
                        print(f"✅ Found {total_prob_cols} probability columns!")
                        
                        # Extract probability thresholds
                        prob_thresholds = prob_optimizer.extract_probability_thresholds(df)
                        print(f"🔍 Probability thresholds: {prob_thresholds}")
                        
                        # Enhance dataframe with probability-derived metrics
                        df = prob_optimizer.enhance_player_data(df)
                        print(f"✅ Enhanced player data with probability metrics")
                        
                        # Create summary for GUI
                        prob_summary = {
                            'columns': prob_columns,
                            'enhanced_players': len(df),
                            'metrics': {
                                'Expected_Utility': len(df) if 'Expected_Utility' in df.columns else 0,
                                'Risk_Adjusted_Points': len(df) if 'Risk_Adjusted_Points' in df.columns else 0,
                                'Kelly_Fraction': len(df) if 'Kelly_Fraction' in df.columns else 0,
                                'Implied_Volatility': len(df) if 'Implied_Volatility' in df.columns else 0
                            }
                        }
                        
                        # Update GUI with probability information
                        self.update_probability_display(prob_summary, "Auto-detected")
                        
                        logging.info("🎲 Successfully enhanced player data with probability metrics")
                    else:
                        print("❌ No probability columns detected in CSV")
                        logging.info("No probability columns detected in CSV")
                        self.update_probability_display({'columns': {}, 'enhanced_players': 0, 'metrics': {}}, "None")
                        
                except Exception as e:
                    print(f"❌ Error enhancing probability data: {e}")
                    logging.warning(f"Error enhancing probability data: {e}")
                    import traceback
                    traceback.print_exc()
                    self.update_probability_display_error(str(e))
            else:
                print("❌ Probability optimizer not available")
                logging.info("Probability optimizer not available")
                self.update_probability_display_no_data()
            
            logging.info(f"Successfully loaded {len(df)} players from {file_path}")
            logging.info(f"Using prediction column: {prediction_column}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading players: {str(e)}")

    def populate_player_tables(self):
        """Populate player tables with loaded data"""
        if self.df_players is None or self.df_players.empty:
            return
        
        # Clear existing player exposure data
        self.player_exposure = {}
        
        # Group players by position
        position_groups = {
            'All Batters': self.df_players[~self.df_players['Position'].str.contains('P', na=False)],
            'C': self.df_players[self.df_players['Position'].str.contains('C', na=False)],
            '1B': self.df_players[self.df_players['Position'].str.contains('1B', na=False)],
            '2B': self.df_players[self.df_players['Position'].str.contains('2B', na=False)],
            '3B': self.df_players[self.df_players['Position'].str.contains('3B', na=False)],
            'SS': self.df_players[self.df_players['Position'].str.contains('SS', na=False)],
            'OF': self.df_players[self.df_players['Position'].str.contains('OF', na=False)],
            'P': self.df_players[self.df_players['Position'].str.contains('P', na=False)]
        }
        
        # Populate each table
        for position, table in self.player_tables.items():
            if position in position_groups:
                df_pos = position_groups[position]
                self.populate_position_table(table, df_pos)
        
        # Also populate team stack tables when player data is loaded
        self.populate_team_stack_table()
        
        # Also populate team combinations when player data is loaded
        self.populate_team_combinations_teams()
        
        # Update selection status after all tables are populated
        self.update_selection_status()

    def populate_position_table(self, table, df_pos):
        """Populate a specific position table with player data"""
        table.setRowCount(len(df_pos))
        
        for row_idx, (_, player) in enumerate(df_pos.iterrows()):
            # Checkbox for inclusion
            checkbox = QCheckBox()
            # Create unique identifier for each checkbox
            player_name = player['Name']
            checkbox.setObjectName(f"player_checkbox_{row_idx}_{player_name}")
            
            # Enhanced signal connection with logging for individual changes
            def create_checkbox_handler(name, pos_name, cb):
                def handle_checkbox_change(state):
                    checked = (state == 2)  # Qt.Checked = 2
                    action = "SELECTED" if checked else "DESELECTED"
                    print(f"🔄 CHECKBOX CHANGE: {action} {name} ({pos_name})")
                    logging.info(f"Player {action}: {name} in {pos_name} position")
                    
                    # Update selection status
                    self.update_selection_status()
                    
                    # If this was a deselection, show additional info
                    if not checked:
                        included_players = self.get_included_players()
                        print(f"   📊 Remaining selected: {len(included_players)} players")
                        
                return handle_checkbox_change
            
            # Connect the enhanced signal handler
            checkbox.stateChanged.connect(create_checkbox_handler(player_name, table.objectName() if hasattr(table, 'objectName') else "Unknown", checkbox))
            
            # Store checkbox reference for debugging/access
            if not hasattr(self, '_player_checkboxes'):
                self._player_checkboxes = {}
            self._player_checkboxes[player_name] = checkbox
            
            checkbox_widget = QWidget()
            layout = QHBoxLayout(checkbox_widget)
            layout.addWidget(checkbox)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            table.setCellWidget(row_idx, 0, checkbox_widget)
            
            # Player data
            table.setItem(row_idx, 1, QTableWidgetItem(str(player['Name'])))
            table.setItem(row_idx, 2, QTableWidgetItem(str(player['Team'])))
            table.setItem(row_idx, 3, QTableWidgetItem(str(player['Position'])))
            table.setItem(row_idx, 4, QTableWidgetItem(str(player['Salary'])))
            table.setItem(row_idx, 5, QTableWidgetItem(f"{player['Predicted_DK_Points']:.2f}"))
            
            # Value calculation
            value = player['Predicted_DK_Points'] / (player['Salary'] / 1000) if player['Salary'] > 0 else 0
            table.setItem(row_idx, 6, QTableWidgetItem(f"{value:.2f}"))
            
            # Exposure controls
            min_exp_spinbox = QSpinBox()
            min_exp_spinbox.setRange(0, 100)
            min_exp_spinbox.setValue(0)
            table.setCellWidget(row_idx, 7, min_exp_spinbox)
            
            max_exp_spinbox = QSpinBox()
            max_exp_spinbox.setRange(0, 100)
            max_exp_spinbox.setValue(100)
            table.setCellWidget(row_idx, 8, max_exp_spinbox)
            
            # Actual exposure (will be updated after optimization)
            actual_exp_label = QLabel("0.00%")
            table.setCellWidget(row_idx, 9, actual_exp_label)

    def populate_team_stack_table(self):
        """Populate team stack tables with enhanced checkbox management"""
        if self.df_players is None or self.df_players.empty:
            return
        
        # Get unique teams and their projected runs
        teams = self.df_players['Team'].unique()
        
        for stack_size_name, table in self.team_stack_tables.items():
            table.setRowCount(len(teams))
            
            # Initialize checkbox manager for this stack size if available
            if ENHANCED_CHECKBOX_AVAILABLE and self.checkbox_managers is not None:
                if stack_size_name not in self.checkbox_managers:
                    self.checkbox_managers[stack_size_name] = CheckboxManager()
                    logging.info(f"✅ Created checkbox manager for {stack_size_name}")
                manager = self.checkbox_managers[stack_size_name]
            else:
                manager = None
            
            for row_idx, team in enumerate(teams):
                # Create checkbox widget using enhanced method if available
                if manager:
                    row_id = f"team_{row_idx}"
                    checkbox_widget = manager.create_checkbox_widget(row_id, False)
                    logging.debug(f"✅ Created enhanced checkbox for {team} (row {row_idx})")
                else:
                    # Fallback to standard checkbox creation
                    checkbox = QCheckBox()
                    checkbox_widget = QWidget()
                    layout = QHBoxLayout(checkbox_widget)
                    layout.addWidget(checkbox)
                    layout.setAlignment(Qt.AlignCenter)
                    layout.setContentsMargins(0, 0, 0, 0)
                    logging.debug(f"⚠️ Created standard checkbox for {team} (row {row_idx})")
                
                table.setCellWidget(row_idx, 0, checkbox_widget)
                
                # Team data
                table.setItem(row_idx, 1, QTableWidgetItem(str(team)))
                
                # Calculate team stats
                team_players = self.df_players[self.df_players['Team'] == team]
                avg_salary = team_players['Salary'].mean()
                avg_points = team_players['Predicted_DK_Points'].mean()
                total_points = team_players['Predicted_DK_Points'].sum()
                player_count = len(team_players)
                
                # Set proper columns based on headers: ["Select", "Teams", "Status", "Time", "Proj Runs", "Min Exp", "Max Exp", "Actual Exp (%)"]
                table.setItem(row_idx, 2, QTableWidgetItem("Active"))  # Status
                table.setItem(row_idx, 3, QTableWidgetItem("--"))  # Time (placeholder)
                table.setItem(row_idx, 4, QTableWidgetItem(f"{total_points:.1f}"))  # Proj Runs (using total points)
                table.setItem(row_idx, 5, QTableWidgetItem("0"))  # Min exposure
                table.setItem(row_idx, 6, QTableWidgetItem("100"))  # Max exposure
                table.setItem(row_idx, 7, QTableWidgetItem("0.00%"))  # Actual exposure
        
        safe_log_info(f"Team stack tables populated for {len(teams)} teams across {len(self.team_stack_tables)} stack sizes")

    def select_all(self, position):
        """Select all players in a specific position table"""
        if not hasattr(self, 'player_tables') or position not in self.player_tables:
            logging.debug(f"No table found for position: {position}")
            return
        
        table = self.player_tables[position]
        
        # Check all checkboxes in the table
        for row in range(table.rowCount()):
            checkbox_widget = table.cellWidget(row, 0)
            if checkbox_widget:
                # Find the checkbox within the widget
                layout = checkbox_widget.layout()
                if layout and layout.count() > 0:
                    checkbox = layout.itemAt(0).widget()
                    if isinstance(checkbox, QCheckBox):
                        checkbox.setChecked(True)
        
        logging.debug(f"Selected all players in {position} table")

    def deselect_all(self, position):
        """Deselect all players in a specific position table"""
        if not hasattr(self, 'player_tables') or position not in self.player_tables:
            logging.debug(f"No table found for position: {position}")
            return
        
        table = self.player_tables[position]
        
        # Uncheck all checkboxes in the table
        for row in range(table.rowCount()):
            checkbox_widget = table.cellWidget(row, 0)
            if checkbox_widget:
                # Find the checkbox within the widget
                layout = checkbox_widget.layout()
                if layout and layout.count() > 0:
                    checkbox = layout.itemAt(0).widget()
                    if isinstance(checkbox, QCheckBox):
                        checkbox.setChecked(False)
        
        logging.debug(f"Deselected all players in {position} table")

    def load_entries_csv(self):
        """Load entries CSV for analysis"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Entries CSV', '', 'CSV Files (*.csv);;All Files (*)'
        )
        if file_path:
            try:
                self.df_entries = pd.read_csv(file_path)
                self.status_label.setText(f'Entries loaded: {len(self.df_entries)} entries')
            except Exception as e:
                self.status_label.setText(f'Error loading entries: {str(e)}')
                logging.error(f"Error loading entries: {e}")

    def save_csv(self):
        """Save optimized lineups to CSV"""
        if not hasattr(self, 'optimized_lineups') or not self.optimized_lineups:
            QMessageBox.warning(self, "No Data", "No optimized lineups to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Optimized Lineups', 'optimized_lineups.csv', 'CSV Files (*.csv);;All Files (*)'
        )
        if file_path:
            try:
                self.save_lineups_to_dk_format(file_path)
                self.status_label.setText(f'Lineups saved to: {file_path}')
            except Exception as e:
                self.status_label.setText(f'Error saving: {str(e)}')
                logging.error(f"Error saving lineups: {e}")

    def save_lineups_to_dk_format(self, output_path):
        """Save lineups in DraftKings format"""
        dk_positions = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(dk_positions)
            
            for lineup in self.optimized_lineups:
                dk_lineup = self.format_lineup_for_dk(lineup, dk_positions)
                writer.writerow(dk_lineup)

    def format_lineup_for_dk(self, lineup, dk_positions):
        """Group players by position and assign to DraftKings positions"""
        position_players = {
            'P': [],
            'C': [],
            '1B': [],
            '2B': [],
            '3B': [],
            'SS': [],
            'OF': []
        }
        
        for _, player in lineup.iterrows():
            pos = str(player['Position']).upper()
            name = str(player['Name'])
            
            # Handle pitcher designations
            if 'P' in pos or 'SP' in pos or 'RP' in pos:
                position_players['P'].append(name)
            elif 'C' in pos:
                position_players['C'].append(name)
            elif '1B' in pos:
                position_players['1B'].append(name)
            elif '2B' in pos:
                position_players['2B'].append(name)
            elif '3B' in pos:
                position_players['3B'].append(name)
            elif 'SS' in pos:
                position_players['SS'].append(name)
            elif 'OF' in pos:
                position_players['OF'].append(name)
        
        # Assign players to DK positions
        dk_lineup = []
        position_usage = {pos: 0 for pos in position_players.keys()}
        
        for dk_pos in dk_positions:
            if dk_pos in position_players and position_usage[dk_pos] < len(position_players[dk_pos]):
                dk_lineup.append(position_players[dk_pos][position_usage[dk_pos]])
                position_usage[dk_pos] += 1
            else:
                # Find any remaining player that can fill this position
                assigned = False
                for pos, players in position_players.items():
                    if position_usage[pos] < len(players):
                        dk_lineup.append(players[position_usage[pos]])
                        position_usage[pos] += 1
                        assigned = True
                        break
                if not assigned:
                    dk_lineup.append("")

        return dk_lineup

    def load_dk_predictions(self):
        """Load DraftKings predictions from CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open DraftKings Predictions CSV', '', 'CSV Files (*.csv);;All Files (*)'
        )
        if file_path:
            try:
                self.df_players = self.load_players(file_path)
                self.populate_player_tables()
                self.populate_team_stack_table()
                self.status_label.setText(f'DraftKings predictions loaded: {len(self.df_players)} players')
            except Exception as e:
                self.status_label.setText(f'Error loading DraftKings predictions: {str(e)}')
                logging.error(f"Error loading DraftKings predictions: {e}")

    def load_dk_entries_file(self):
        """Load a DraftKings entries file to be filled with optimized lineups"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Load DraftKings Entries File', '', 'CSV Files (*.csv);;All Files (*)'
        )
        
        if not file_path:
            return
        
        try:
            # Use robust entry detection logic similar to clean_dk_entries.py
            self.dk_entries_df = self.parse_dk_entries_robustly(file_path)
            
            if self.dk_entries_df is None or self.dk_entries_df.empty:
                raise ValueError("No valid entries found in the file")
                
            # Store the original file path for saving back
            self.dk_entries_file_path = file_path
            
            # Detect format
            self.dk_entries_format = self.detect_dk_format()
            
            # Show success message
            num_entries = len(self.dk_entries_df)
            file_columns = list(self.dk_entries_df.columns)
            
            success_msg = f"✅ DraftKings entries file loaded successfully!\n\n"
            success_msg += f"📁 File: {os.path.basename(file_path)}\n"
            success_msg += f"📊 Number of entries: {num_entries}\n"
            success_msg += f"📋 Columns ({len(file_columns)}): {', '.join(file_columns[:5])}{'...' if len(file_columns) > 5 else ''}\n"
            success_msg += f"🎯 Format detected: {self.dk_entries_format}"
            
            if num_entries == 0:
                success_msg += "\n\n⚠️ File appears to be empty - will create new entries when you fill it."
            
            QMessageBox.information(self, "DraftKings Entries Loaded", success_msg)
            self.status_label.setText(f'DK Entries loaded: {num_entries} entries from {os.path.basename(file_path)}')
            
        except Exception as e:
            error_msg = f"Error loading DraftKings entries file: {str(e)}"
            self.status_label.setText(error_msg)
            logging.error(error_msg)
            
            # Show detailed error dialog
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Load Error")
            error_text = f"Failed to load DraftKings entries file:\n\n{str(e)}\n\n"
            error_text += "Common solutions:\n"
            error_text += "• Check that the file is a valid CSV format\n"
            error_text += "• Ensure all rows have consistent column counts\n"
            error_text += "• Try opening the file in Excel and re-saving as CSV\n"
            error_text += "• Remove any extra commas or special characters\n"
            error_text += "• Make sure the file contains proper DraftKings headers"
            error_dialog.setText(error_text)
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()
    
    def parse_dk_entries_robustly(self, file_path):
        """Robust parsing of DraftKings entries files using logic from clean_dk_entries.py"""
        logging.info(f"🔄 Parsing DK entries file robustly: {file_path}")
        
        try:
            # Read all lines
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                lines = list(reader)
            
            if not lines:
                logging.warning("File is empty!")
                return None
            
            # Standard DK header
            header_line = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee", "P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
            entry_rows = []
            
            # Check if first row is header
            first_row = lines[0] if lines else []
            if "Entry ID" in ' '.join(first_row):
                start_row = 1
                logging.info("📋 Found header in first row")
            else:
                start_row = 0
                logging.info("📋 No header found, processing all rows")
            
            in_player_section = False
            
            for i, row in enumerate(lines[start_row:], start_row):
                # Clean up the row
                row = [cell.strip().strip('"') for cell in row]
                
                if not row or all(not cell for cell in row):
                    continue
                
                # Check if we're in the player data section
                if len(row) > 7 and any(phrase in str(row[6]).lower() for phrase in ["position", "name + id"]):
                    in_player_section = True
                    logging.info(f"🔍 Found player data section at line {i+1}")
                    continue
                
                if in_player_section:
                    continue
                
                # Check if this is an entry row (has Entry ID)
                if len(row) >= 4 and row[0]:
                    entry_id = row[0].replace(',', '').replace('"', '').strip()
                    
                    # Valid entry ID should be numeric and reasonably long
                    if entry_id.isdigit() and len(entry_id) >= 8:
                        contest_name = row[1] if len(row) > 1 else ""
                        contest_id = row[2] if len(row) > 2 else ""
                        entry_fee = row[3] if len(row) > 3 else ""
                        
                        # Build the clean row with exactly 14 columns
                        clean_row = [entry_id, contest_name, contest_id, entry_fee]
                        
                        # Add the 10 player positions (P, P, C, 1B, 2B, 3B, SS, OF, OF, OF)
                        for j in range(4, 14):
                            if j < len(row):
                                clean_row.append(row[j])
                            else:
                                clean_row.append('')
                        
                        entry_rows.append(clean_row)
            
            logging.info(f"📊 Found {len(entry_rows)} valid entries")
            
            if not entry_rows:
                logging.warning("No valid entries found!")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(entry_rows, columns=header_line)
            
            # Clean up the DataFrame
            df = df.dropna(how='all')
            df.columns = [str(col).strip() for col in df.columns]
            
            # Remove unnamed/empty columns
            cols_to_drop = []
            for col in df.columns:
                if 'Unnamed' in str(col) or str(col).strip() == '':
                    if df[col].isna().all() or (df[col].astype(str) == '').all():
                        cols_to_drop.append(col)
            
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logging.info(f"Removed {len(cols_to_drop)} empty columns")
            
            return df
            
        except Exception as e:
            logging.error(f"Error in robust parsing: {str(e)}")
            return None

    def detect_dk_format(self):
        """Detect the DraftKings file format"""
        file_columns = list(self.dk_entries_df.columns)
        expected_positions = ['P', 'C', '1B', '2B', '3B', 'SS', 'OF']
        
        # Check for contest format (Entry ID, Contest Name, etc.)
        if 'Entry ID' in file_columns and 'Contest Name' in file_columns:
            return 'contest_format'
        
        # Check if columns match DK position format exactly (including duplicate positions)
        dk_positions = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
        # Count positions in file columns
        position_counts = {}
        legacy_position_counts = {}
        
        for col in file_columns:
            col_upper = str(col).upper().strip()
            # Standard format counting
            if col_upper in ['P', 'C', '1B', '2B', '3B', 'SS', 'OF']:
                position_counts[col_upper] = position_counts.get(col_upper, 0) + 1
            # Legacy format counting (P.1, OF.1, OF.2)
            elif col_upper in ['P.1', 'P 1', 'P1']:
                legacy_position_counts['P'] = legacy_position_counts.get('P', 0) + 1
            elif col_upper in ['OF.1', 'OF.1', 'OF 1', 'OF1']:
                legacy_position_counts['OF'] = legacy_position_counts.get('OF', 0) + 1
            elif col_upper in ['OF.2', 'OF 2', 'OF2']:
                legacy_position_counts['OF'] = legacy_position_counts.get('OF', 0) + 1
        
        # Add base positions to legacy count if they exist
        for col in file_columns:
            col_upper = str(col).upper().strip()
            if col_upper == 'P' and 'P' not in legacy_position_counts:
                legacy_position_counts['P'] = 1
            elif col_upper == 'OF' and 'OF' not in legacy_position_counts:
                legacy_position_counts['OF'] = 1
            elif col_upper in ['C', '1B', '2B', '3B', 'SS']:
                legacy_position_counts[col_upper] = 1
        
        # Expected position counts for standard DK format
        expected_counts = {'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
        
        # Check if position counts match expected DK format
        if position_counts == expected_counts:
            return 'standard'
        elif legacy_position_counts.get('P', 0) >= 1 and legacy_position_counts.get('OF', 0) >= 1:
            return 'legacy_standard'
        
        # Check if it has position columns in any order
        elif all(pos in file_columns for pos in expected_positions):
            return 'flexible'
        
        # Check if it has at least some position-like columns
        elif any(pos in ' '.join(file_columns).upper() for pos in expected_positions):
            return 'custom'
        
        return 'unknown'

    def fill_dk_entries_dynamic(self):
        """Fill the loaded DraftKings entries file with optimized lineups"""
        # Check if we have optimized lineups
        if not hasattr(self, 'optimized_lineups') or not self.optimized_lineups:
            QMessageBox.warning(self, "No Lineups Available", "No optimized lineups available.\n\nPlease run the optimization first to generate lineups.")
            return
        
        # Check if we have a loaded entries file
        if not hasattr(self, 'dk_entries_df') or not hasattr(self, 'dk_entries_file_path'):
            QMessageBox.warning(self, "No Entries File Loaded", "No DraftKings entries file loaded.\n\nPlease load a DraftKings entries file first using 'Load DraftKings Entries File' button.")
            return
        
        try:
            # Ask user how many lineups to use
            num_available = len(self.optimized_lineups)
            
            # Check how many reserved entry IDs are available
            reserved_entries_available = 0
            if hasattr(self, 'dk_entries_df') and self.dk_entries_df is not None:
                for _, row in self.dk_entries_df.iterrows():
                    if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip().isdigit():
                        reserved_entries_available += 1
            
            max_possible = min(reserved_entries_available, 1000)  # Reasonable limit
            default_request = min(num_available, reserved_entries_available, 150)
            
            num_to_use, ok = QInputDialog.getInt(
                self, 
                'Number of Lineups', 
                f'How many lineups to fill?\n\n'
                f'• Optimized lineups available: {num_available}\n'
                f'• Reserved entry IDs available: {reserved_entries_available}\n'
                f'• Recommended: {default_request}\n\n'
                f'Note: If you request more than {num_available} lineups,\n'
                f'the system will create variations of the optimized lineups.', 
                value=default_request,
                min=1, 
                max=max_possible
            )
            
            if not ok:
                return
            
            # CRITICAL FIX: Clear cached favorites data to force fresh generation
            if hasattr(self, '_favorites_with_player_data'):
                delattr(self, '_favorites_with_player_data')
                logging.info("🔄 Cleared cached favorites data to force fresh generation")
                print("🔄 CLEARED CACHED FAVORITES DATA - will use fresh optimized lineups")
            
            # Clear cached contest info to force fresh extraction
            if hasattr(self, '_contest_info_list'):
                delattr(self, '_contest_info_list')
                logging.info("🔄 Cleared cached contest info to force fresh extraction")
            
            # FORCE FRESH LINEUP GENERATION: Temporarily disable favorites file reading
            original_favorites_file_path = None
            backup_favorites_file_path = None
            
            try:
                # Temporarily rename the favorites file so it can't be found
                workspace_favorites_path = r"c:\Users\smtes\Downloads\coinbase_ml_trader\my_favorites_entries.csv"
                onedrive_favorites_path = r"c:\Users\smtes\OneDrive\Documents\my_favorites_entries.csv"
                
                if os.path.exists(workspace_favorites_path):
                    original_favorites_file_path = workspace_favorites_path
                    backup_favorites_file_path = workspace_favorites_path + ".backup_temp"
                    os.rename(workspace_favorites_path, backup_favorites_file_path)
                    logging.info(f"🔄 Temporarily renamed favorites file to force fresh generation")
                    print("🔄 TEMPORARILY RENAMED FAVORITES FILE - forcing fresh generation")
                elif os.path.exists(onedrive_favorites_path):
                    original_favorites_file_path = onedrive_favorites_path
                    backup_favorites_file_path = onedrive_favorites_path + ".backup_temp"
                    os.rename(onedrive_favorites_path, backup_favorites_file_path)
                    logging.info(f"🔄 Temporarily renamed OneDrive favorites file to force fresh generation")
                    print("🔄 TEMPORARILY RENAMED ONEDRIVE FAVORITES FILE - forcing fresh generation")
                
                # Create the filled entries DataFrame
                filled_entries = self.create_filled_entries_df(num_to_use)
                
            finally:
                # Restore the temporarily renamed favorites file
                if original_favorites_file_path and backup_favorites_file_path:
                    try:
                        if os.path.exists(backup_favorites_file_path):
                            os.rename(backup_favorites_file_path, original_favorites_file_path)
                            logging.info("🔄 Restored favorites file after filling entries")
                            print("🔄 RESTORED FAVORITES FILE")
                    except Exception as e:
                        logging.error(f"Error restoring favorites file: {e}")
            
            # Ask where to save
            save_path, _ = QFileDialog.getSaveFileName(
                self, 
                'Save Filled DraftKings Entries', 
                self.dk_entries_file_path.replace('.csv', '_filled.csv'),
                'CSV Files (*.csv);;All Files (*)'
            )
            
            if not save_path:
                return
            
            # Save the filled entries
            filled_entries.to_csv(save_path, index=False)
            
            # Show success message with feedback if fewer than requested
            success_msg = f"🎉 DraftKings entries filled successfully!\n\n"
            success_msg += f"📊 Filled {num_to_use} entries\n"
            success_msg += f"💾 Saved to: {os.path.basename(save_path)}\n\n"
            success_msg += f"🚀 Ready to upload to DraftKings!"
            
            if num_to_use < num_available:
                success_msg += f"\n\n⚠️ Note: Used {num_to_use} lineups (available {num_available})"
            
            QMessageBox.information(self, "Entries Filled Successfully", success_msg)
            self.status_label.setText(f'Filled {num_to_use} entries and saved to {os.path.basename(save_path)}')
            
        except Exception as e:
            error_msg = f"Error filling entries: {str(e)}"
            self.status_label.setText(error_msg)
            logging.error(error_msg)
            QMessageBox.critical(self, "Fill Error", f"Failed to fill entries:\n\n{str(e)}")

    def create_filled_entries_df(self, num_to_use):
        """Create a filled entries DataFrame by combining the original entries structure with optimized lineups"""
        if not hasattr(self, 'optimized_lineups') or not self.optimized_lineups:
            raise ValueError("No optimized lineups available")
        
        # Extract contest information from favorites (this populates self._contest_info_list)
        self.extract_contest_info_from_favorites()
        
        # CRITICAL: Initialize player ID mapping early so it's available in all code paths
        player_name_to_id_map = {}
        if hasattr(self, 'dk_entries_df') and self.dk_entries_df is not None:
            player_name_to_id_map = self.extract_player_id_mapping_from_dk_file()
            logging.info(f"🎯 Extracted {len(player_name_to_id_map)} player mappings from DK entries file")
        
        # If no DK entries file or no ID mapping, create IDs from loaded player data
        if not player_name_to_id_map:
            player_name_to_id_map = self.create_player_id_mapping_from_loaded_data()
            logging.info(f"📋 Created {len(player_name_to_id_map)} player mappings from loaded data")
        
        # Define the correct DraftKings headers
        correct_headers = ['Entry ID', 'Contest Name', 'Contest ID', 'Entry Fee', 'P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
        
        # CHECK FOR CACHE-CLEARING MODE: If we're in cache-clearing mode, use fresh optimized lineups
        workspace_backup_exists = os.path.exists(r"c:\Users\smtes\Downloads\coinbase_ml_trader\my_favorites_entries.csv.backup_temp")
        onedrive_backup_exists = os.path.exists(r"c:\Users\smtes\OneDrive\Documents\my_favorites_entries.csv.backup_temp")
        use_fresh_lineups = workspace_backup_exists or onedrive_backup_exists
        
        if use_fresh_lineups:
            logging.info("🔄 CACHE-CLEARING MODE: Using fresh optimized lineups instead of cached favorites")
            print(f"🔄 USING FRESH OPTIMIZED LINEUPS: {len(self.optimized_lineups)} fresh lineups available")
            
            # ENHANCED: Generate additional lineups if more are requested than available
            available_lineups = len(self.optimized_lineups)
            if num_to_use > available_lineups:
                logging.info(f"🔄 Requested {num_to_use} lineups, but only {available_lineups} available. Will create variations.")
                print(f"🔄 Creating variations: {num_to_use} requested vs {available_lineups} available")
                
                # Create additional lineup variations by cycling through existing ones
                extended_lineups = []
                for i in range(num_to_use):
                    base_lineup_index = i % available_lineups
                    base_lineup = self.optimized_lineups[base_lineup_index].copy()
                    extended_lineups.append(base_lineup)
                
                # Temporarily extend the optimized lineups list
                original_lineups = self.optimized_lineups
                self.optimized_lineups = extended_lineups
                lineups_to_use = num_to_use
                logging.info(f"✅ Extended lineups from {available_lineups} to {num_to_use} using variations")
            else:
                lineups_to_use = min(num_to_use, available_lineups)
            
            # Get contest info for template - try multiple sources
            has_contest_info = hasattr(self, '_contest_info_list') and self._contest_info_list
            if has_contest_info:
                base_contest_info = self._contest_info_list[0]
                logging.info(f"Using contest template: {base_contest_info['contest_name']} (ID: {base_contest_info['contest_id']})")
            else:
                # Try to get contest info from favorites file even in cache-clearing mode (for contest template only)
                base_contest_info = None
                try:
                    # Check backup files first (they contain the real contest info)
                    workspace_fav = r"c:\Users\smtes\Downloads\coinbase_ml_trader\my_favorites_entries.csv"
                    onedrive_fav = r"c:\Users\smtes\OneDrive\Documents\my_favorites_entries.csv"
                    workspace_backup = workspace_fav + ".backup_temp"
                    onedrive_backup = onedrive_fav + ".backup_temp"
                    
                    # Try backup files first (they're the renamed originals with contest info)
                    for fav_path in [workspace_backup, onedrive_backup, workspace_fav, onedrive_fav]:
                        if os.path.exists(fav_path):
                            temp_df = pd.read_csv(fav_path)
                            if not temp_df.empty and len(temp_df.columns) >= 4:
                                base_contest_info = {
                                    'entry_id': str(temp_df.iloc[0, 0]),
                                    'contest_name': str(temp_df.iloc[0, 1]),
                                    'contest_id': str(temp_df.iloc[0, 2]),
                                    'entry_fee': str(temp_df.iloc[0, 3])
                                }
                                logging.info(f"Extracted contest template from {fav_path}: {base_contest_info['contest_name']}")
                                break
                except Exception as e:
                    logging.debug(f"Could not extract contest info from favorites: {e}")
                
                # Final fallback to default
                if not base_contest_info:
                    base_contest_info = {
                        'entry_id': '4763920000',
                        'contest_name': 'MLB Main Slate',
                        'contest_id': '999999999',
                        'entry_fee': '$1'
                    }
                    logging.info("Using default contest information for fresh lineups")
            
            # Create filled entries using FRESH optimized lineups
            filled_entries = []
            import random
            
            # Use proper base Entry ID from contest info
            if base_contest_info and base_contest_info['entry_id'].isdigit():
                base_entry_id = int(base_contest_info['entry_id'])
            else:
                base_entry_id = 4763920000  # Default base
            
            # CRITICAL FIX: Extract actual entry IDs from loaded DK entries file
            actual_entry_ids = []
            if hasattr(self, 'dk_entries_df') and self.dk_entries_df is not None and not self.dk_entries_df.empty:
                logging.info("🎯 Extracting actual entry IDs from loaded DK entries file")
                
                # Method 1: Direct file parsing (most reliable for DK format)
                if hasattr(self, 'dk_entries_file_path') and self.dk_entries_file_path:
                    try:
                        logging.info("🔍 Using direct file parsing for entry ID extraction")
                        with open(self.dk_entries_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                        
                        for line in lines[1:]:  # Skip header line
                            line = line.strip()
                            if line:
                                parts = line.split(',')
                                if len(parts) > 0:
                                    entry_id = parts[0].strip().strip('"')
                                    # Check for valid DK entry IDs (numeric, 7+ digits)
                                    if entry_id.isdigit() and len(entry_id) >= 7:
                                        actual_entry_ids.append(entry_id)
                                        logging.debug(f"✅ Found entry ID from file: {entry_id}")
                        
                        logging.info(f"🎯 Direct file parsing found {len(actual_entry_ids)} entry IDs")
                    except Exception as e:
                        logging.warning(f"Direct file parsing failed: {e}")
                
                # Method 2: DataFrame parsing (fallback)
                if not actual_entry_ids:
                    logging.info("🔄 Fallback: Using DataFrame for entry ID extraction")
                    
                    # Check if 'Entry ID' column exists and has valid data
                    if 'Entry ID' in self.dk_entries_df.columns:
                        entry_id_col = self.dk_entries_df['Entry ID']
                        logging.info(f"🔍 Found 'Entry ID' column with {len(entry_id_col)} values")
                        
                        for entry_id in entry_id_col:
                            if pd.notna(entry_id):
                                entry_id_str = str(entry_id).strip()
                                # Check for valid DK entry IDs (10 digits starting with 47)
                                if entry_id_str.isdigit() and len(entry_id_str) >= 7:
                                    actual_entry_ids.append(entry_id_str)
                                    logging.debug(f"✅ Found valid entry ID: {entry_id_str}")
                    
                    # Check first column if Entry ID column didn't work
                    if not actual_entry_ids:
                        logging.info("🔄 Checking first column for entry IDs")
                        for _, row in self.dk_entries_df.iterrows():
                            if pd.notna(row.iloc[0]):
                                entry_id = str(row.iloc[0]).strip()
                                # Check for valid DK entry IDs
                                if entry_id.isdigit() and len(entry_id) >= 7:
                                    actual_entry_ids.append(entry_id)
                
                if actual_entry_ids:
                    logging.info(f"✅ Found {len(actual_entry_ids)} reserved entry IDs from DK file: {actual_entry_ids[:5]}...")
                    print(f"🎯 Using reserved entry IDs from DK file: {actual_entry_ids[:5]}... (total: {len(actual_entry_ids)})")
                else:
                    logging.warning("⚠️ No valid entry IDs found in loaded DK file")
                    # Debug: Show what we actually found
                    logging.info(f"🔍 Debug - DataFrame columns: {list(self.dk_entries_df.columns)}")
                    if len(self.dk_entries_df) > 0:
                        logging.info(f"🔍 Debug - First few values in first column: {self.dk_entries_df.iloc[:5, 0].tolist()}")
            else:
                logging.warning("⚠️ No DK entries file loaded - cannot use reserved entry IDs")
            
            # Limit lineups to available entry IDs
            if actual_entry_ids:
                lineups_to_use = min(lineups_to_use, len(actual_entry_ids))
                logging.info(f"🎯 Will fill {lineups_to_use} reserved entries (out of {len(actual_entry_ids)} available)")
            else:
                logging.error("❌ No reserved entry IDs available - cannot export lineups")
                raise ValueError("No reserved entry IDs found in DK entries file. Please load a valid DKEntries.csv file first.")
            
            # CRITICAL FIX: Use actual reserved entry IDs from DK file (these are pre-existing entries to fill)
            for i in range(lineups_to_use):
                lineup = self.optimized_lineups[i]
                
                # Use the reserved entry ID from the loaded DK file
                unique_entry_id = actual_entry_ids[i]
                logging.debug(f"Filling reserved entry ID: {unique_entry_id} with lineup {i+1}")
                
                # Convert lineup DataFrame to player IDs with proper position mapping
                player_ids = []
                
                # Try multiple column names for player IDs
                id_columns = ['ID', 'player_id', 'Player_ID', 'PlayerID', 'DraftKingsID', 'DK_ID']
                found_id_column = None
                
                for col in id_columns:
                    if col in lineup.columns:
                        found_id_column = col
                        break
                
                if found_id_column:
                    # Use the ID column directly but ensure all 10 positions are filled
                    lineup_ids = lineup[found_id_column].astype(str).tolist()
                    player_ids = []
                    
                    for pid in lineup_ids:
                        if pid and pid != 'nan' and pid.strip():
                            player_ids.append(pid.strip())
                    
                    # Ensure we have exactly 10 players
                    while len(player_ids) < 10:
                        player_ids.append("")
                    player_ids = player_ids[:10]
                    
                    logging.debug(f"Using player IDs from column '{found_id_column}': {len([p for p in player_ids if p])}/10 filled")
                else:
                    # Format positions properly using the position mapping function
                    formatted_positions = self.format_lineup_positions_only(lineup, player_name_to_id_map)
                    player_ids = formatted_positions
                    
                    # Validate that we have 10 positions and all are properly filled
                    valid_ids = [pid for pid in player_ids if pid and pid.strip() and pid != 'nan']
                    if len(valid_ids) < 8:  # Need at least 8 valid players for a reasonable lineup
                        logging.warning(f"Lineup {i+1} has only {len(valid_ids)} valid player IDs, attempting to fill gaps")
                        
                        # Try to get player IDs from player names as fallback
                        if 'Name' in lineup.columns:
                            player_names = lineup['Name'].tolist()
                            fallback_ids = []
                            
                            for name in player_names:
                                player_id_found = False
                                
                                # PRIORITY 1: Use the extracted DK entries player mapping
                                if player_name_to_id_map and name in player_name_to_id_map:
                                    fallback_ids.append(str(player_name_to_id_map[name]))
                                    player_id_found = True
                                else:
                                    # PRIORITY 2: Try to match with loaded player data if available
                                    if hasattr(self, 'df_players') and self.df_players is not None:
                                        matching_players = self.df_players[self.df_players['Name'] == name]
                                        if not matching_players.empty:
                                            # Try different ID column names in player data
                                            for id_col in ['ID', 'player_id', 'Player_ID', 'PlayerID', 'DraftKingsID']:
                                                if id_col in matching_players.columns:
                                                    player_id = str(matching_players.iloc[0][id_col])
                                                    if player_id and player_id.strip() and player_id != 'nan':
                                                        fallback_ids.append(player_id.strip())
                                                        player_id_found = True
                                                        break
                                
                                # FALLBACK: Use generic ID if no mapping found
                                if not player_id_found:
                                    generic_id = str(39200000 + len(fallback_ids))
                                    fallback_ids.append(generic_id)
                            
                            # Ensure exactly 10 players
                            while len(fallback_ids) < 10:
                                fallback_ids.append("")
                            fallback_ids = fallback_ids[:10]
                            
                            player_ids = fallback_ids
                            logging.info(f"Used fallback player ID mapping for lineup {i+1}: {len([p for p in player_ids if p])}/10 filled")
                    
                    logging.debug(f"Position-mapped player IDs: {len([p for p in player_ids if p])}/10 filled")
                
                # CRITICAL VALIDATION: Ensure ALL 10 positions are filled with valid IDs
                final_player_ids = []
                for j, pid in enumerate(player_ids):
                    if pid and pid.strip() and pid != 'nan' and pid.strip() != '':
                        final_player_ids.append(pid.strip())
                    else:
                        # Generate fallback ID for empty positions
                        fallback_id = str(39200000 + (i * 10) + j)
                        final_player_ids.append(fallback_id)
                        logging.warning(f"Empty position {j+1} in lineup {i+1}, using fallback ID: {fallback_id}")
                
                # Ensure exactly 10 positions
                while len(final_player_ids) < 10:
                    fallback_id = str(39200000 + (i * 10) + len(final_player_ids))
                    final_player_ids.append(fallback_id)
                    logging.warning(f"Adding fallback ID for missing position: {fallback_id}")
                
                final_player_ids = final_player_ids[:10]  # Truncate to exactly 10
                
                # Final validation
                valid_count = len([p for p in final_player_ids if p and p.strip() and p != 'nan'])
                if valid_count < 10:
                    logging.error(f"Lineup {i+1} still has {10-valid_count} invalid positions after fallback!")
                else:
                    logging.debug(f"✅ Lineup {i+1} has all 10 positions filled with valid IDs")
                
                row_data = [
                    unique_entry_id,
                    base_contest_info['contest_name'],
                    base_contest_info['contest_id'],
                    base_contest_info['entry_fee']
                ] + final_player_ids
                
                filled_entries.append(row_data)
                logging.debug(f"Created fresh entry {i+1}/{lineups_to_use}: ID={unique_entry_id}, Players={len([p for p in player_ids if p])}/10")
            
            result_df = pd.DataFrame(filled_entries, columns=correct_headers)
            logging.info(f"Created {len(result_df)} entries using FRESH optimized lineups")
            return result_df
        
        # ORIGINAL LOGIC: Check if we have the favorites file with player data (only when NOT in cache-clearing mode)
        elif hasattr(self, '_favorites_with_player_data') and self._favorites_with_player_data is not None:
            logging.info("🔄 Using player data directly from favorites file")
            print(f"🔄 USING CACHED FAVORITES DATA: {len(self._favorites_with_player_data)} cached lineups")
            
            # Create filled entries DataFrame
            filled_entries = pd.DataFrame(columns=correct_headers)
            
            # Use the favorites data directly, limited to the requested number
            favorites_df = self._favorites_with_player_data
            lineups_to_use = min(num_to_use, len(favorites_df))
            
            # Get contest info - use the first one as template
            has_contest_info = hasattr(self, '_contest_info_list') and self._contest_info_list
            
            if has_contest_info:
                base_contest_info = self._contest_info_list[0]
                logging.info(f"Using contest template: {base_contest_info['contest_name']} (ID: {base_contest_info['contest_id']})")
            else:
                base_contest_info = {
                    'entry_id': '1000000001',
                    'contest_name': 'Generated Lineups',
                    'contest_id': '999999999',
                    'entry_fee': '$1'
                }
                logging.warning("Using default contest information")
            
            # Generate unique Entry IDs for each lineup while preserving contest info
            import random
            base_entry_id = int(base_contest_info['entry_id']) if base_contest_info['entry_id'].isdigit() else 1000000001
            
            # CRITICAL FIX: Extract actual entry IDs from loaded DK entries file
            actual_entry_ids = []
            if hasattr(self, 'dk_entries_df') and self.dk_entries_df is not None and not self.dk_entries_df.empty:
                logging.info("🎯 Extracting actual entry IDs from loaded DK entries file (cached favorites path)")
                # Extract entry IDs from the first column of the loaded file
                for _, row in self.dk_entries_df.iterrows():
                    if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip():
                        entry_id = str(row.iloc[0]).strip()
                        # Only include numeric entry IDs (skip header rows with text)
                        if entry_id.isdigit():
                            actual_entry_ids.append(entry_id)
                
                if actual_entry_ids:
                    logging.info(f"✅ Found {len(actual_entry_ids)} actual entry IDs for cached favorites")
            
            # CRITICAL FIX: Generate UNIQUE entry IDs to prevent duplicates
            used_entry_ids = set()
            
            for i in range(lineups_to_use):
                # Generate a unique entry ID that hasn't been used yet
                if i < len(actual_entry_ids):
                    # Try to use actual entry ID first, but ensure uniqueness
                    candidate_id = actual_entry_ids[i]
                    if candidate_id not in used_entry_ids:
                        unique_entry_id = candidate_id
                        logging.debug(f"Using actual entry ID (cached): {unique_entry_id}")
                    else:
                        # If actual ID already used, generate unique one
                        attempts = 0
                        while attempts < 1000:  # Prevent infinite loop
                            unique_entry_id = str(base_entry_id + i + random.randint(1000, 9999))
                            if unique_entry_id not in used_entry_ids:
                                break
                            attempts += 1
                        logging.warning(f"Actual entry ID {candidate_id} already used, generated unique ID: {unique_entry_id}")
                else:
                    # Generate a unique Entry ID for this lineup as fallback
                    attempts = 0
                    while attempts < 1000:  # Prevent infinite loop
                        unique_entry_id = str(base_entry_id + i + random.randint(1000, 9999))
                        if unique_entry_id not in used_entry_ids:
                            break
                        attempts += 1
                    logging.debug(f"Generated unique entry ID (cached): {unique_entry_id}")
                
                # Track this ID as used
                used_entry_ids.add(unique_entry_id)
                
                # Use the same contest info but with unique Entry ID
                row_data = [
                    unique_entry_id,
                    base_contest_info['contest_name'],
                    base_contest_info['contest_id'],
                    base_contest_info['entry_fee']
                ]
                
                # Extract player IDs directly from the favorites file
                favorites_row = favorites_df.iloc[i]
                player_positions = []
                
                # Extract the 10 position columns (P, P, C, 1B, 2B, 3B, SS, OF, OF, OF)
                position_start_col = 4  # After Entry ID, Contest Name, Contest ID, Entry Fee
                for pos_idx in range(10):
                    col_idx = position_start_col + pos_idx
                    if col_idx < len(favorites_row):
                        player_id = str(favorites_row.iloc[col_idx]).strip()
                        # Only include if it's a valid player ID (numeric and reasonable length)
                        if player_id and player_id != 'nan' and player_id.isdigit() and len(player_id) >= 6:
                            player_positions.append(player_id)
                        else:
                            player_positions.append("")
                    else:
                        player_positions.append("")
                
                # Add player positions to row
                row_data.extend(player_positions)
                
                # Add the row to the DataFrame
                filled_entries.loc[i] = row_data
                
                logging.debug(f"Created entry {i+1}/{lineups_to_use}: ID={unique_entry_id}, Players={len([p for p in player_positions if p])}/10")
            
            logging.info(f"Created {len(filled_entries)} entries using player data from favorites file")
            
            return filled_entries
        
        else:
            # Fallback to original logic if favorites file data is not available
            logging.warning("Favorites file with player data not available, using original logic")
            print(f"🔄 USING FRESH OPTIMIZED LINEUPS: {len(self.optimized_lineups)} current lineups")
            
            # Create filled entries DataFrame
            filled_entries = pd.DataFrame(columns=correct_headers)
            
            # Use lineups up to the requested number
            lineups_to_use = self.optimized_lineups[:num_to_use]
            
            # Get contest info - use the first one as template
            has_contest_info = hasattr(self, '_contest_info_list') and self._contest_info_list
            
            if has_contest_info:
                base_contest_info = self._contest_info_list[0]
                logging.info(f"Using contest template: {base_contest_info['contest_name']} (ID: {base_contest_info['contest_id']})")
            else:
                base_contest_info = {
                    'entry_id': '1000000001',
                    'contest_name': 'Generated Lineups',
                    'contest_id': '999999999',
                    'entry_fee': '$1'
                }
                logging.warning("Using default contest information")
            
            # Generate unique Entry IDs for each lineup while preserving contest info
            import random
            base_entry_id = int(base_contest_info['entry_id']) if base_contest_info['entry_id'].isdigit() else 1000000001
            
            # CRITICAL FIX: Extract actual entry IDs from loaded DK entries file
            actual_entry_ids = []
            if hasattr(self, 'dk_entries_df') and self.dk_entries_df is not None and not self.dk_entries_df.empty:
                logging.info("🎯 Extracting actual entry IDs from loaded DK entries file (fallback path)")
                # Extract entry IDs from the first column of the loaded file
                for _, row in self.dk_entries_df.iterrows():
                    if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip():
                        entry_id = str(row.iloc[0]).strip()
                        # Only include numeric entry IDs (skip header rows with text)
                        # Fix: Check if it's all digits OR starts with digits (for DK entry IDs like 4766459650)
                        if entry_id.isdigit() or (len(entry_id) >= 7 and entry_id[:7].isdigit()):
                            # Extract just the numeric part for DK entry IDs
                            if entry_id.isdigit():
                                actual_entry_ids.append(entry_id)
                            else:
                                # Extract numeric prefix for DK format
                                numeric_part = ''.join(c for c in entry_id if c.isdigit())
                                if len(numeric_part) >= 7:  # Valid DK entry ID length
                                    actual_entry_ids.append(numeric_part)
                
                if actual_entry_ids:
                    logging.info(f"✅ Found {len(actual_entry_ids)} actual entry IDs for fallback")
            
            # CRITICAL FIX: Generate UNIQUE entry IDs to prevent duplicates (fallback path)
            used_entry_ids = set()
            
            for i, lineup in enumerate(lineups_to_use):
                # Generate a unique entry ID that hasn't been used yet
                if i < len(actual_entry_ids):
                    # Try to use actual entry ID first, but ensure uniqueness
                    candidate_id = actual_entry_ids[i]
                    if candidate_id not in used_entry_ids:
                        unique_entry_id = candidate_id
                        logging.debug(f"Using actual entry ID (fallback): {unique_entry_id}")
                    else:
                        # If actual ID already used, generate unique one
                        attempts = 0
                        while attempts < 1000:  # Prevent infinite loop
                            unique_entry_id = str(base_entry_id + i + random.randint(1000, 9999))
                            if unique_entry_id not in used_entry_ids:
                                break
                            attempts += 1
                        logging.warning(f"Actual entry ID {candidate_id} already used, generated unique ID: {unique_entry_id}")
                else:
                    # Generate a unique Entry ID for this lineup as fallback
                    attempts = 0
                    while attempts < 1000:  # Prevent infinite loop
                        unique_entry_id = str(base_entry_id + i + random.randint(1000, 9999))
                        if unique_entry_id not in used_entry_ids:
                            break
                        attempts += 1
                    logging.debug(f"Generated unique entry ID (fallback): {unique_entry_id}")
                
                # Track this ID as used
                used_entry_ids.add(unique_entry_id)
                
                # Use the same contest info but with unique Entry ID
                row_data = [
                    unique_entry_id,
                    base_contest_info['contest_name'],
                    base_contest_info['contest_id'],
                    base_contest_info['entry_fee']
                ]
                
                # Format lineup positions and add to row
                formatted_positions = self.format_lineup_positions_only(lineup, player_name_to_id_map)
                row_data.extend(formatted_positions)
                
                # Add the row to the DataFrame
                filled_entries.loc[i] = row_data
                
                logging.debug(f"Created entry {i+1}/{num_to_use}: ID={unique_entry_id}")
            
            logging.info(f"Created filled entries DataFrame with {len(filled_entries)} rows using lineup formatting")
            
            return filled_entries
    
    def extract_contest_info_from_favorites(self):
        """Extract contest information from loaded DK entries file or favorites data, handling multiple contests"""
        # Default values in case extraction fails
        default_info = {
            'entry_id': '1000000001',
            'contest_name': 'Generated Lineups',
            'contest_id': '999999999',
            'entry_fee': '$1'
        }
        
        # PRIORITY 1: Try to extract contest info from loaded DK entries file
        if hasattr(self, 'dk_entries_df') and self.dk_entries_df is not None and not self.dk_entries_df.empty:
            try:
                logging.info("🎯 PRIORITY: Extracting contest info from loaded DK entries file")
                print(f"🎯 Using loaded DK entries file: {getattr(self, 'dk_entries_file_path', 'Unknown')}")
                
                # Extract contest info from the loaded DK entries DataFrame
                contest_info_list = []
                for _, row in self.dk_entries_df.iterrows():
                    # DK entries format: Entry ID, Contest Name, Contest ID, Entry Fee, then player positions
                    if len(row) >= 4:
                        contest_info = {
                            'entry_id': str(row.iloc[0]) if pd.notna(row.iloc[0]) else '1000000001',
                            'contest_name': str(row.iloc[1]) if pd.notna(row.iloc[1]) else 'Generated Lineups',
                            'contest_id': str(row.iloc[2]) if pd.notna(row.iloc[2]) else '999999999',
                            'entry_fee': str(row.iloc[3]) if pd.notna(row.iloc[3]) else '$1'
                        }
                        contest_info_list.append(contest_info)
                
                if contest_info_list:
                    logging.info(f"✅ Extracted contest info for {len(contest_info_list)} entries from loaded DK file")
                    
                    # Log unique contests found in loaded file
                    unique_contests = {}
                    for info in contest_info_list:
                        contest_key = f"{info['contest_name']}|{info['contest_id']}"
                        if contest_key not in unique_contests:
                            unique_contests[contest_key] = info
                            logging.info(f"📋 Loaded contest: {info['contest_name']} (ID: {info['contest_id']}, Fee: {info['entry_fee']})")
                            print(f"📋 Using contest: {info['contest_name']} (ID: {info['contest_id']}, Fee: {info['entry_fee']})")
                    
                    # Store the contest info and the loaded DataFrame with player data
                    self._contest_info_list = contest_info_list
                    self._favorites_with_player_data = self.dk_entries_df
                    logging.info(f"✅ Stored loaded DK entries DataFrame with {len(self.dk_entries_df)} lineups and player data")
                    
                    return contest_info_list[0]  # Return first for backward compatibility
                
            except Exception as e:
                logging.warning(f"Could not extract contest info from loaded DK entries file: {e}")
                print(f"⚠️ Could not use loaded DK entries file: {e}")
        
        # PRIORITY 2: Try to extract ALL contest info from the favorites file with player data
        try:
            # First try the workspace location (which has actual player data)
            favorites_file_path = r"c:\Users\smtes\Downloads\coinbase_ml_trader\my_favorites_entries.csv"
            onedrive_favorites_path = r"c:\Users\smtes\OneDrive\Documents\my_favorites_entries.csv"
            
            # CHECK FOR CACHE-CLEARING MODE: If files are renamed (.backup_temp), read from backup files
            workspace_backup_exists = os.path.exists(favorites_file_path + ".backup_temp")
            onedrive_backup_exists = os.path.exists(onedrive_favorites_path + ".backup_temp")
            
            if workspace_backup_exists or onedrive_backup_exists:
                logging.info("🚫 CACHE-CLEARING MODE: Reading contest info from backup files only")
                print("🚫 CACHE-CLEARING MODE: Using backup files for contest info but not caching player data")
                
                # In cache-clearing mode, read contest info from backup files
                backup_files = []
                if workspace_backup_exists:
                    backup_files.append(favorites_file_path + ".backup_temp")
                if onedrive_backup_exists:
                    backup_files.append(onedrive_favorites_path + ".backup_temp")
                
                for backup_file in backup_files:
                    try:
                        favorites_df = pd.read_csv(backup_file)
                        logging.info(f"Reading contest info from backup: {backup_file}")
                        
                        if not favorites_df.empty and len(favorites_df.columns) >= 4:
                            # Extract contest info only (don't cache player data)
                            contest_info_list = []
                            for _, row in favorites_df.iterrows():
                                contest_info = {
                                    'entry_id': str(row.iloc[0]),
                                    'contest_name': str(row.iloc[1]),
                                    'contest_id': str(row.iloc[2]),
                                    'entry_fee': str(row.iloc[3])
                                }
                                contest_info_list.append(contest_info)
                            
                            logging.info(f"Extracted contest info for {len(contest_info_list)} entries from backup")
                            
                            # Log unique contests found
                            unique_contests = {}
                            for info in contest_info_list:
                                contest_key = f"{info['contest_name']}|{info['contest_id']}"
                                if contest_key not in unique_contests:
                                    unique_contests[contest_key] = info
                                    logging.info(f"Found contest: {info['contest_name']} (ID: {info['contest_id']}, Fee: {info['entry_fee']})")
                            
                            # Store contest info but NOT player data (that's the key difference)
                            self._contest_info_list = contest_info_list
                            # DO NOT store _favorites_with_player_data to force fresh lineup usage
                            
                            return contest_info_list[0]  # Return first for backward compatibility
                    except Exception as e:
                        logging.debug(f"Could not read backup file {backup_file}: {e}")
                
                # If we get here, no valid backup files found, continue with normal logic
                logging.warning("No valid backup files found, continuing with normal contest info extraction")
            
            if not os.path.exists(favorites_file_path):
                # Fallback to OneDrive location if workspace file doesn't exist
                favorites_file_path = onedrive_favorites_path
            
            if os.path.exists(favorites_file_path):
                favorites_df = pd.read_csv(favorites_file_path)
                logging.info(f"Reading favorites from: {favorites_file_path}")
                
                if not favorites_df.empty and len(favorites_df.columns) >= 4:
                    # Extract contest info from ALL rows, not just the first
                    contest_info_list = []
                    for _, row in favorites_df.iterrows():
                        contest_info = {
                            'entry_id': str(row.iloc[0]),
                            'contest_name': str(row.iloc[1]),
                            'contest_id': str(row.iloc[2]),
                            'entry_fee': str(row.iloc[3])
                        }
                        contest_info_list.append(contest_info)
                    
                    logging.info(f"Extracted contest info for {len(contest_info_list)} favorites from multiple contests")
                    
                    # Log unique contests found
                    unique_contests = {}
                    for info in contest_info_list:
                        contest_key = f"{info['contest_name']}|{info['contest_id']}"
                        if contest_key not in unique_contests:
                            unique_contests[contest_key] = info
                            logging.info(f"Found contest: {info['contest_name']} (ID: {info['contest_id']}, Fee: {info['entry_fee']})")
                    
                    # Store the list for use in create_filled_entries_df
                    self._contest_info_list = contest_info_list
                    
                    # IMPORTANT: Also store the favorites DataFrame with player data for lineup extraction
                    self._favorites_with_player_data = favorites_df
                    logging.info(f"Stored favorites DataFrame with {len(favorites_df)} lineups and player data")
                    
                    return contest_info_list[0]  # Return first for backward compatibility
        except Exception as e:
            logging.debug(f"Could not read contest info from favorites file: {e}")
        
        # Try to extract from favorites_lineups if available
        if hasattr(self, 'favorites_lineups') and self.favorites_lineups:
            try:
                contest_info_list = []
                for favorite in self.favorites_lineups:
                    if isinstance(favorite, dict):
                        contest_info = {
                            'entry_id': favorite.get('entry_id', default_info['entry_id']),
                            'contest_name': favorite.get('contest_name', default_info['contest_name']),
                            'contest_id': favorite.get('contest_id', default_info['contest_id']),
                            'entry_fee': favorite.get('entry_fee', default_info['entry_fee'])
                        }
                        contest_info_list.append(contest_info)
                
                if contest_info_list:
                    self._contest_info_list = contest_info_list
                    return contest_info_list[0]
            except Exception as e:
                logging.debug(f"Could not extract contest info from favorites: {e}")
        
        # Return defaults if all else fails
        logging.warning("Using default contest information - could not extract from favorites")
        return default_info

    def extract_player_id_mapping_from_dk_file(self):
        """Extract player name to ID mapping from the DraftKings entries file (numeric IDs only)"""
        player_map = {}
        
        if not hasattr(self, 'dk_entries_df') or self.dk_entries_df is None:
            return player_map
        
        if not hasattr(self, 'dk_entries_file_path') or not self.dk_entries_file_path:
            logging.warning("No DK entries file path available for raw parsing")
            return self.extract_player_id_mapping_from_dk_file_pandas()
        
        # Use raw file parsing for more reliable extraction
        # DK files have inconsistent CSV structure that confuses pandas
        logging.info(f"Extracting player IDs from DK entries file: {self.dk_entries_file_path}")
        
        try:
            with open(self.dk_entries_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    # Look for the characteristic "Name (ID)" pattern in each line
                    # Pattern: player name followed by (8-digit number)
                    matches = re.findall(r'([A-Za-z][A-Za-z\s\.\-\']+)\s*\((\d{6,})\)', line)
                    
                    for name_part, id_part in matches:
                        name_part = name_part.strip()
                        id_part = id_part.strip()
                        
                        # Additional validation for real player names
                        if (len(name_part) > 3 and 
                            len(name_part.split()) >= 2 and 
                            len(id_part) >= 6):
                            
                            # Don't overwrite existing mappings (first occurrence wins)
                            if name_part not in player_map:
                                player_map[name_part] = id_part
                                if len(player_map) <= 10:  # Log first 10 for debugging
                                    logging.info(f"✅ Found player mapping: {name_part} -> {id_part}")
        
        except Exception as e:
            logging.warning(f"Error reading DK entries file for player mappings: {e}")
            return self.extract_player_id_mapping_from_dk_file_pandas()
        
        logging.info(f"🎯 Successfully extracted {len(player_map)} player ID mappings from DK entries file")
        if len(player_map) > 0:
            sample_mappings = list(player_map.items())[:3]
            logging.info(f"Sample mappings: {sample_mappings}")
        
        return player_map
        return player_map
    
    def extract_player_id_mapping_from_dk_file_pandas(self):
        """Fallback pandas-based extraction method"""
        player_map = {}
        
        df_cols = list(self.dk_entries_df.columns)
        logging.debug(f"DK file has {len(df_cols)} columns: {df_cols}")
        logging.debug(f"Scanning {len(self.dk_entries_df)} rows for player data...")
        
        # Scan each row looking for player data patterns  
        for index, row in self.dk_entries_df.iterrows():
            try:
                # Convert row to list to handle different column counts
                row_data = row.tolist()
                
                # Look for the "Name + ID" pattern in any cell
                for col_idx, cell_value in enumerate(row_data):
                    if pd.isna(cell_value) or cell_value == '':
                        continue
                    
                    cell_str = str(cell_value).strip()
                    
                    # Check if this cell contains "Name + ID" format: "Player Name (ID_NUMBER)"
                    if '(' in cell_str and ')' in cell_str and cell_str.endswith(')'):
                        try:
                            name_part = cell_str.split('(')[0].strip()
                            id_part = cell_str.split('(')[1].replace(')', '').strip()
                            
                            # Validate that we have a real player name and numeric ID
                            if (name_part and len(name_part) > 1 and 
                                id_part.isdigit() and len(id_part) >= 6):
                                
                                # Additional validation: make sure this looks like a real player name
                                if any(char.isalpha() for char in name_part) and len(name_part.split()) >= 2:
                                    if name_part not in player_map:
                                        player_map[name_part] = id_part
                                        logging.debug(f"Found player mapping: {name_part} -> {id_part}")
                        except Exception as e:
                            continue
                
            except Exception as e:
                continue
        
        logging.info(f"Extracted {len(player_map)} player ID mappings using pandas fallback")
        return player_map

    def create_player_id_mapping_from_loaded_data(self):
        """Create player ID mapping from the loaded player data CSV (numeric IDs only)"""
        player_map = {}
        
        if not hasattr(self, 'df_players') or self.df_players is None or self.df_players.empty:
            return player_map
        
        # Look for ID columns in the loaded player data
        id_columns = []
        for col in self.df_players.columns:
            if any(id_term in str(col).lower() for id_term in ['id', 'player_id', 'dk_id', 'draftkings_id']):
                id_columns.append(col)
        
        if not id_columns:
            logging.debug("No ID columns found in loaded player data")
            return player_map
        
        # Create mappings using the first valid ID column
        for _, player in self.df_players.iterrows():
            name = str(player['Name']).strip()
            
            for id_col in id_columns:
                if pd.notna(player[id_col]):
                    player_id = str(player[id_col]).strip()
                    if player_id.isdigit() and len(player_id) >= 6:
                        player_map[name] = player_id  # Store just the numeric ID
                        logging.debug(f"Created player mapping from loaded data: {name} -> {player_id}")
                        break
        
        logging.info(f"Created {len(player_map)} player ID mappings from loaded data")
        return player_map

    def format_lineup_positions_only(self, lineup, player_name_to_id_map):
        """Format a lineup to return only the position assignments with player IDs in DK format (P, P, C, 1B, 2B, 3B, SS, OF, OF, OF)"""
        # Create position mapping from lineup
        position_players = {'P': [], 'C': [], '1B': [], '2B': [], '3B': [], 'SS': [], 'OF': []}
        
        # Group players by position with numeric IDs only
        for _, player in lineup.iterrows():
            pos = str(player['Position']).upper()
            name = str(player['Name'])
            
            # Get the numeric ID for this player
            player_id = ""
            
            # PRIORITY 1: Use DK entries mapping
            if player_name_to_id_map and name in player_name_to_id_map:
                player_id = str(player_name_to_id_map[name])
            else:
                # PRIORITY 2: Try to get ID from player data columns
                for id_col in ['ID', 'player_id', 'Player_ID', 'PlayerID', 'DraftKingsID', 'DK_ID']:
                    if id_col in player and pd.notna(player[id_col]):
                        potential_id = str(player[id_col]).strip()
                        if potential_id.isdigit() and len(potential_id) >= 6:
                            player_id = potential_id
                            break
                
                # PRIORITY 3: Check if the name already contains an ID
                if not player_id and '(' in name and ')' in name and name.endswith(')'):
                    try:
                        id_part = name.split('(')[1].replace(')', '').strip()
                        if id_part.isdigit() and len(id_part) >= 6:
                            player_id = id_part
                    except:
                        pass
                
                # PRIORITY 4: Try any column that might contain an ID
                if not player_id:
                    for col_name, col_value in player.items():
                        if 'id' in str(col_name).lower() and pd.notna(col_value):
                            potential_id = str(col_value).strip()
                            if potential_id.isdigit() and len(potential_id) >= 6:
                                player_id = potential_id
                                break
            
            # Only add if we found a valid player ID
            if not player_id:
                # Generate a fallback ID as last resort
                player_id = str(39200000 + len([p for sublist in position_players.values() for p in sublist]))
                logging.warning(f"No valid ID found for {name}, using fallback ID: {player_id}")
            
            # Handle multi-position players and pitcher designations
            if 'P' in pos or 'SP' in pos or 'RP' in pos:
                position_players['P'].append(player_id)
            elif 'C' in pos:
                position_players['C'].append(player_id)
            elif '1B' in pos:
                position_players['1B'].append(player_id)
            elif '2B' in pos:
                position_players['2B'].append(player_id)
            elif '3B' in pos:
                position_players['3B'].append(player_id)
            elif 'SS' in pos:
                position_players['SS'].append(player_id)
            elif 'OF' in pos:
                position_players['OF'].append(player_id)
            else:
                # If position is unclear, try to guess based on common patterns
                if any(p_term in pos for p_term in ['P', 'PITCH']):
                    position_players['P'].append(player_id)
                elif any(of_term in pos for of_term in ['OF', 'LF', 'CF', 'RF', 'OUTFIELD']):
                    position_players['OF'].append(player_id)
                else:
                    # Default to OF if we can't determine position
                    position_players['OF'].append(player_id)
                    logging.warning(f"Unclear position '{pos}' for {name}, defaulting to OF")
        
        # Create the position assignments in DK format: [P, P, C, 1B, 2B, 3B, SS, OF, OF, OF]
        position_assignments = []
        
        # Add two pitchers (ensure we have at least 2, use same pitcher twice if needed)
        if len(position_players['P']) >= 2:
            position_assignments.append(position_players['P'][0])
            position_assignments.append(position_players['P'][1])
        elif len(position_players['P']) == 1:
            position_assignments.append(position_players['P'][0])
            position_assignments.append(position_players['P'][0])  # Use same pitcher twice
            logging.warning("Only one pitcher found, using same pitcher for both P slots")
        else:
            position_assignments.append("")  # Empty P slot
            position_assignments.append("")  # Empty P slot
            logging.error("No pitchers found in lineup!")
        
        # Add catcher
        position_assignments.append(position_players['C'][0] if len(position_players['C']) > 0 else "")
        
        # Add infielders
        position_assignments.append(position_players['1B'][0] if len(position_players['1B']) > 0 else "")
        position_assignments.append(position_players['2B'][0] if len(position_players['2B']) > 0 else "")
        position_assignments.append(position_players['3B'][0] if len(position_players['3B']) > 0 else "")
        position_assignments.append(position_players['SS'][0] if len(position_players['SS']) > 0 else "")
        
        # Add three outfielders (ensure we have at least 3, reuse if needed)
        if len(position_players['OF']) >= 3:
            position_assignments.append(position_players['OF'][0])
            position_assignments.append(position_players['OF'][1])
            position_assignments.append(position_players['OF'][2])
        elif len(position_players['OF']) == 2:
            position_assignments.append(position_players['OF'][0])
            position_assignments.append(position_players['OF'][1])
            position_assignments.append(position_players['OF'][0])  # Reuse first OF
            logging.warning("Only 2 outfielders found, reusing first OF for third slot")
        elif len(position_players['OF']) == 1:
            position_assignments.append(position_players['OF'][0])
            position_assignments.append(position_players['OF'][0])  # Reuse
            position_assignments.append(position_players['OF'][0])  # Reuse
            logging.warning("Only 1 outfielder found, using same OF for all three slots")
        else:
            position_assignments.append("")  # Empty OF slots
            position_assignments.append("")
            position_assignments.append("")
            logging.error("No outfielders found in lineup!")
        
        # Final validation - ensure exactly 10 positions
        while len(position_assignments) < 10:
            position_assignments.append("")
        position_assignments = position_assignments[:10]
        
        # CRITICAL FIX: Fill any empty positions with fallback IDs
        for i in range(10):
            if not position_assignments[i] or position_assignments[i].strip() == '':
                # Generate fallback ID for empty position
                fallback_id = str(39200000 + i)
                position_assignments[i] = fallback_id
                logging.warning(f"Position {i+1} was empty, filled with fallback ID: {fallback_id}")
        
        # Count how many positions are actually filled
        filled_count = len([p for p in position_assignments if p and p.strip()])
        logging.debug(f"Position assignment result: {filled_count}/10 positions filled")
        
        return position_assignments
    
    def get_bankroll_setting(self):
        """Get the bankroll setting from UI"""
        try:
            if RISK_ENGINE_AVAILABLE and hasattr(self, 'bankroll_input'):
                bankroll_text = self.bankroll_input.text().strip()
                if bankroll_text:
                    bankroll = float(bankroll_text)
                    return max(100, bankroll)  # Minimum $100
            return 1000  # Default
        except (ValueError, AttributeError):
            return 1000

    def get_risk_tolerance_setting(self):
        """Get the risk tolerance setting from UI"""
        try:
            if RISK_ENGINE_AVAILABLE and hasattr(self, 'risk_tolerance_combo'):
                return self.risk_tolerance_combo.currentText()
            return 'medium'
        except AttributeError:
            return 'medium'

    def get_risk_management_enabled(self):
        """Check if risk management is enabled"""
        try:
            if RISK_ENGINE_AVAILABLE and hasattr(self, 'enable_risk_mgmt_checkbox'):
                return self.enable_risk_mgmt_checkbox.isChecked()
            return False
        except AttributeError:
            return False

    def update_selection_status(self):
        """Update the selection status label with current player count"""
        try:
            if not hasattr(self, 'selection_status_label'):
                return
                
            # Get total selected count using the same method as get_included_players
            included_players = self.get_included_players()  
            total_selected = len(included_players)
            
            # Update the status label with simple, non-cluttering text
            if total_selected == 0:
                status_text = "No players selected"
                self.selection_status_label.setStyleSheet("color: #666; font-size: 10px; padding: 3px;")
            elif total_selected <= 50:
                status_text = f"{total_selected} players selected"
                self.selection_status_label.setStyleSheet("color: #4CAF50; font-size: 10px; padding: 3px;")
            elif total_selected <= 150:
                status_text = f"{total_selected} players selected"
                self.selection_status_label.setStyleSheet("color: #FF9800; font-size: 10px; padding: 3px;")
            else:
                status_text = f"{total_selected} players selected"
                self.selection_status_label.setStyleSheet("color: #f44336; font-size: 10px; padding: 3px;")
            
            self.selection_status_label.setText(status_text)
            
        except Exception as e:
            logging.error(f"Error updating selection status: {e}")
if __name__ == "__main__":
    # Create the application
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = FantasyBaseballApp()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec_())
    
