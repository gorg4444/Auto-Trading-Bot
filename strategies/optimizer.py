import logging
import random
import asyncio
import numpy as np # Required for Sharpe Ratio calculation
from datetime import datetime, timedelta
from .renko_strategy import OptimizedRenkoStrategy
from .market_simulator import SimulatedAlpacaClient

class OptimizationEngine:
    """
    Uses a genetic algorithm with Walk-Forward Validation and Sharpe Ratio
    to find the most robust and consistently profitable strategy parameters.
    """
    def __init__(self, symbols, socketio_instance):
        self.symbols = symbols
        self.socketio = socketio_instance

        # Parameter space for optimization
        self.param_space = {
            'adx_threshold': (20, 35),
            'trending_atr_multiplier': (0.8, 2.0),
            'ranging_atr_multiplier': (0.3, 1.0),
            'stop_loss_atr_multiplier': (2.0, 4.0)
        }
        
        # Genetic Algorithm Settings
        self.population_size = 20
        self.generations = 10
        self.mutation_rate = 0.1

    def _create_individual(self):
        """Creates a single random set of parameters."""
        return {
            'adx_threshold': random.randint(*self.param_space['adx_threshold']),
            'trending_atr_multiplier': round(random.uniform(*self.param_space['trending_atr_multiplier']), 2),
            'ranging_atr_multiplier': round(random.uniform(*self.param_space['ranging_atr_multiplier']), 2),
            'stop_loss_atr_multiplier': round(random.uniform(*self.param_space['stop_loss_atr_multiplier']), 2),
        }

    async def _run_simulation(self, params):
        """
        Runs a full simulation and returns the Sharpe Ratio and final equity.
        Uses Walk-Forward Validation by training on 2 years of data and validating on the last year.
        """
        # --- NEW: Walk-Forward Validation ---
        # We now download 3 years of data.
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)
        
        # The last year (approx 252 days) is our unseen "validation" set.
        validation_start_date = end_date - timedelta(days=365)

        sim_client = SimulatedAlpacaClient(symbols=self.symbols, silent_logging=True)
        if not sim_client.load_historical_data(start=start_date, end=end_date):
            return 0, 0 # Return 0 for both Sharpe and equity

        strategy = OptimizedRenkoStrategy(
            api_key=None, api_secret=None, base_url=None,
            symbols=self.symbols, client=sim_client, **params
        )
        
        equity_history = []
        while sim_client.tick():
            # Only record equity during the validation period
            current_sim_date = sim_client.current_date.to_pydatetime().replace(tzinfo=None)
            if current_sim_date >= validation_start_date:
                equity_history.append(sim_client.get_account().equity)

            snapshots = sim_client.get_snapshots(strategy.symbols)
            tasks = []
            for symbol, snapshot in snapshots.items():
                if snapshot and snapshot.minute_bar:
                    ohlc_data = {
                        'open': snapshot.minute_bar.o, 'high': snapshot.minute_bar.h,
                        'low': snapshot.minute_bar.l, 'close': snapshot.minute_bar.c
                    }
                    try:
                        position = sim_client.get_position(symbol)
                    except Exception:
                        position = None
                    
                    bars_data = sim_client.get_bars(symbol, '1D', limit=strategy.adx_period * 3).df
                    daily_bars_data = sim_client.get_bars(symbol, '1D', limit=strategy.long_sma_period + 5).df
                    tasks.append(strategy.run_for_symbol(symbol, ohlc_data, position, bars_data, daily_bars_data))
            
            if tasks:
                await asyncio.gather(*tasks)

        # --- NEW: Calculate Sharpe Ratio ---
        # The Sharpe Ratio measures risk-adjusted return. Higher is better.
        if len(equity_history) > 1:
            returns = np.diff(equity_history) / equity_history[:-1]
            # Assuming a risk-free rate of 0 for simplicity
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
            
        final_equity = sim_client.get_account().equity
        return sharpe_ratio, final_equity

    async def run_optimization(self):
        """The main loop for the genetic algorithm."""
        population = [self._create_individual() for _ in range(self.population_size)]
        best_individual = None
        best_fitness = -1 # Start with a negative fitness
        best_equity = 0

        for gen in range(self.generations):
            fitness_scores = []
            for i, individual in enumerate(population):
                self.socketio.emit('optimization_update', {
                    'status': 'running',
                    'message': f"Generation {gen + 1}/{self.generations}: Sim {i + 1}/{self.population_size}..."
                })
                # --- MODIFIED: The fitness score is now the Sharpe Ratio ---
                sharpe_ratio, final_equity = await self._run_simulation(individual)
                fitness_scores.append((sharpe_ratio, final_equity, individual))
                await asyncio.sleep(0.01)

            # Sort by the fitness score (Sharpe Ratio)
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            
            current_best_sharpe = fitness_scores[0][0]
            if current_best_sharpe > best_fitness:
                best_fitness = current_best_sharpe
                best_individual = fitness_scores[0][2]
                best_equity = fitness_scores[0][1]

            self.socketio.emit('optimization_update', {
                'status': 'running',
                'message': f"Gen {gen + 1} complete. Best Sharpe Ratio: {best_fitness:.2f} (Profit: ${best_equity:,.2f})"
            })
            await asyncio.sleep(1)

            parents = [ind for _, _, ind in fitness_scores[:self.population_size // 2]]
            next_generation = parents[:]

            while len(next_generation) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                child = {}
                for key in self.param_space:
                    child[key] = random.choice([p1[key], p2[key]])
                    if random.random() < self.mutation_rate:
                        if isinstance(child[key], int):
                            child[key] = random.randint(*self.param_space[key])
                        else:
                            child[key] = round(random.uniform(*self.param_space[key]), 2)
                next_generation.append(child)
            
            population = next_generation

        logging.info(f"Optimization complete. Best parameters found: {best_individual}")
        self.socketio.emit('optimization_update', {
            'status': 'complete',
            'message': "Optimization Finished!",
            'best_params': best_individual,
            'best_profit': best_equity,
            'best_sharpe': best_fitness
        })
        return best_individual