from typing import Dict, Optional
import logging
from datetime import datetime, timedelta
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenSystem:
    def __init__(self, storage_path: str = "token_data.json"):
        self.storage_path = storage_path
        self.balances: Dict[str, float] = {}
        self.compute_history: Dict[str, list] = {}
        self.token_rate = 1.0  # Tokens per hour of compute
        self._load_data()

    def _load_data(self):
        """Load token data from storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.balances = data.get('balances', {})
                    self.compute_history = data.get('compute_history', {})
        except Exception as e:
            logger.error(f"Error loading token data: {str(e)}")

    def _save_data(self):
        """Save token data to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'balances': self.balances,
                    'compute_history': self.compute_history
                }, f)
        except Exception as e:
            logger.error(f"Error saving token data: {str(e)}")

    def get_balance(self, user_id: str) -> float:
        """Get token balance for a user"""
        return self.balances.get(user_id, 0.0)

    def add_tokens(self, user_id: str, amount: float) -> bool:
        """Add tokens to a user's balance"""
        try:
            current_balance = self.balances.get(user_id, 0.0)
            self.balances[user_id] = current_balance + amount
            self._save_data()
            return True
        except Exception as e:
            logger.error(f"Error adding tokens: {str(e)}")
            return False

    def spend_tokens(self, user_id: str, amount: float) -> bool:
        """Spend tokens from a user's balance"""
        try:
            current_balance = self.balances.get(user_id, 0.0)
            if current_balance < amount:
                return False
            self.balances[user_id] = current_balance - amount
            self._save_data()
            return True
        except Exception as e:
            logger.error(f"Error spending tokens: {str(e)}")
            return False

    def record_compute_contribution(self, user_id: str, node_id: str, 
                                 start_time: datetime, end_time: datetime,
                                 resources: Dict[str, float]):
        """Record compute resources shared by a user"""
        try:
            duration = (end_time - start_time).total_seconds() / 3600  # Convert to hours
            
            # Calculate token reward based on resources and time
            reward = self._calculate_reward(duration, resources)
            
            # Record contribution
            if user_id not in self.compute_history:
                self.compute_history[user_id] = []
                
            self.compute_history[user_id].append({
                'node_id': node_id,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration': duration,
                'resources': resources,
                'tokens_earned': reward
            })
            
            # Add tokens to user's balance
            self.add_tokens(user_id, reward)
            self._save_data()
            
            return reward
            
        except Exception as e:
            logger.error(f"Error recording compute contribution: {str(e)}")
            return 0.0

    def _calculate_reward(self, duration: float, resources: Dict[str, float]) -> float:
        """Calculate token reward based on compute contribution"""
        try:
            # Basic reward calculation
            base_reward = duration * self.token_rate
            
            # Adjust reward based on resources
            resource_multiplier = 1.0
            if 'gpu' in resources:
                resource_multiplier *= 2.0  # Double reward for GPU resources
            if 'cpu_cores' in resources:
                resource_multiplier *= (1.0 + resources['cpu_cores'] / 10.0)  # Increase reward with CPU cores
                
            return base_reward * resource_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating reward: {str(e)}")
            return 0.0

    def check_compute_access(self, user_id: str, required_tokens: float) -> bool:
        """Check if user has enough tokens for compute access"""
        return self.get_balance(user_id) >= required_tokens

    def get_compute_history(self, user_id: str) -> list:
        """Get compute contribution history for a user"""
        return self.compute_history.get(user_id, [])

    def estimate_compute_cost(self, resources: Dict[str, float], 
                            duration_hours: float) -> float:
        """Estimate token cost for compute resources"""
        try:
            # Basic cost calculation based on duration and resources
            base_cost = duration_hours * self.token_rate
            
            # Adjust cost based on resources
            resource_multiplier = 1.0
            if 'gpu' in resources:
                resource_multiplier *= 2.0
            if 'cpu_cores' in resources:
                resource_multiplier *= (1.0 + resources['cpu_cores'] / 10.0)
                
            return base_cost * resource_multiplier
            
        except Exception as e:
            logger.error(f"Error estimating compute cost: {str(e)}")
            return 0.0 