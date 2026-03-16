from typing import Dict, Any, List, Optional

class NarrativeGenerator:
    """
    Translates technical AI events into business-friendly narratives and insights.
    """
    def __init__(self):
        self.history = []
        # Mapping technical ops to business concepts
        self.op_meanings = {
            'relu': 'a harsh cutoff filter',
            'tanh': 'a smooth, constrained transition',
            'sigmoid': 'a probability-like gate',
            'sin': 'a cyclical/seasonal pattern',
            'cos': 'a cyclical/seasonal pattern',
            'gaussian': 'a bell-curve distribution',
            'identity': 'a direct pass-through',
            'multiply': 'an interaction effect',
            'add': 'a cumulative effect'
        }

    def generate_event_log(self, event_type: str, details: Dict[str, Any]) -> Dict[str, str]:
        """
        Generates a human-readable log entry for a system event.
        
        Args:
            event_type: 'mutation', 'stagnation', 'breakthrough', 'start', 'finish'
            details: Context specific to the event (e.g., old_op, new_op, score_delta)
            
        Returns:
            Dict containing 'timestamp', 'message', 'insight_type'
        """
        message = ""
        insight_type = "info" # info, success, warning, alert
        
        if event_type == 'mutation':
            old_op = details.get('old_op', '').lower()
            new_op = details.get('new_op', '').lower()
            delta = details.get('score_delta', 0.0)
            
            # Basic Translation
            meaning = self._interpret_mutation(old_op, new_op)
            
            if delta > 0:
                message = f"Evolution Successful: {meaning} This improved accuracy by {delta:.4f}."
                insight_type = "success"
                
                # Check for specific "Insight Alerts"
                alert = self._check_insight_alerts(new_op, delta)
                if alert:
                    message += f" \n\n{alert}"
                    insight_type = "alert"
            else:
                message = f"Experiment Failed: {meaning} discarded as it did not improve results."
                insight_type = "info"

        elif event_type == 'stagnation':
            message = "The model has plateaued. Increasing 'Temperature' to encourage riskier, more creative mutations."
            insight_type = "warning"
            
        elif event_type == 'breakthrough':
            img_pct = details.get('improvement_pct', 0)
            message = f"🚀 MAJOR BREAKTHROUGH: Found a structural change that improved performance by {img_pct:.1f}%!"
            insight_type = "alert"

        return {
            "message": message,
            "type": insight_type,
            "technical_details": str(details)
        }

    def _interpret_mutation(self, old_op: str, new_op: str) -> str:
        """Helper to create the sentence for a mutation."""
        
        # Detect composite parts for richer explanation
        # e.g. "sin_of_tanh"
        
        narrative = f"Changed '{old_op}' to '{new_op}'."
        
        if "sin" in new_op or "cos" in new_op:
            narrative = f"applied a Cyclical Transformation ({new_op}) to capture seasonality."
        elif "gaussian" in new_op:
            narrative = "switched to a Gaussian function to model normal distributions."
        elif "tanh" in new_op and "relu" in old_op:
            narrative = "smoothed out the response curve (swapping ReLU for Tanh) to handle outliers better."
        elif "multiply" in new_op:
            narrative = "introduced a Multiplicative Interaction to capture synergy between features."
            
        return narrative

    def _check_insight_alerts(self, new_op: str, delta: float) -> Optional[str]:
        """Checks for the 3 specific Insight Alerts defined in the plan."""
        
        # 1. Simplicity Alert (Not easy to track count here without more state, skipping for now or approximating)
        # We'll rely on delta magnitude for "Breakthroughs" mostly.
        
        # 2. Physics Alert
        physics_ops = ['gaussian', 'decay', 'sin', 'cos']
        if any(p in new_op for p in physics_ops) and delta > 0.05: # High impact
            return "⚛️ **Natural Law Detected**: The model discovered a pattern common in physics (Wave/Distribution)."

        # 3. Efficiency Breakthrough (High impact with simple op)
        simple_ops = ['identity', 'relu', 'tanh']
        if new_op in simple_ops and delta > 0.10:
             return "📉 **Efficiency Breakthrough**: Achieved a massive gain using a very simple mathematical structure."
             
        return None
