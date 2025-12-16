import json
import os
from typing import Dict, Any, Optional
from fin_agent.config import Config

class UserProfileManager:
    def __init__(self, file_path: str = None):
        if file_path:
            self.file_path = file_path
        else:
            # Default to user config directory
            config_dir = Config.get_config_dir()
            os.makedirs(config_dir, exist_ok=True)
            self.file_path = os.path.join(config_dir, "user_profile.json")
            
        self.profile = self._load_profile()

    def _load_profile(self) -> Dict[str, Any]:
        if not os.path.exists(self.file_path):
            return self._default_profile()
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return self._default_profile()

    def _default_profile(self) -> Dict[str, Any]:
        return {
            "risk_tolerance": "Unknown", # Conservative, Balanced, Aggressive
            "investment_horizon": "Unknown", # Short-term, Medium-term, Long-term
            "favorite_sectors": [],
            "avoid_sectors": [],
            "investment_style": "", # Free text description
            "custom_preferences": {} # Any other key-value pairs
        }

    def _save_profile(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.profile, f, ensure_ascii=False, indent=2)

    def update_profile(self, 
                       risk_tolerance: Optional[str] = None,
                       investment_horizon: Optional[str] = None,
                       favorite_sectors: Optional[list] = None,
                       avoid_sectors: Optional[list] = None,
                       investment_style: Optional[str] = None,
                       **kwargs):
        """
        Update user profile fields.
        """
        if risk_tolerance:
            self.profile["risk_tolerance"] = risk_tolerance
        if investment_horizon:
            self.profile["investment_horizon"] = investment_horizon
        if favorite_sectors is not None:
            self.profile["favorite_sectors"] = favorite_sectors
        if avoid_sectors is not None:
            self.profile["avoid_sectors"] = avoid_sectors
        if investment_style:
            self.profile["investment_style"] = investment_style
            
        # Update custom preferences
        for k, v in kwargs.items():
            self.profile["custom_preferences"][k] = v
            
        self._save_profile()
        return "User profile updated successfully."

    def get_profile_summary(self) -> str:
        """
        Return a string summary of the user profile for LLM context.
        """
        p = self.profile
        summary = f"""
User Profile:
- Risk Tolerance: {p.get('risk_tolerance', 'Unknown')}
- Investment Horizon: {p.get('investment_horizon', 'Unknown')}
- Favorite Sectors: {', '.join(p.get('favorite_sectors', []))}
- Avoid Sectors: {', '.join(p.get('avoid_sectors', []))}
- Investment Style: {p.get('investment_style', 'Not specified')}
"""
        custom = p.get("custom_preferences", {})
        if custom:
            summary += "- Other Preferences:\n"
            for k, v in custom.items():
                summary += f"  - {k}: {v}\n"
        
        return summary.strip()

    def get_profile(self) -> Dict[str, Any]:
        return self.profile

