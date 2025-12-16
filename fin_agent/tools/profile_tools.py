import json
from fin_agent.user_profile import UserProfileManager

# Global instance
profile_manager = UserProfileManager()

def update_user_profile(risk_tolerance=None, investment_horizon=None, favorite_sectors=None, avoid_sectors=None, investment_style=None):
    """
    Update the user's investment profile and preferences.
    """
    return profile_manager.update_profile(
        risk_tolerance=risk_tolerance,
        investment_horizon=investment_horizon,
        favorite_sectors=favorite_sectors,
        avoid_sectors=avoid_sectors,
        investment_style=investment_style
    )

def get_user_profile():
    """
    Get the current user profile settings.
    """
    return json.dumps(profile_manager.get_profile(), ensure_ascii=False, indent=2)

# Tool definitions
PROFILE_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": "Update the user's investment profile. Call this when the user explicitly states their preferences or when you infer them from the conversation (e.g., 'I prefer low risk', 'I like tech stocks').",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_tolerance": {
                        "type": "string",
                        "enum": ["Conservative", "Balanced", "Aggressive", "Unknown"],
                        "description": "User's risk tolerance level."
                    },
                    "investment_horizon": {
                        "type": "string",
                        "enum": ["Short-term", "Medium-term", "Long-term", "Unknown"],
                        "description": "User's expected investment duration."
                    },
                    "favorite_sectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of sectors the user is interested in."
                    },
                    "avoid_sectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of sectors the user wants to avoid."
                    },
                    "investment_style": {
                        "type": "string",
                        "description": "A brief textual description of the user's investment style or specific preferences."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "Retrieve the current user profile and preferences.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

