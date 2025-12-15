import os
import sys
import platform
from pathlib import Path
from dotenv import load_dotenv

class Config:
    TUSHARE_TOKEN = None
    LLM_PROVIDER = None
    LLM_STREAM = True
    
    # DeepSeek Config
    DEEPSEEK_API_KEY = None
    DEEPSEEK_BASE_URL = None
    DEEPSEEK_MODEL = None
    
    # Generic OpenAI Config
    OPENAI_API_KEY = None
    OPENAI_BASE_URL = None
    OPENAI_MODEL = None
    
    # Email Config
    EMAIL_SMTP_SERVER = None
    EMAIL_SMTP_PORT = None
    EMAIL_SENDER = None
    EMAIL_PASSWORD = None
    EMAIL_RECEIVER = None
    
    @staticmethod
    def get_config_dir():
        """Get the configuration directory based on the operating system."""
        app_name = "fin-agent"
        
        if platform.system() == "Windows":
            base_dir = os.getenv("APPDATA")
            if not base_dir:
                base_dir = os.path.expanduser("~")
            config_dir = os.path.join(base_dir, app_name)
        else:
            # Linux/Mac (XDG standard)
            # Default to ~/.config/fin-agent/.env
            base_dir = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
            config_dir = os.path.join(base_dir, app_name)
            
        return config_dir

    @staticmethod
    def get_env_path():
        """Get the full path to the .env file."""
        return os.path.join(Config.get_config_dir(), ".env")

    @classmethod
    def load(cls):
        # Explicitly define path to ensure we are loading the right file
        env_path = cls.get_env_path()
        
        # Also try loading from current directory for local overrides/dev
        local_env = os.path.join(os.getcwd(), ".env")
        if os.path.exists(local_env):
            load_dotenv(local_env, override=True)
            
        # Load from user config dir
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
        
        cls.TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
        cls.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
        cls.LLM_STREAM = os.getenv("LLM_STREAM", "True").lower() == "true"
        
        cls.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        cls.DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        cls.DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        cls.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        cls.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
        cls.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        cls.EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
        cls.EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "465")) if os.getenv("EMAIL_SMTP_PORT") else 465
        cls.EMAIL_SENDER = os.getenv("EMAIL_SENDER")
        cls.EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
        cls.EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

    @classmethod
    def validate(cls):
        cls.load() # Ensure latest env is loaded
        missing = []
        if not cls.TUSHARE_TOKEN:
            missing.append("TUSHARE_TOKEN")
            
        if cls.LLM_PROVIDER == "deepseek":
            if not cls.DEEPSEEK_API_KEY:
                missing.append("DEEPSEEK_API_KEY")
        elif cls.LLM_PROVIDER == "openai" or cls.LLM_PROVIDER == "local":
             if not cls.OPENAI_API_KEY and cls.LLM_PROVIDER != "local":
                # For strictly OpenAI provider, key is usually required.
                # For 'local', we might allow empty key if user really wants to, 
                # but standard check:
                pass
        
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
            
    @staticmethod
    def is_email_configured():
        """Check if email service is configured."""
        # Receiver is optional (defaults to sender), so we only check sender/pass/server
        return all([Config.EMAIL_SMTP_SERVER, Config.EMAIL_SENDER, Config.EMAIL_PASSWORD])

    @classmethod
    def update_email_config(cls, server, port, sender, password, receiver):
        """Update email configuration in the .env file."""
        env_file = cls.get_env_path()
        
        # Read existing content
        lines = []
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                lines = f.readlines()
        
        # Filter out existing email config
        new_lines = [line for line in lines if not line.strip().startswith(("EMAIL_", "# Email Configuration"))]
        
        # Add new config
        if new_lines and not new_lines[-1].endswith('\n'):
            new_lines.append('\n')
            
        new_lines.append("\n# Email Configuration\n")
        new_lines.append(f"EMAIL_SMTP_SERVER={server}\n")
        new_lines.append(f"EMAIL_SMTP_PORT={port}\n")
        new_lines.append(f"EMAIL_SENDER={sender}\n")
        new_lines.append(f"EMAIL_PASSWORD={password}\n")
        new_lines.append(f"EMAIL_RECEIVER={receiver}\n")
        
        # Write back
        with open(env_file, "w") as f:
            f.writelines(new_lines)
            
        # Update current env
        os.environ["EMAIL_SMTP_SERVER"] = server
        os.environ["EMAIL_SMTP_PORT"] = str(port)
        os.environ["EMAIL_SENDER"] = sender
        os.environ["EMAIL_PASSWORD"] = password
        os.environ["EMAIL_RECEIVER"] = receiver
        
        cls.load()

    @classmethod
    def setup_email(cls):
        """Interactive setup for email configuration."""
        print("\nEmail Configuration (Required for notifications):")
        print("Common SMTP Servers: smtp.gmail.com, smtp.qq.com, smtp.163.com")
        
        email_server = input("SMTP Server: ").strip()
        email_port = input("SMTP Port [465]: ").strip() or "465"
        email_sender = input("Sender Email Address: ").strip()
        email_password = input("Sender Email Password (or App Password): ").strip()
        email_receiver = input(f"Receiver Email Address [{email_sender}]: ").strip() or email_sender

        cls.update_email_config(email_server, email_port, email_sender, email_password, email_receiver)
        
        print(f"Email configuration saved to {cls.get_env_path()}")
        return True

    @classmethod
    def update_core_config(cls, tushare_token, provider, deepseek_key, deepseek_base, deepseek_model, openai_key, openai_base, openai_model):
        """Update core configuration (Tushare & LLM) in the .env file."""
        env_file = cls.get_env_path()
        
        # Read existing content
        lines = []
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                lines = f.readlines()
        
        # Filter out existing core config (everything except EMAIL_)
        # Actually it's safer to filter OUT the ones we are replacing.
        keys_to_remove = ["TUSHARE_TOKEN", "LLM_PROVIDER", 
                          "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
                          "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]
        
        new_lines = []
        for line in lines:
            is_removed = False
            for key in keys_to_remove:
                if line.strip().startswith(key + "="):
                    is_removed = True
                    break
            if not is_removed:
                new_lines.append(line)
        
        # Add new core config at the top (or append if file was empty/only comments)
        # We'll just append them, order in .env doesn't strictly matter for loading, but for readability maybe top is better.
        # But appending is safer to not mess up existing structure too much.
        
        core_config = []
        core_config.append(f"TUSHARE_TOKEN={tushare_token}\n")
        core_config.append(f"LLM_PROVIDER={provider}\n")
        
        if provider == "deepseek":
            core_config.append(f"DEEPSEEK_API_KEY={deepseek_key}\n")
            core_config.append(f"DEEPSEEK_BASE_URL={deepseek_base}\n")
            core_config.append(f"DEEPSEEK_MODEL={deepseek_model}\n")
        else:
            core_config.append(f"OPENAI_API_KEY={openai_key}\n")
            core_config.append(f"OPENAI_BASE_URL={openai_base}\n")
            core_config.append(f"OPENAI_MODEL={openai_model}\n")
            
        # Insert at beginning? Or just append. 
        # If we append, we might have email config before core config. That's fine.
        new_lines.extend(core_config)

        # Write back
        with open(env_file, "w") as f:
            f.writelines(new_lines)

        # Update current env vars
        os.environ["TUSHARE_TOKEN"] = tushare_token
        os.environ["LLM_PROVIDER"] = provider
        
        if provider == "deepseek":
            os.environ["DEEPSEEK_API_KEY"] = deepseek_key
            os.environ["DEEPSEEK_BASE_URL"] = deepseek_base
            os.environ["DEEPSEEK_MODEL"] = deepseek_model
        else:
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["OPENAI_BASE_URL"] = openai_base
            os.environ["OPENAI_MODEL"] = openai_model
            
        cls.load()

    @classmethod
    def setup(cls):
        """Interactive setup for environment variables"""
        print("Configuration missing (or reset requested). Starting setup wizard...")
        
        tushare_token = input("Enter your Tushare Token: ").strip()
        
        print("\nSelect LLM Provider:")
        print("1. DeepSeek (Default)")
        print("2. Moonshot (Kimi)")
        print("3. ZhipuAI (GLM-4)")
        print("4. Yi (01.AI)")
        print("5. Qwen (Aliyun DashScope)")
        print("6. SiliconFlow (Aggregator)")
        print("7. Custom (Manual Input)")
        print("8. Local / Self-Hosted (Ollama, LM Studio, etc.)")
        
        choice = input("Enter choice (1-8): ").strip()
        
        provider = "deepseek"
        deepseek_key = ""
        deepseek_base = "https://api.deepseek.com"
        deepseek_model = "deepseek-chat"
        
        openai_key = ""
        openai_base = ""
        openai_model = ""
        
        if choice == "1" or not choice:
            # DeepSeek logic
            provider = "deepseek"
            deepseek_key = input("Enter DeepSeek API Key: ").strip()
            
        else:
            # All others use the generic OpenAI provider logic
            provider = "openai"
            
            if choice == "2": # Moonshot
                openai_base = "https://api.moonshot.cn/v1"
                default_model = "moonshot-v1-8k"
                print(f"Using Base URL: {openai_base}")
                openai_key = input("Enter Moonshot API Key: ").strip()
                openai_model = input(f"Enter Model Name [default: {default_model}]: ").strip() or default_model
                
            elif choice == "3": # ZhipuAI
                openai_base = "https://open.bigmodel.cn/api/paas/v4"
                default_model = "glm-4"
                print(f"Using Base URL: {openai_base}")
                openai_key = input("Enter ZhipuAI API Key: ").strip()
                openai_model = input(f"Enter Model Name [default: {default_model}]: ").strip() or default_model
                
            elif choice == "4": # Yi
                openai_base = "https://api.lingyiwanwu.com/v1"
                default_model = "yi-34b-chat-0205"
                print(f"Using Base URL: {openai_base}")
                openai_key = input("Enter Yi API Key: ").strip()
                openai_model = input(f"Enter Model Name [default: {default_model}]: ").strip() or default_model
                
            elif choice == "5": # Qwen
                openai_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
                default_model = "qwen-turbo"
                print(f"Using Base URL: {openai_base}")
                openai_key = input("Enter DashScope API Key: ").strip()
                openai_model = input(f"Enter Model Name [default: {default_model}]: ").strip() or default_model
                
            elif choice == "6": # SiliconFlow
                openai_base = "https://api.siliconflow.cn/v1"
                default_model = "deepseek-ai/DeepSeek-V3"
                print(f"Using Base URL: {openai_base}")
                openai_key = input("Enter SiliconFlow API Key: ").strip()
                openai_model = input(f"Enter Model Name [default: {default_model}]: ").strip() or default_model

            elif choice == "8": # Local
                provider = "local"
                default_base = "http://localhost:11434/v1"
                openai_base = input(f"Enter Base URL [default: {default_base}]: ").strip() or default_base
                default_model = "llama3"
                openai_model = input(f"Enter Model Name [default: {default_model}]: ").strip() or default_model
                openai_key = "ollama" 
                
            else: # Custom
                openai_key = input("Enter API Key: ").strip()
                openai_base = input("Enter Base URL: ").strip()
                openai_model = input("Enter Model Name: ").strip()
        
        # Use update_core_config to save
        cls.update_core_config(tushare_token, provider, deepseek_key, deepseek_base, deepseek_model, openai_key, openai_base, openai_model)
            
        print(f"Configuration saved to {cls.get_env_path()}")
        
        # Check if local .env exists and warn user
        local_env = os.path.join(os.getcwd(), ".env")
        if os.path.exists(local_env):
            print(f"WARNING: A local .env file exists at {local_env}")
            print("This local file might override or conflict with the global configuration.")

    @classmethod
    def clear(cls):
        """Clear the configuration file."""
        env_path = cls.get_env_path()
        if os.path.exists(env_path):
            os.remove(env_path)
            print(f"Configuration file removed: {env_path}")
        else:
            print(f"No configuration file found at: {env_path}")
            
        # Also clear local .env if it exists
        local_env = os.path.join(os.getcwd(), ".env")
        if os.path.exists(local_env):
            os.remove(local_env)
            print(f"Local configuration file removed: {local_env}")
            
        # Clear environment variables
        env_vars = [
            "TUSHARE_TOKEN", "LLM_PROVIDER", 
            "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
            "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
        
        # Reset class variables
        cls.TUSHARE_TOKEN = None
        cls.LLM_PROVIDER = None
        cls.DEEPSEEK_API_KEY = None
        cls.DEEPSEEK_BASE_URL = None
        cls.DEEPSEEK_MODEL = None
        cls.OPENAI_API_KEY = None
        cls.OPENAI_BASE_URL = None
        cls.OPENAI_MODEL = None

# Load on module import
Config.load()
