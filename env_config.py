"""
env_config.py - Configuración automática de entorno (Colab vs Local)

Este módulo detecta automáticamente si el código se ejecuta en Google Colab
o localmente y configura matplotlib apropiadamente.
"""

def detect_environment():
    """Detecta automáticamente si estamos en Colab o Local."""
    try:
        import google.colab
        return 'colab'
    except ImportError:
        return 'local'

def setup_matplotlib():
    """Configura matplotlib según el entorno detectado."""
    import matplotlib
    
    env = detect_environment()
    
    if env == 'colab':
        matplotlib.use('Agg')  # Backend sin GUI para Colab
        try:
            from IPython.display import clear_output, display
            globals()['clear_output'] = clear_output
            globals()['display'] = display
        except ImportError:
            pass
        print("✓ Configurado para Google Colab (Backend: Agg)")
    else:
        try:
            matplotlib.use('TkAgg')  # Backend interactivo para local
            print("✓ Configurado para Local (Backend: TkAgg)")
        except Exception as e:
            matplotlib.use('Agg')
            print(f"⚠ TkAgg no disponible, usando Agg: {e}")
    
    print(f"Entorno: {env} | Backend: {matplotlib.get_backend()}")
    return env

# Variables globales
ENV = None
IN_COLAB = False

def init():
    """Inicializa la configuración de entorno."""
    global ENV, IN_COLAB
    ENV = setup_matplotlib()
    IN_COLAB = (ENV == 'colab')
    return ENV

# Auto-inicializar al importar
if ENV is None:
    init()
