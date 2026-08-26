from functools import wraps
import time

def calcular_tempo(func):
    """Decorator para medir o tempo de execução de uma função."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.perf_counter()
        resultado = func(*args, **kwargs)
        fim = time.perf_counter()
        tempo_total = (fim - inicio)
        if tempo_total < 60:
            print(f"[Tempo de Execução] A função '{func.__name__}' levou {tempo_total:.4f} segundos.")
        else:
            tempo_total /= 60
            print(f"[Tempo de Execução] A função '{func.__name__}' levou {tempo_total:.4f} minutos.")
        return resultado
    return wrapper