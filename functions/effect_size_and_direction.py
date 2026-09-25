import numpy as np
import pandas as pd

"""
rank biserial correlation effect size and direction calculation
Evidence MAR:
p_value < 0.05 (hay una evidencia significativa)
0.1 < effect_size < 0.99 (Existe un correlacion de leve a muy fuerte)
"""
class EffectSizeAndDirection:
    @staticmethod
    def rank_biserial(stat, p_value, missing=None, present=None):
        if missing is None or present is None:
            return None

        lenMissing, lenPresent = len(missing), len(present)
        # Rango biseral
        rankBiseral = (2 * stat) / (lenMissing * lenPresent) - 1
        # Diferencia de medianas
        diff_means = np.median(missing) - np.median(present)
        direction = "Neutral"
        #Direction
        if rankBiseral > 0:
            #positivo (+) indica que los valores numéricos tienden a ser más altos cuando el dato falta.
            direction = "positive"
        elif rankBiseral < 0:
            #Un resultado negativo (-) indica que tienden a ser más bajos.
            direction = "negative"  # Nulos tienden a valores menores
        #Interpretacion:
        interpretation = ""
        if abs(rankBiseral) < 0.1:
            interpretation = "Insignificante"
        elif abs(rankBiseral) < 0.3:
            interpretation = "Pequeño"
        elif abs(rankBiseral) < 0.5:
            interpretation = "Mediano"
        else:
            interpretation = "Grande"
        return {
            "rankBiseral": abs(rankBiseral),
            "diff_means": diff_means,
            "direction": direction,
            "interpretation": interpretation,
        }
