# Evaluación de normalidad de features numéricas para EDA

Esta guía evalúa si la distribución de una feature es compatible con una normal y qué tan relevantes son sus desviaciones. **Ningún test demuestra normalidad.** Las recomendaciones de prioridad son para EDA de dataframes, no una clasificación universal de potencia.

## 1. Reglas de interpretación y preparación

- **Hipótesis nula (H0):** las observaciones provienen de una distribución normal. En el KS clásico se contrasta una normal completamente especificada; en los tests habituales de normalidad se permiten media y varianza desconocidas.
- Elegir `alpha` antes de interpretar los resultados, por ejemplo `0.05`. Si `p < alpha`, rechazar H0; en caso contrario, **no rechazar H0**, no afirmar que la feature es normal.
- `n` es el número de observaciones válidas efectivamente analizadas, después de tratar los faltantes. Registrar los faltantes excluidos, detectar infinitos y no convertirlos silenciosamente en datos válidos.
- Una feature constante, con varianza cero o con observaciones insuficientes debe marcarse como **no evaluable**. No asignarle un p-valor ficticio.
- Los p-valores usuales suponen observaciones independientes e idénticamente distribuidas. Series temporales, mediciones repetidas y agrupamientos requieren un procedimiento que considere esa dependencia.
- Con `n` pequeño hay poca potencia: no rechazar puede deberse a falta de información. Con `n` grande, desviaciones irrelevantes pueden producir p-valores muy pequeños.
- No eliminar outliers, redondear, añadir ruido ni elegir una submuestra únicamente para obtener `p >= alpha`. Investigar errores de medición y documentar cualquier cambio.
- Examinar histograma o ECDF y gráfico Q-Q junto al contraste. Si se contrastan muchas features, reportar p-valores crudos y, cuando se interpreten rechazos simultáneos, considerar Benjamini-Hochberg (`statsmodels.stats.multitest.multipletests`, `method="fdr_bh"`); su garantía depende de los supuestos de dependencia.
- La mayoría de los modelos **no exige features marginalmente normales**. En regresión lineal, la normalidad relevante para cierta inferencia es la de los errores condicionales, no la de cada predictor. Tampoco el teorema central del límite convierte en normales los datos originales.

## 2. Resumen de selección

Los mínimos de una API, la precisión de su p-valor y una recomendación práctica son conceptos distintos. No existe una frontera universal de tamaño muestral que determine qué test es el mejor.

| Método | Tamaño muestral y precisión | Prioridad para EDA | Implementación recomendada |
|---|---|---|---|
| Shapiro-Wilk | Mínimo `n = 3`; p-valor habitual hasta `n = 5000`, inclusive. Para `n > 5000`, SciPy advierte que el p-valor puede ser inexacto, aunque W siga siendo preciso. | Opción general para muestras pequeñas o medianas; no interpretar ausencia de rechazo con muy pocos datos como evidencia fuerte. | `scipy.stats.shapiro` |
| Kolmogorov-Smirnov (KS) clásico | Sin límite general de 5000. La validez exige una CDF continua completamente especificada, sin parámetros ajustados con la misma muestra. | No priorizar para normalidad con parámetros desconocidos; usar Lilliefors o un contraste calibrado por simulación. | `scipy.stats.ks_1samp` o `scipy.stats.kstest` |
| Anderson-Darling (AD) | **No requiere `n >= 5000`**. Estas APIs no documentan un mínimo universal que garantice la validez de la aproximación para normalidad; con muestras diminutas, no basar decisiones en tablas o aproximaciones sin verificar su calibración. | Muy útil cuando importan las colas; también para muestras pequeñas o medianas con cautela. | `statsmodels.stats.diagnostic.normal_ad`; alternativa actual de SciPy indicada abajo |
| D'Agostino-Pearson | Mínimo de la API `n = 8`; preferir `n > 20` para el p-valor asintótico, por la advertencia de validez de su componente de curtosis. | Diagnóstico conjunto de asimetría y curtosis; complementar con Q-Q. | `scipy.stats.normaltest` |
| Jarque-Bera (JB) | SciPy recomienda muestras grandes, **`n > 2000`**, por su aproximación asintótica chi-cuadrado. No recomendar desde `n = 20`. | Secundario en EDA general; útil por convención en econometría. | `scipy.stats.jarque_bera` |
| Lilliefors para normalidad | Mínimo `n = 4` en statsmodels. Sin límite general de 5000; el p-valor tabulado tiene límites de resolución. | Alternativa correcta al KS clásico cuando se estiman media y desviación con los datos. | `statsmodels.stats.diagnostic.lilliefors`, `dist="norm"`, `pvalmethod="table"` |
| Cramér-von Mises (CvM) | La función directa admite `n >= 2`, pero eso no garantiza utilidad con tan pocos datos. Si se ajustan parámetros, calibrar por bootstrap paramétrico. | Alternativa de discrepancia global; opcional frente a Shapiro-Wilk o AD. | `scipy.stats.goodness_of_fit`, `statistic="cvm"` |
| Gráfico Q-Q normal | Sin mínimo inferencial; con muy pocos puntos resulta poco informativo y las colas son muy inciertas. | Complemento recomendado siempre; no es un test con p-valor. | `scipy.stats.probplot` y Matplotlib, o `statsmodels.api.qqplot` |

## 3. Discretas frente a continuas

**Una variable verdaderamente discreta no puede seguir exactamente una distribución normal continua.** Esto aplica a todos los contrastes de normalidad continua de esta guía.

| Tipo de feature | Recomendación |
|---|---|
| Continua, sin redondeo importante | Aplicar los tests bajo sus supuestos y límites muestrales. |
| Continua registrada con redondeo fino y muchos valores distintos | Puede estudiarse como aproximación continua; inspeccionar empates y resolución. Los p-valores dejan de tener necesariamente su calibración nominal. |
| Discreta con muchos niveles, por ejemplo un conteo grande | Usar Q-Q y discrepancias descriptivas para valorar una aproximación normal. Para inferencia formal, contrastar un modelo discreto apropiado o simular el mecanismo real de observación. |
| Binaria, ordinal codificada con números o conteo con pocos niveles | No recomendar estos tests de normalidad continua. Usar frecuencias y modelos acordes con su naturaleza. |

No hay un número mágico de valores únicos que vuelva continua una variable. Si se necesita un contraste formal para mediciones redondeadas, simular también el redondeo y reestimar los parámetros en cada réplica: **Manejo Manual** del mecanismo de observación. Los p-valores estándar no se corrigen añadiendo ruido aleatorio.

## 4. Tamaño del efecto: qué reportar realmente

No existe un tamaño del efecto estandarizado y universalmente aceptado asociado a cada test de normalidad. **El p-valor no mide magnitud**, y estadísticas como `K²`, `JB` y `A²` dependen de `n`. No trasladar automáticamente Cohen's d, eta-cuadrado o reglas de efecto pequeño/mediano/grande de otros contrastes.

Las recomendaciones por método de la siguiente sección son medidas descriptivas operativas, no un consenso de umbrales universales:

- **D de KS/Lilliefors:** máxima diferencia absoluta entre la ECDF y la CDF de referencia. Es adimensional, está entre 0 y 1 y tiene interpretación directa: `D = 0.08` significa una discrepancia máxima de 8 puntos porcentuales de probabilidad acumulada. Su estimación también fluctúa con `n`.
- **Asimetría y exceso de curtosis:** reportar `scipy.stats.skew(x, bias=False)` y `scipy.stats.kurtosis(x, fisher=True, bias=False)`. En una normal poblacional ambos son cero. Son descriptores, no demostraciones de normalidad; otras distribuciones pueden compartir esos momentos.
- **Wasserstein-1 estandarizada (`W1_std`):** complemento global si se necesita una discrepancia comparable entre features con distintas unidades. Definir `W1_std = integral_0^1 |Q_emp(u) - (media + s * Phi^-1(u))| du / s`, con `s` muestral (`ddof=1`) positivo. Mide el desplazamiento absoluto medio hacia la normal ajustada en unidades de desviación estándar. No es un tamaño del efecto canónico de ninguno de estos tests ni tiene cortes universales.
- Para `W1_std`, integrar la diferencia de cuantiles con control numérico: **Manejo Manual** de la comparación empírica frente a normal teórica. `scipy.stats.wasserstein_distance` sirve directamente para dos distribuciones discretas/empíricas; compararla con una muestra normal aleatoria añade error Monte Carlo y no calcula exactamente esta integral.
- **Desviaciones de cuantiles:** para localizar diferencias en las colas, reportar `(Q_emp(p) - (media + s * Phi^-1(p))) / s`, por ejemplo para `p = 0.05` y `0.95`. Usar `numpy.quantile` y `scipy.stats.norm.ppf`: **Manejo Manual** de la combinación. Los cuantiles extremos son poco fiables con muestras pequeñas; documentar el método de cuantiles.

Incluso estas medidas presentan discrepancias distintas de cero por variabilidad muestral bajo normalidad. Evaluar su relevancia según el dominio y, si se requiere incertidumbre, usar intervalos o simulación adecuados al diseño muestral.

## 5. Fichas de los métodos

### 5.1. Shapiro-Wilk

1. **Qué detecta:** desviaciones globales de normalidad mediante la relación entre observaciones ordenadas y lo esperado bajo una normal; sensible a asimetrías, colas y observaciones extremas.
2. **Resultado:** estadístico `W` y p-valor. Valores menores de W indican mayor desviación; rechazar si `p < alpha`.
3. **Muestras:** `3 <= n <= 5000` para el uso habitual del p-valor de SciPy. Superar 5000 no invalida W, pero sí exige cautela con su p-valor; considerar otro contraste o calibración específica por simulación.
4. **Librería:** `scipy.stats.shapiro(x)`. No implementarlo manualmente.
5. **Magnitud recomendada:** reportar W; `1 - W` puede acompañarlo como índice descriptivo de alejamiento, **no como tamaño del efecto estandarizado**. Para una magnitud de interpretación práctica entre features, complementar con `W1_std` y Q-Q.
6. **Tipo de feature:** recomendado para continuas. No para normalidad exacta de discretas; para redondeadas, aplicar las cautelas de la sección 3.
7. **Prioridad:** primera opción general para muestras pequeñas o medianas, sin afirmar que siempre sea el test más potente.

### 5.2. Kolmogorov-Smirnov clásico de una muestra

1. **Qué detecta:** máxima diferencia entre la ECDF y una CDF normal **completamente especificada**.
2. **Resultado:** estadístico `D` y p-valor; rechazar si `p < alpha`.
3. **Muestras:** no existe el límite `n <= 5000`. El problema principal es la especificación de la referencia, no ese umbral.
4. **Librería:** `scipy.stats.ks_1samp(x, cdf)` o `scipy.stats.kstest(x, "norm", args=(mu, sigma))`, donde `mu` y `sigma > 0` deben estar fijados independientemente de la muestra. Si se estiman con otra muestra, su incertidumbre tampoco desaparece automáticamente.
5. **Magnitud recomendada:** D, por su interpretación en probabilidad acumulada. No usar `D * sqrt(n)` como medida de efecto independiente del tamaño muestral.
6. **Tipo de feature:** continua; sus p-valores habituales no son válidos para discretas ni para empates importantes sin calibración específica.
7. **Homólogo preferible en el EDA habitual:** Lilliefors si se estiman media y desviación con la misma feature. Estandarizar la muestra y luego aplicar KS contra `N(0, 1)` **no resuelve** la estimación de parámetros. Si interesan especialmente las colas, considerar AD.

### 5.3. Anderson-Darling

1. **Qué detecta:** discrepancia entre ECDF y CDF normal, con mayor ponderación de las colas que KS.
2. **Resultado:** estadístico `A²` y, según implementación, p-valor o valores críticos. Con p-valor, rechazar si `p < alpha`; con tablas, si `A²` supera el valor crítico correspondiente. No mezclar estadísticos corregidos y tablas de otra implementación.
3. **Muestras:** no está reservado para `n >= 5000`. No imponer un mínimo arbitrario como garantía estadística: que el estadístico pueda calcularse no asegura buena calibración con muy pocos datos. En muestras pequeñas, considerar Shapiro-Wilk o calibración por simulación bajo el modelo apropiado.
4. **Librería:** `statsmodels.stats.diagnostic.normal_ad(x)` devuelve `(A², p)` para normal con media y varianza desconocidas. En SciPy 1.18, `scipy.stats.anderson(x, dist="norm", method="interpolate")` devuelve estadístico y p-valor interpolado de tablas. Usar `"interpolate"`, como exige la implementación; la docstring instalada contiene la variante errónea `"interpolated"`. El modo sin `method`, basado en valores críticos, está deprecado desde SciPy 1.17; no asumir que todas las versiones retornan lo mismo.
5. **Calibración alternativa:** `scipy.stats.goodness_of_fit(scipy.stats.norm, x, statistic="ad", ...)` permite bootstrap paramétrico con reajuste en cada réplica. Es más costoso; el p-valor interpolado tampoco tiene resolución arbitraria en las colas.
6. **Magnitud recomendada:** `W1_std` como complemento global y desviaciones estandarizadas de cuantiles para las colas. Reportar A² como estadístico de contraste, no como efecto comparable directamente entre tamaños muestrales distintos.
7. **Tipo de feature:** continua; no recomendar p-valores estándar para discretas o empates importantes.
8. **Prioridad:** alternativa principal a Shapiro-Wilk cuando interesa el comportamiento de las colas; no existe dominancia universal entre ambos.

### 5.4. D'Agostino-Pearson

1. **Qué detecta:** desviaciones de asimetría y curtosis mediante su combinación en el estadístico **`K²`**, no D.
2. **Resultado:** `K² = Z_asimetría² + Z_curtosis²` y p-valor con aproximación chi-cuadrado de 2 grados de libertad; rechazar si `p < alpha`.
3. **Muestras:** SciPy exige `n >= 8`. Para uso rutinario del p-valor asintótico, preferir `n > 20`, siguiendo la advertencia de `kurtosistest`. Tener 20 o 21 datos no garantiza por sí solo buena calibración o potencia.
4. **Librería:** `scipy.stats.normaltest(x)`. Para muestras pequeñas, preferir Shapiro-Wilk o una calibración apropiada por simulación, no ignorar las advertencias.
5. **Magnitud recomendada:** asimetría y exceso de curtosis por separado, con las funciones de la sección 4. Conservan signo e interpretación de forma; no usar K² como tamaño del efecto.
6. **Tipo de feature:** continua. En discretas, estos momentos son descriptivos útiles, pero el contraste de normalidad exacta no es recomendable.
7. **Limitación:** no caracteriza toda la distribución; desviaciones con momentos similares a los normales pueden pasar inadvertidas. Complementar con Q-Q o un test basado en la distribución completa.

### 5.5. Jarque-Bera

1. **Qué detecta:** discrepancias de asimetría y curtosis frente a la normal.
2. **Resultado:** `JB = n * (g1² / 6 + g2² / 24)`, donde `g1` y `g2` son asimetría y exceso de curtosis basados en momentos utilizados por el test; p-valor asintótico chi-cuadrado de 2 grados de libertad.
3. **Muestras:** SciPy recomienda `n > 2000` para su aproximación asintótica. No es un mínimo matemático para calcular JB ni una garantía universal de precisión. Para menos datos, no confiar automáticamente en ese p-valor.
4. **Librería:** `scipy.stats.jarque_bera(x)`. No sustituir manualmente sus momentos por versiones corregidas de sesgo sin recalibrar el contraste.
5. **Magnitud recomendada:** asimetría y exceso de curtosis por separado. Las versiones descriptivas corregidas de sesgo de la sección 4 son útiles para reportar forma, pero no necesariamente reconstruyen el JB de la librería. No usar JB como efecto: crece con `n`.
6. **Tipo de feature:** continua; en discretas, limitarse a interpretar los momentos descriptivamente o contrastar otro modelo.
7. **Homólogo preferible para esta guía:** D'Agostino-Pearson suele ser una opción más práctica para un diagnóstico por momentos con muestras moderadas, por sus transformaciones de asimetría y curtosis. Shapiro-Wilk o AD son alternativas generales. No implica superioridad de potencia para toda distribución.

### 5.6. Lilliefors para normalidad — método adicional recomendado

1. **Qué detecta:** usa la discrepancia D de KS, pero calibra el contraste para media y varianza normales estimadas con la misma muestra.
2. **Resultado:** `(D, p)`; rechazar si `p < alpha`.
3. **Muestras:** statsmodels exige `n >= 4` para `dist="norm"`. Con `pvalmethod="table"`, los p-valores fuera del rango se devuelven acotados a `0.001` o `0.990`; no interpretarlos como valores exactos más allá de esos límites.
4. **Librería:** `statsmodels.stats.diagnostic.lilliefors(x, dist="norm", pvalmethod="table")`. Preferir las tablas frente a `pvalmethod="approx"`, cuya fórmula tiene un rango de validez más restringido.
5. **Magnitud recomendada:** D, la máxima discrepancia acumulada respecto a la normal ajustada. La corrección de Lilliefors cambia la calibración del p-valor, no la interpretación descriptiva de D.
6. **Tipo de feature:** continua; no corrige la discreción, el redondeo importante ni la dependencia entre observaciones.
7. **Prioridad:** reemplazo apropiado del KS clásico para normalidad con parámetros estimados; no necesariamente mejor que Shapiro-Wilk o AD en potencia.

### 5.7. Cramér-von Mises — método adicional opcional

1. **Qué detecta:** discrepancia cuadrática integrada entre ECDF y CDF normal; AD da mayor peso relativo a las colas.
2. **Resultado:** estadístico `T = n * integral (F_n - F)² dF` y p-valor; rechazar si `p < alpha`.
3. **Muestras:** `scipy.stats.cramervonmises` exige al menos 2 observaciones, pero un resultado calculable con tan pocos datos aporta muy poca evidencia. No hay un límite general de 5000 ni un tamaño que garantice potencia.
4. **Librería:** en el EDA con parámetros desconocidos, preferir `scipy.stats.goodness_of_fit(scipy.stats.norm, x, statistic="cvm", ...)`, que ajusta los parámetros y los reestima en cada réplica. No pasar como `known_params` parámetros estimados con la misma muestra. Registrar semilla (`rng`), número de réplicas (`n_mc_samples`) y error Monte Carlo; la resolución mínima del p-valor es `1 / (n_mc_samples + 1)`.
5. **Advertencia:** el p-valor directo de `scipy.stats.cramervonmises(x, "norm", args=(mu, sigma))` supone parámetros conocidos. No es fiable si se sustituyen por la media y desviación de la muestra.
6. **Magnitud recomendada:** `sqrt(T / n)`, como discrepancia cuadrática media integrada en la escala de probabilidad, sin categorías universales de efecto; `W1_std` es un complemento si se desea interpretación en unidades de desviación estándar. **Manejo Manual** de esas transformaciones, no del contraste.
7. **Tipo de feature:** continua. El bootstrap normal estándar no calibra por sí solo una variable discreta ni un proceso de redondeo.
8. **Prioridad:** opcional; no hace falta ejecutarlo junto con todos los demás. Elegir según la desviación de interés y el coste de simulación.

### 5.8. Gráfico Q-Q normal — complemento adicional recomendado

1. **Qué muestra:** compara cuantiles observados y normales; curvaturas, asimetría, colas y valores extremos.
2. **Resultado:** gráfico descriptivo, no decisión automática ni p-valor.
3. **Muestras:** útil a distintos tamaños; con muy pocos puntos no permite descartar desviaciones y con muchas observaciones conviene inspeccionar especialmente las colas.
4. **Librería:** `scipy.stats.probplot(x, dist="norm", plot=ax)` o `statsmodels.api.qqplot(x, dist=scipy.stats.norm, fit=True, line="45", ax=ax)`. Las bandas de referencia deben corresponder al ajuste y al diseño; no confundir bandas puntuales con simultáneas.
5. **Magnitud recomendada:** desviaciones estandarizadas de cuantiles y, si hace falta un resumen global, `W1_std`. No usar solo el R² de la recta Q-Q ni un corte arbitrario de correlación como prueba de normalidad.
6. **Tipo de feature:** recomendable para continuas y para valorar visualmente una aproximación en discretas; los escalones por empates son esperables y no convierten la variable en continua.

## 6. Métodos que no conviene priorizar si ya se usa su alternativa

Esta sección identifica métodos redundantes o menos adecuados **para este objetivo**, no métodos inútiles en cualquier contexto.

| Método | Alternativa preferible para esta guía | Motivo y excepción |
|---|---|---|
| KS clásico con media y desviación ajustadas a los mismos datos | Lilliefors o bootstrap paramétrico con reajuste | El p-valor clásico no está calibrado para ese ajuste. Con parámetros realmente conocidos, KS sí es válido. |
| Jarque-Bera como test rutinario desde muestras moderadas | D'Agostino-Pearson para momentos; Shapiro-Wilk o AD para evaluación general | JB depende de una aproximación de muestras grandes. Conservarlo si se requiere por convención econométrica. |
| Test de asimetría aislado (`scipy.stats.skewtest`) o de curtosis aislado (`scipy.stats.kurtosistest`) como prueba de normalidad | D'Agostino-Pearson si interesa combinar ambos componentes | Evaluar un solo momento no evalúa normalidad global. Siguen siendo útiles si ese momento es la pregunta específica. |
| Chi-cuadrado de Pearson sobre intervalos para una feature continua | Shapiro-Wilk, AD o CvM | Agrupar pierde información y depende de los intervalos; además exige frecuencias esperadas adecuadas y ajustar grados de libertad por parámetros estimados. Para datos discretos o ya agrupados puede ser apropiado, con el modelo correcto. |
| Shapiro-Francia como segundo contraste rutinario junto a Shapiro-Wilk | Mantener Shapiro-Wilk como opción general | Es una alternativa emparentada, no una mejora universal; añade poca variedad diagnóstica a esta guía y no tiene una función dedicada en la API pública de SciPy. Puede tener usos específicos, no se afirma que Shapiro-Wilk domine siempre en potencia. |

No ejecutar varios tests y elegir el p-valor que permita la conclusión deseada. El acuerdo de varios tests tampoco demuestra normalidad.

## 7. Flujo recomendado por columna de un dataframe

1. **Clasificar la variable por su significado**, no solo por su dtype: continua, redondeada, conteo, binaria u ordinal.
2. **Validar datos:** registrar `n_total`, faltantes, infinitos, `n_validas`, valores únicos, resolución y varianza; indicar explícitamente si no es evaluable y por qué.
3. **Visualizar:** histograma/ECDF y Q-Q. Investigar si hay mezclas de grupos, truncamiento, acumulación en cero u otros mecanismos que expliquen la forma.
4. **Elegir un contraste principal cuando tenga sentido:** Shapiro-Wilk para `3 <= n <= 5000`; AD si preocupan las colas o como alternativa fuera de ese rango. D'Agostino-Pearson si el foco son los momentos y `n > 20`. No usar estos rangos como selección automática ciega.
5. **Reportar magnitud:** el descriptor recomendado en la ficha; añadir cuantiles o una discrepancia global cuando la decisión práctica lo necesite. No decidir transformar una feature solo por `p < alpha`.
6. **Registrar resultados:** método, estadístico, p-valor (o valor crítico y nivel), `alpha`, decisión «rechazar/no rechazar/no evaluable», descriptores de magnitud y advertencias. Añadir p-valor ajustado si se usa corrección múltiple.
7. **Documentar simulación o muestreo:** semilla, réplicas, ajuste de parámetros y cualquier selección de observaciones. Si se usa una submuestra por coste, el resultado se refiere a ella; no buscar una que deje de rechazar.

## 8. Fuentes y compatibilidad

Las implementaciones citadas están disponibles en las dependencias del proyecto: SciPy, statsmodels, NumPy y Matplotlib. No se necesitan paquetes adicionales. Las recomendaciones prácticas de esta guía no deben confundirse con garantías de las librerías.

- [SciPy: Shapiro-Wilk, mínimo de 3 y advertencia para n > 5000](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html).
- [SciPy: KS de una muestra contra una CDF continua](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_1samp.html).
- [SciPy: Anderson-Darling y cambios de API desde 1.17](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html). La opción `method="interpolate"` se contrastó con los ejemplos y la validación del argumento en el código de SciPy 1.18 instalado en el proyecto, por la errata de su docstring.
- [statsmodels: AD con media y varianza desconocidas](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.normal_ad.html).
- [SciPy: D'Agostino-Pearson, mínimo de 8](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html) y [advertencia de validez del componente de curtosis para n > 20](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosistest.html).
- [SciPy: Jarque-Bera y recomendación de n > 2000](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html).
- [statsmodels: Lilliefors y p-valores tabulados](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.lilliefors.html) y [código fuente con mínimo de 4 para normalidad](https://www.statsmodels.org/stable/_modules/statsmodels/stats/_lilliefors.html).
- [SciPy: CvM y advertencia sobre parámetros estimados](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cramervonmises.html).
- [SciPy: bondad de ajuste por bootstrap paramétrico y reajuste de parámetros](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html).
- [SciPy: gráfico de probabilidad normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html) y [Wasserstein-1 entre distribuciones discretas](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html).