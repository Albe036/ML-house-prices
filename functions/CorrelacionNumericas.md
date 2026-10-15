# Asociación entre la ausencia de una feature y una feature numérica

## 1. Qué se está midiendo

Sea `X` la feature con valores faltantes y `Y` la feature numérica que se quiere explorar. Se define:

- `M = 1` si `X` está ausente.
- `M = 0` si `X` está presente.
- `Y1`: valores observados de `Y` en registros con `M = 1`.
- `Y0`: valores observados de `Y` en registros con `M = 0`.
- `n1` y `n0`: tamaños de estos dos grupos, después de excluir los faltantes de `Y`.

**El objetivo de esta guía es estudiar la asociación entre `M` e `Y`, no la correlación entre los valores de `X` e `Y`.** `X` puede ser numérica o categórica: su tipo no afecta a esta comparación porque se utiliza su indicador de ausencia.

Por ejemplo, si `X` es el área del garaje y `Y` el precio, se pregunta si el precio cambia entre viviendas con área del garaje ausente y viviendas con área registrada. No se está midiendo si un garaje más grande se asocia con un precio mayor.

La mayoría de los métodos siguientes son **contrastes de dos grupos**, no coeficientes de correlación. Se deben acompañar de un tamaño del efecto y su intervalo de confianza (IC). El valor p evalúa evidencia contra una hipótesis; no cuantifica la intensidad de la asociación ni la probabilidad de que la hipótesis nula sea cierta.

### Preparación y supuestos compartidos

1. Construir `M` **antes de imputar `X`**. No reemplazar sus faltantes por cero para este análisis.
2. Si `Y` también tiene faltantes, comparar solamente registros con `Y` observada y documentar cuántos se excluyen en cada grupo. La conclusión queda condicionada a que `Y` esté observada; esta selección puede introducir sesgo. No imputar `Y` de forma automática para obtener un contraste.
3. Verificar que ambos grupos existan y que haya variabilidad suficiente. Con un grupo vacío, o una variable constante, algunos efectos y contrastes no están definidos; informar el problema, no devolver una asociación nula.
4. Revisar el significado de los faltantes. En datos de viviendas, un `NA` puede indicar «no tiene garaje» en lugar de «dato desconocido». Una asociación puede reflejar esta ausencia estructural, no un fallo de captura.
5. Las observaciones deben ser independientes. Para mediciones repetidas, hogares relacionados, series temporales o agrupación espacial, usar inferencia que respete esa dependencia: bloques, clústeres o modelos adecuados.
6. Inspeccionar tamaños, porcentaje de ausencia, histogramas, ECDF, cuantiles y valores extremos por grupo. Las recomendaciones sobre muestras pequeñas o grandes dependen sobre todo de `min(n1, n0)`, del desbalance y de la forma de las distribuciones, no solo del total de filas.

No existe un umbral universal de «30 registros» que garantice validez. Miles de filas con solo cinco casos ausentes siguen ofreciendo muy poca información sobre ese grupo. Tampoco conviene escoger el contraste únicamente según un test de normalidad o de igualdad de varianzas.

## 2. Selección rápida

| Pregunta principal | Método recomendado | Efecto que conviene reportar |
|---|---|---|
| ¿Cambia el promedio de `Y`? | t de Welch | Diferencia de medias con IC; opcionalmente diferencia estandarizada |
| ¿Cambia el centro, pero hay colas pesadas o extremos influyentes? | t recortada de Yuen con varianzas desiguales | Diferencia de medias recortadas con IC |
| ¿Los valores de un grupo tienden a ser mayores, sin asumir normalidad? | Mann-Whitney U si sus supuestos son razonables | Probabilidad de superioridad `A` o delta de Cliff |
| ¿Interesa esa superioridad, pero las formas o dispersiones difieren? | Brunner-Munzel | `A` o delta de Cliff con IC compatible |
| ¿Cambia la distribución, no solo su centro? | Kolmogorov-Smirnov de dos muestras | Distancia `D`; cuantiles o distancia de Wasserstein como complemento |
| ¿Cambia principalmente la dispersión? | Brown-Forsythe, variante de Levene centrada en la mediana | IQR o MAD de cada grupo y su diferencia o razón |
| ¿Se necesita calibrar un estadístico específico sin una aproximación paramétrica? | Permutación, si hay intercambiabilidad bajo la hipótesis nula | El efecto correspondiente al estadístico elegido |

Si se necesita **un coeficiente entre -1 y 1**, usar la correlación punto-biserial para resumir diferencias de medias, o delta de Cliff para superioridad por rangos. Ninguno detecta por sí solo todas las formas de dependencia.

## 3. Métodos y condiciones de uso

### 3.1. t de Welch: diferencia de medias

- **Qué contrasta:** `H0: media(Y1) = media(Y0)`. Utiliza la diferencia de medias dividida por su error estándar, con grados de libertad de Welch-Satterthwaite.
- **Tipo:** paramétrico. No exige varianzas iguales. La inferencia t es una aproximación incluso bajo normalidad con varianzas desiguales; con muestras pequeñas conviene que las distribuciones sean aproximadamente normales.
- **Outliers:** poco tolerante, porque utiliza medias y varianzas. Revisar especialmente extremos en el grupo minoritario.
- **Pocos registros:** puede ser útil con normalidad aproximada y sin extremos influyentes, pero el IC puede ser muy amplio. No confiar en la aproximación con un grupo diminuto y muy asimétrico.
- **Muchos registros:** suele funcionar bien por la aproximación normal de las medias si las varianzas son finitas. Una cola muy pesada o un desbalance extremo puede retrasar esa aproximación.
- **Discretas o continuas:** ambas, si la media tiene sentido y hay variabilidad suficiente. Con conteos muy escasos, exceso de ceros o soporte muy limitado, revisar la aproximación; un modelo de conteo puede ser más informativo.
- **Tamaño del efecto:** priorizar `Δ = media(Y1) - media(Y0)` con IC de Welch, expresado en las unidades originales. Como complemento, una diferencia estandarizada con denominador explícito; véase la sección 4.
- **Cuándo evitarlo:** si interesa la mediana, la superioridad ordinal o un cambio de forma sin cambio de media. Una transformación logarítmica cambia la pregunta a la escala transformada y debe documentarse.
- **Referencia de implementación:** `scipy.stats.ttest_ind(Y1, Y0, equal_var=False)`.

**Por qué reemplaza a la t de Student clásica en esta guía:** Welch no requiere homocedasticidad y suele perder poca eficiencia cuando las varianzas son iguales. Se conserva como opción general para dos grupos; no es universalmente más potente, pero evita mantener dos alternativas casi equivalentes y escoger entre ellas mediante un pretest de varianzas.

### 3.2. t recortada de Yuen: diferencia de centros robustos

- **Qué contrasta:** igualdad de medias poblacionales recortadas. Se elimina una proporción predefinida de cada cola para estimar el centro y se usan varianzas winsorizadas para calcular el error estándar.
- **Tipo:** contraste robusto de tipo t, con inferencia aproximada. No es una prueba exacta libre de distribución ni exige normalidad estricta; suele clasificarse como alternativa robusta a los métodos paramétricos clásicos.
- **Outliers:** bastante tolerante a contaminación en las colas. Por ejemplo, un recorte del 20 % elimina ese porcentaje **en cada extremo** y conserva aproximadamente el 60 % central.
- **Pocos registros:** el recorte reduce aún más la información disponible. Con grupos muy pequeños puede ser inestable o no recortar ninguna observación por redondeo.
- **Muchos registros:** apropiada para colas pesadas o extremos válidos que no deberían dominar el centro. Fijar el porcentaje de recorte antes de comparar resultados.
- **Discretas o continuas:** ambas cuando la media recortada sea interpretable; menos atractiva con pocos valores distintos o muchos ceros, donde el recorte puede alterar mucho el significado del centro.
- **Tamaño del efecto:** diferencia de medias recortadas con IC compatible con Yuen; opcionalmente un efecto robusto estandarizado con definición explícita de la escala winsorizada.
- **Cuándo evitarlo:** si los valores de las colas son precisamente el fenómeno de interés. No responde a la misma pregunta que Welch: un efecto en la media aritmética puede desaparecer al recortar.
- **Referencia de implementación:** `scipy.stats.ttest_ind(Y1, Y0, equal_var=False, trim=0.2)`.

No sustituye universalmente a Welch: cambia el parámetro que se estima. Se conserva por su utilidad específica frente a extremos y colas pesadas.

### 3.3. Mann-Whitney U: superioridad basada en rangos

- **Qué hace:** combina los valores de ambos grupos, asigna rangos medios a los empates y calcula `U`. Bajo la hipótesis nula usual, las distribuciones son iguales. Su estadístico resume la tendencia de un grupo a presentar valores mayores.
- **Tipo:** no paramétrico. No requiere normalidad, pero no está libre de supuestos.
- **Outliers:** tolerante a su magnitud, porque utiliza rangos. Un extremo puede afectar el orden, pero no recibe más peso por ser diez o mil veces mayor.
- **Pocos registros:** utilizar inferencia exacta cuando no haya empates y sea computacionalmente viable. El cálculo exacto usual de SciPy no corrige empates; con empates y grupos pequeños, considerar una permutación válida bajo igualdad de distribuciones.
- **Muchos registros:** la aproximación asintótica con corrección de empates suele ser práctica. Con fuerte desbalance, valorar el tamaño del grupo minoritario.
- **Discretas o continuas:** ambas, incluso ordinales. En discretas con muchos empates, usar una calibración que los contemple y reconocer la menor resolución del efecto.
- **Tamaño del efecto:** probabilidad de superioridad `A`, delta de Cliff o correlación biserial por rangos; estos dos últimos son equivalentes con la convención de la sección 4. No es necesario reportarlos todos.
- **Cuándo evitarlo o reinterpretarlo:** **no es, en general, un test de medianas**. Esa interpretación requiere condiciones adicionales, como distribuciones de la misma forma y dispersión que difieran solo en localización. Tampoco es un detector de cualquier diferencia de distribución: con distribuciones que se cruzan puede haber diferencias importantes y superioridad cercana a 0.5.
- **Referencia de implementación:** `scipy.stats.mannwhitneyu(Y1, Y0, alternative="two-sided", method="auto")`, revisando si la calibración elegida es adecuada para los empates y tamaños reales.

Para conservar la dirección del efecto, utilizar `U1`, asociado al primer grupo `Y1`, no `min(U1, U0)`. La elección del menor puede servir en ciertas formulaciones del contraste bilateral, pero pierde el signo necesario para interpretar el efecto.

### 3.4. Brunner-Munzel: superioridad con formas o dispersiones distintas

- **Qué contrasta:** `H0: A = 0.5`, donde `A = P(Y1 > Y0) + 0.5 P(Y1 = Y0)`. Permite distribuciones con formas o dispersiones diferentes; no exige que sean idénticas bajo esa hipótesis.
- **Tipo:** no paramétrico, basado en rangos y una estimación del error estándar adecuada al problema de superioridad con heterogeneidad.
- **Outliers:** tolerante a su magnitud por trabajar con ordenamientos.
- **Pocos registros:** la aproximación t es preferible a la normal en muestras moderadas o pequeñas. Con grupos muy pequeños, especialmente de menos de unos diez casos, considerar una versión de permutación **studentizada** especializada; no asumir que una permutación genérica de rangos resuelve el problema. Estos tamaños son orientativos, no garantías.
- **Muchos registros:** útil para comparar superioridad cuando hay heterocedasticidad o formas distintas. No elimina los problemas de un grupo minoritario casi vacío.
- **Discretas o continuas:** ambas; admite empates, aunque demasiados empates o separación completa pueden producir estimaciones de varianza degeneradas.
- **Tamaño del efecto:** `A` o delta de Cliff con IC calculado con un procedimiento compatible con la heterogeneidad. El estadístico de Brunner-Munzel no es un tamaño del efecto.
- **Cuándo evitarlo:** si se busca igualdad de medias, medianas o de toda la distribución. `A = 0.5` no implica independencia ni igualdad de distribuciones. Si falla por varianza nula o separación completa, describir la separación y no interpretar un resultado indefinido como ausencia de asociación.
- **Referencia de implementación:** `scipy.stats.brunnermunzel(Y1, Y0, alternative="two-sided", distribution="t")`.

**Relación con Mann-Whitney:** comparten el efecto de superioridad, pero no son intercambiables en toda situación. Brunner-Munzel es preferible para inferir sobre `A` con heterogeneidad; Mann-Whitney conserva utilidad para inferencia exacta sin empates bajo igualdad de distribuciones y para alternativas de desplazamiento con formas similares. Ninguno domina al otro en todos los escenarios.

### 3.5. Kolmogorov-Smirnov de dos muestras: diferencia global de distribuciones

- **Qué contrasta:** igualdad de las distribuciones de `Y1` e `Y0`. Calcula `D = sup_y |F1(y) - F0(y)|`, la mayor distancia vertical entre sus ECDF.
- **Tipo:** no paramétrico. La calibración estándar del valor p supone una variable continua.
- **Outliers:** poco sensible a la magnitud de un extremo aislado porque compara proporciones acumuladas. Esto no significa que sea especialmente potente para detectar diferencias en las colas.
- **Pocos registros:** el valor p exacto es útil con datos continuos sin empates, pero la potencia puede ser baja.
- **Muchos registros:** permite detectar cambios pequeños de localización, dispersión o forma; interpretar `D` además del valor p. Tiende a ser más sensible en la zona central que en las colas.
- **Discretas o continuas:** preferible para continuas sin empates importantes. Para discretas o continuas redondeadas con muchos empates, `D` sigue siendo descriptivo, pero **no usar sin más el valor p estándar, ni siquiera con `method="exact"`**. Calibrar `D` por permutación bajo igualdad de distribuciones o usar un procedimiento específico para datos discretos.
- **Tamaño del efecto:** el propio `D`, entre 0 y 1, es una distancia descriptiva. Añadir diferencias de cuantiles o distancia de Wasserstein si se desea una magnitud en las unidades de `Y`; Wasserstein es más sensible a extremos.
- **Cuándo evitarlo:** si se necesita una dirección única o un contraste específico de medias. Dos ECDF pueden cruzarse: el signo en el punto de máxima separación no representa una dirección global.
- **Referencia de implementación:** `scipy.stats.ks_2samp(Y1, Y0, alternative="two-sided")`, con las precauciones anteriores.

### 3.6. Brown-Forsythe: diferencia de dispersión

Aquí se usa el nombre **Brown-Forsythe para la variante de Levene centrada en la mediana**, no para otros contrastes de medias que reciben ese nombre.

- **Qué contrasta:** igualdad de varianzas mediante una comparación de las desviaciones absolutas respecto a la mediana de cada grupo. Su interpretación como cambio de escala es más clara cuando las formas son comparables; cambios de forma también pueden influir.
- **Tipo:** prueba robusta de homogeneidad de varianzas, sin exigencia de normalidad de `Y`; su calibración F es aproximada. No equivale a una prueba exacta libre de distribución.
- **Outliers:** más tolerante que Bartlett o Levene centrado en la media, pero no inmune: las desviaciones absolutas todavía crecen con la magnitud del extremo.
- **Pocos registros:** la aproximación y la potencia pueden ser deficientes; complementar con gráficos e incertidumbre. No permutar etiquetas sin justificar intercambiabilidad: igualdad de varianzas por sí sola no la garantiza.
- **Muchos registros:** útil si la ausencia se asocia con mayor o menor variabilidad aun cuando los centros sean parecidos.
- **Discretas o continuas:** ambas, pero soporte muy reducido, muchos ceros o desviaciones constantes pueden hacer el contraste poco informativo o degenerado.
- **Tamaño del efecto:** mostrar desviación estándar, rango intercuartílico (`IQR`) o desviación absoluta mediana (`MAD`) por grupo. Para una medida robusta, reportar diferencia o razón de IQR/MAD con IC; estas medidas describen dispersión, no son idénticas al parámetro de varianza contrastado. No calcular razones si el denominador es cero.
- **Cuándo evitarlo:** como supuesto «coeficiente de correlación» o como filtro obligatorio para decidir entre Student y Welch. No mide diferencias de centro.
- **Referencia de implementación:** `scipy.stats.levene(Y1, Y0, center="median")`.

Se incluye porque detecta un aspecto distinto de la asociación. Para esta finalidad exploratoria robusta se prefiere esta variante a añadir Bartlett y Levene centrado en la media como opciones paralelas.

### 3.7. Pruebas de permutación: un marco de inferencia, no otro efecto

- **Qué hacen:** eligen un estadístico, redistribuyen las etiquetas de grupo conservando `n1` y `n0`, y comparan el estadístico observado con su distribución de permutación.
- **Tipo:** remuestreo no paramétrico; no exige normalidad, pero **sí una justificación de intercambiabilidad bajo la hipótesis nula**. No es una solución automática para todo incumplimiento de supuestos.
- **Outliers:** depende del estadístico. Permutar una diferencia de medias no la vuelve robusta; una diferencia de medias recortadas o un estadístico de rangos puede ser más tolerante.
- **Pocos registros:** enumerar todas las asignaciones permite una prueba exacta bajo intercambiabilidad. Puede haber poca potencia y valores p muy discretos; la exactitud no crea información.
- **Muchos registros:** usar permutaciones aleatorias si la enumeración es inviable. Fijar semilla, registrar número de permutaciones y reconocer el error Monte Carlo. Para una cola, una estimación habitual es `(b + 1) / (B + 1)`, con `b` permutaciones al menos tan extremas entre `B`; documentar la convención bilateral y no reportar `p = 0`.
- **Discretas o continuas:** ambas. Los empates quedan incorporados al recalcular el estadístico en cada permutación.
- **Tamaño del efecto:** el correspondiente a la pregunta: diferencia de medias, diferencia recortada, `A`, delta de Cliff o `D`. El valor p de permutación no es un tamaño del efecto y el IC requiere un procedimiento adicional adecuado.
- **Advertencia sobre varianzas:** permutar etiquetas para una diferencia de medias sin estandarizar es exacto bajo igualdad de distribuciones, pero no garantiza nivel correcto bajo la sola igualdad de medias con varianzas distintas, especialmente con grupos desbalanceados. Un estadístico **studentizado**, como el t de Welch, puede ofrecer validez asintótica para esa hipótesis más débil bajo condiciones regulares; no garantiza exactitud con grupos pequeños.
- **Cuándo evitar la permutación simple:** series temporales, clústeres, observaciones relacionadas o hipótesis bajo las que los grupos no son intercambiables. Se necesita un esquema especializado o un modelo, no solo aumentar el número de permutaciones.
- **Referencia de implementación:** `scipy.stats.permutation_test`, seleccionando el estadístico y el esquema apropiados. Para dos grupos independientes e intercambiables, `permutation_type="independent"`.

## 4. Tamaños del efecto y dirección

Usar siempre la convención **ausentes menos presentes** o **ausentes mayores que presentes**. Así, un efecto positivo indica valores de `Y` mayores cuando falta `X`. Invertir la codificación de `M` invierte el signo.

### Diferencias de medias y efectos estandarizados

- **Diferencia de medias:** `Δ = media(Y1) - media(Y0)`. Es la opción principal con Welch porque conserva las unidades y no exige una desviación estándar común.
- **Hedges g:** corrección de sesgo de muestra pequeña de una diferencia de medias estandarizada. La versión habitual utiliza una desviación estándar combinada. Con varianzas distintas, explicitar el denominador y el procedimiento de IC: no asumir que existe una única desviación poblacional común ni que el IC clásico homocedástico es compatible con Welch.
- **Delta de Glass:** divide la diferencia por la desviación estándar de un grupo de referencia, por ejemplo los presentes si existe una justificación sustantiva. Es asimétrica y depende de esa referencia; no usar si su desviación es cero.
- **Diferencia recortada:** preferible con Yuen; no sustituirla por un Cohen d clásico calculado sobre datos sin recortar, pues resumiría otro aspecto.

### Correlación punto-biserial

`r_pb = Pearson(M, Y)`, con rango de -1 a 1. Es un coeficiente descriptivo legítimo entre un indicador binario y una variable numérica, también discreta si la escala es cuantitativa. No hace falta normalidad para calcularlo, pero es sensible a extremos y resume solo la separación de medias.

- Se complementa con `Δ` e IC. Su magnitud depende del balance entre ausentes y presentes: con ausencia muy rara, puede ser pequeña aun con una diferencia de medias relevante.
- Su contraste t convencional equivale al de Student con varianza combinada, **no al de Welch**. No usar automáticamente su valor p como sustituto de una inferencia robusta a heterocedasticidad.
- No añadir Pearson como otro método separado: con `M` binaria produce el mismo coeficiente. La referencia de implementación es `scipy.stats.pointbiserialr(M, Y)`.

### Probabilidad de superioridad y delta de Cliff

Para observaciones independientes elegidas al azar de cada grupo:

```text
A = P(Y1 > Y0) + 0.5 * P(Y1 = Y0)
A estimada = U1 / (n1 * n0)
delta de Cliff = 2 * A - 1
```

- `A = 0.5` o `delta = 0`: ninguna tendencia de superioridad; **no implica distribuciones iguales**.
- `A > 0.5` o `delta > 0`: los registros con `X` ausente tienden a tener `Y` mayor.
- `A < 0.5` o `delta < 0`: tienden a tener `Y` menor.
- Con esta convención, delta de Cliff y la correlación biserial por rangos para grupos independientes coinciden. Reportar una, no contarlas como dos hallazgos.
- `A` también coincide con el AUC empírico al usar `Y` como puntuación y `M = 1` como clase positiva. Es una equivalencia descriptiva; no demuestra rendimiento predictivo fuera de muestra.

### Incertidumbre e interpretación

Reportar IC, por ejemplo al 95 %, usando el procedimiento del contraste o un bootstrap adecuado para el efecto. Un bootstrap por grupo puede respetar la heterogeneidad en observaciones independientes, pero con grupos diminutos, muchos empates o estimadores degenerados sus IC pueden ser poco fiables. Con dependencia, el remuestreo debe respetar bloques o clústeres.

No aplicar umbrales universales de efecto «pequeño/mediano/grande» sin contexto. Una diferencia en precio puede ser importante aunque el coeficiente estandarizado sea pequeño. El estadístico t, U o W y el valor p no son sustitutos de una magnitud interpretable.

## 5. Flujo recomendado durante el EDA

1. **Definir la pregunta:** promedio, centro robusto, superioridad, dispersión o distribución completa. No escoger el método que produzca el menor valor p.
2. **Describir ambos grupos:** `n1`, `n0`, porcentaje de ausencia, media, mediana, desviación estándar, IQR, cuantiles y gráficos. Revisar asimetría, ceros, empates y posibles errores de captura.
3. **Elegir un análisis principal:** Welch para medias; Yuen para centros robustos; Mann-Whitney o Brunner-Munzel para superioridad según sus supuestos. Usar KS si interesa un cambio global, o Brown-Forsythe si la dispersión es la pregunta específica.
4. **Si la variable es discreta:** no asumir que «no paramétrico» resuelve automáticamente los empates. Para conteos o escalas con pocos valores, priorizar gráficos de frecuencias y una calibración compatible; si es binaria, suele ser más claro analizar una tabla 2 x 2 con diferencia de proporciones e inferencia apropiada.
5. **Reportar efecto, IC y valor p:** conservar siempre el orden ausentes/presentes y documentar la hipótesis y la calibración. Si se usa otro método como sensibilidad, explicar que puede responder a otra pregunta.
6. **Controlar multiplicidad:** al explorar muchas parejas de features, definir la familia de contrastes y ajustar los valores p, por ejemplo con Benjamini-Hochberg para controlar FDR cuando sus condiciones de dependencia sean razonables. No tratar cada contraste como una confirmación independiente.
7. **Interpretar con cautela:** en muestras grandes puede haber valores p pequeños para efectos irrelevantes; en pequeñas, un valor p grande puede reflejar falta de potencia. La asociación bivariada puede estar confundida por otras variables. Para explorar ajustes, un modelo de `P(M = 1)` con covariables es un análisis adicional, no una prueba causal.
8. **Separar exploración y modelado:** si `Y` es el objetivo predictivo, hacer este análisis y cualquier selección derivada solo con entrenamiento. No elegir features ni reglas de imputación usando validación o test.

Un reporte mínimo útil incluye: feature con faltantes, feature numérica, definición del faltante, filas excluidas por `Y` ausente en cada grupo, `n1`, `n0`, descriptivos, método, hipótesis, efecto, IC, valor p, valor p ajustado cuando corresponda y advertencias.

### Qué se puede concluir sobre el mecanismo de ausencia

- Una asociación detectada entre `M` e `Y` observada es evidencia de que la ausencia no es independiente de esa variable, con las limitaciones del contraste y de la selección de registros.
- Si `Y` está completamente observada, una dependencia real contradice MCAR respecto a `Y`. Si se excluyen filas por faltantes en `Y`, la asociación condicional puede verse afectada por selección.
- No encontrar asociación **no demuestra MCAR**: puede haber baja potencia, una dependencia no capturada por el método o relaciones con otras variables.
- Estos contrastes bivariados **no permiten distinguir MAR de MNAR**. Tampoco prueban causalidad ni determinan por sí solos cómo imputar.
- Una ausencia estructural puede requerir una categoría o indicador específico. La decisión de imputación debe incorporar el significado de la feature, no solo un resultado estadístico.

## 6. Métodos no añadidos como alternativas independientes

- **Student con varianza combinada:** sustituido por Welch como opción general de comparación de medias; no se recomienda un pretest para alternarlos.
- **Wilcoxon rank-sum:** pertenece a la misma familia y es equivalente a Mann-Whitney para grupos independientes; no confundirlo con Wilcoxon de rangos con signo, que es para datos pareados.
- **Kruskal-Wallis:** aporta poco como opción adicional en este problema de dos grupos; resulta más pertinente con tres o más grupos.
- **Pearson sobre el indicador binario:** idéntico a punto-biserial; se documenta una sola vez.
- **Biserial por rangos y delta de Cliff:** conservar una sola medida con signo documentado.

Se mantienen métodos que comparten parte del objetivo solo cuando sus supuestos, parámetro de interés o comportamiento con muestras pequeñas justifican una elección distinta. No hay un único método que domine a todos.

## 7. Asociación bivariada mediante machine learning

### 7.1. Qué aportan estos modelos

Es pertinente incluirlos como **complemento predictivo**, no como nuevos coeficientes de correlación ni sustitutos de los contrastes anteriores. Permiten explorar si conocer `Y` ayuda a predecir la ausencia de `X`, incluso mediante relaciones no lineales:

```text
Entrada del modelo: Y, una sola feature numérica
Objetivo binario: M = 1 si falta X; M = 0 si X está presente
Salida: probabilidad estimada de ausencia, P(M = 1 | Y)
```

El análisis sigue siendo **bivariado**: intervienen `Y` y `M`. Crear una base de splines de `Y` no añade nuevas variables originales. En cambio, introducir barrio, antigüedad u otras features como entradas produce un análisis con múltiples predictores y cambia la pregunta.

Por ejemplo, la ausencia del área del garaje podría ser más frecuente tanto en viviendas muy baratas como en viviendas muy caras. La correlación punto-biserial podría ser cercana a cero, mientras que un modelo no lineal podría detectar ese patrón en U.

Estos modelos no calculan la correlación entre los valores de `X` e `Y`, no recuperan los valores faltantes de `X` y no distinguen MAR de MNAR. Un rendimiento bajo tampoco demuestra independencia: puede reflejar falta de datos o un modelo inadecuado.

### 7.2. Modelos recomendados y dónde encontrarlos

| Modelo | Patrón que puede captar | Librería y objeto principal | Papel en el EDA |
|---|---|---|---|
| Regresión logística con un predictor | Relación lineal en log-odds y probabilidad monotónica | [scikit-learn: `sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | Referencia sencilla e interpretable |
| GAM logístico | Relación suave no lineal, incluidos patrones en U | [pyGAM: `pygam.LogisticGAM`, con un término `s(0)`](https://pygam.readthedocs.io/en/latest/) | Opción principal para visualizar una asociación no lineal suave |
| Árbol de clasificación poco profundo | Umbrales e intervalos con distintas probabilidades de ausencia | [scikit-learn: `sklearn.tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) | Reglas simples e interpretación por tramos |
| Random Forest de clasificación | Patrones no lineales y cambios locales flexibles | [scikit-learn: `sklearn.ensemble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) | Alternativa opcional para comprobar capacidad predictiva |

**Regresión logística con un predictor**

- **Tipo y supuestos:** modelo paramétrico para una respuesta Bernoulli. Supone `logit(P(M = 1 | Y)) = beta0 + beta1 * Y`; no requiere que `Y` sea normal ni igualdad de varianzas entre grupos.
- **Outliers:** valores extremos de `Y` pueden tener mucho apalancamiento e influir en la pendiente. Estandarizar ayuda al ajuste numérico y a interpretar la penalización, pero no elimina esa influencia.
- **Tamaño muestral:** suele requerir menos información que los modelos más flexibles, pero importan los casos de ambas clases. Con muy pocos ausentes o separación completa, la estimación sin penalización puede ser inestable o no finita. Regularizar estabiliza el ajuste, pero no crea información.
- **Discretas o continuas:** ambas si el efecto lineal en log-odds es razonable. Una escala discreta ordinal no garantiza incrementos equivalentes; tratarla como numérica supone que las distancias entre valores tienen sentido.
- **Efecto e interpretación:** `exp(beta1 * h)` es la odds ratio por un incremento `h` de `Y`; no es una razón de probabilidades. Complementar con probabilidades estimadas y su cambio entre valores relevantes de `Y`. Documentar unidades, escalado y regularización.
- **Cuándo evitarla como único análisis:** patrones en U, umbrales o saturaciones que la forma lineal no capture. Un coeficiente pequeño no descarta asociación no lineal.
- **Inferencia:** scikit-learn se orienta a predicción y no proporciona directamente los IC o valores p clásicos de los coeficientes. Para inferencia logística convencional, [statsmodels: `statsmodels.discrete.discrete_model.Logit`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html) permite obtenerlos; requiere incluir el intercepto explícitamente y comprobar convergencia, separación y supuestos. No trasladar sin más los IC clásicos a coeficientes penalizados.

**GAM logístico**

- **Tipo y supuestos:** modelo semiparamétrico, `logit(P(M = 1 | Y)) = beta0 + f(Y)`, con una función suave regularizada. No exige normalidad de `Y`; sí una complejidad adecuada a la información disponible.
- **Outliers:** no es automáticamente robusto. Valores aislados y zonas con pocos registros pueden distorsionar la curva, especialmente en los extremos del rango.
- **Tamaño muestral:** necesita suficientes casos de ambas clases a lo largo del rango de `Y`. Con pocos registros, reducir la flexibilidad; con muchos, permite describir patrones suaves más complejos. Elegir suavizado y número de funciones base dentro de la validación de entrenamiento.
- **Discretas o continuas:** especialmente útil con continuas o discretas con bastantes niveles ordenados. Con dos o tres valores distintos, una curva suave aporta poco y puede imponer una estructura artificial.
- **Efecto e interpretación:** graficar `P(M = 1 | Y)` junto con la distribución de observaciones; comparar probabilidades entre valores o cuantiles predefinidos. No hay una única odds ratio constante ni un signo global si la curva cambia de dirección. Los intervalos de la curva dependen del método y pueden no incorporar toda la incertidumbre de selección del suavizado.
- **Librerías:** pyGAM ofrece `LogisticGAM` con un término suave `s(0)`. Como alternativa de construcción en scikit-learn, una `Pipeline` con [`SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html) y `LogisticRegression` permite una regresión logística con base de splines. No es idéntica a todas las formulaciones de GAM: la penalización y la selección de suavidad pueden diferir.
- **Cuándo evitarlo:** extrapolación fuera del soporte observado, regiones casi vacías o búsqueda de umbrales exactos que los datos no permiten estimar.

**Árbol de clasificación poco profundo**

- **Tipo y supuestos:** no paramétrico. Divide el rango de `Y` en intervalos y estima una probabilidad de ausencia en cada hoja. No exige normalidad ni linealidad.
- **Outliers:** suele ser menos sensible a su magnitud que una logística lineal, pero puede aislar observaciones extremas en hojas pequeñas y crear reglas espurias.
- **Tamaño muestral:** un árbol muy simple puede ser exploratorio en muestras moderadas; los umbrales son inestables con pocos ausentes. Más datos no justifican automáticamente un árbol profundo.
- **Discretas o continuas:** ambas, si los cortes ordenados tienen sentido; no aplicar a códigos de categorías nominales como si fueran cantidades.
- **Regularización:** limitar `max_depth`, aumentar `min_samples_leaf` y, si procede, ajustar la poda mediante `ccp_alpha`. Un mínimo de filas por hoja no garantiza suficientes ausentes en ella: revisar también los recuentos de cada clase.
- **Efecto e interpretación:** reportar probabilidades y recuentos por intervalo, y diferencias de probabilidad entre intervalos relevantes. Son resúmenes dependientes del modelo, no tamaños del efecto universales. Los intervalos elegidos por el árbol no deben tratarse después como si hubieran sido definidos antes de observar los datos.
- **Cuándo evitarlo:** si se necesita una curva suave, umbrales muy precisos o probabilidades estables con pocos registros por hoja.

**Random Forest de clasificación**

- **Tipo y supuestos:** ensamble no paramétrico de árboles; puede capturar relaciones no lineales sin especificar su forma.
- **Outliers:** relativamente poco sensible a la magnitud de extremos, pero no inmune a observaciones aisladas, ruido o regiones escasas.
- **Tamaño muestral:** preferible cuando hay suficientes registros de ambas clases distribuidos por el rango de `Y`. Con pocos ausentes puede memorizar patrones locales; agregar árboles reduce variabilidad del ensamble, pero no corrige por sí solo el sobreajuste.
- **Discretas o continuas:** ambas. Con un único predictor y pocos valores distintos puede no aportar ventajas frente a un árbol simple o una curva suavizada.
- **Regularización:** revisar `min_samples_leaf`, `max_depth` y el muestreo. Con una sola feature no hay diversidad por selección de predictores: la diversidad depende principalmente del remuestreo de filas y de los árboles.
- **Efecto e interpretación:** visualizar probabilidades predichas frente a `Y` y reportar mejora predictiva fuera de muestra. La importancia interna de la única feature puede ser 1 simplemente porque es la única que produce cortes; **no mide una asociación perfecta**.
- **Cuándo evitarlo:** si la prioridad es una explicación sencilla o si no mejora de manera estable a la logística, al GAM o al árbol regularizado. No es superior por ser más complejo.

No se añaden SVM, k-NN, redes neuronales ni varios ensambles de boosting como opciones paralelas: pueden aplicarse, pero no son prioritarios para este EDA con una sola entrada. No son equivalentes exactos; se limita el catálogo por claridad y utilidad, no porque uno domine universalmente a todos.

### 7.3. Cómo cuantificar la asociación predictiva

El algoritmo no es la medida de asociación. Evaluar probabilidades de la clase `M = 1` en registros **no usados para entrenar ni seleccionar el modelo**.

| Medida | Interpretación y precauciones | Implementación |
|---|---|---|
| Log-loss | Menor es mejor; penaliza probabilidades incorrectas muy seguras. Comparar con el modelo de prevalencia | `sklearn.metrics.log_loss` |
| Brier | Promedio de `(M - probabilidad_predicha)^2`; menor es mejor. Refleja tanto calibración como capacidad de separar riesgos, no solo calibración | `sklearn.metrics.brier_score_loss` |
| ROC-AUC | Capacidad de ordenar ausentes por encima de presentes usando la probabilidad predicha. 0.5 es el nivel sin discriminación por ranking; no mide calibración | `sklearn.metrics.roc_auc_score` |
| Average precision (AP) | Complemento cuando la ausencia es rara. Su referencia sin información depende de la prevalencia; no comparar entre parejas con prevalencias distintas sin contextualizar | `sklearn.metrics.average_precision_score` |
| Curva de probabilidad frente a `Y` | Forma y magnitud local de la relación; acompañar con soporte de datos y, cuando sea posible, incertidumbre | Predicciones del modelo sobre una rejilla dentro del rango observado |

Estas funciones se documentan en la [guía de métricas de clasificación de scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html). AP es un resumen de precisión-recall, pero no es idéntico al área trapezoidal bajo esa curva.

**Modelo de referencia:** predecir para todos los registros la proporción de ausentes estimada **solo en el subconjunto de entrenamiento de cada partición**. Se puede usar [`sklearn.dummy.DummyClassifier(strategy="prior")`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html). Evaluarlo con las mismas filas y particiones que el modelo que utiliza `Y`.

- Reportar `mejora = pérdida_referencia - pérdida_modelo` para log-loss o Brier: positiva indica mejor rendimiento que la referencia, negativa indica peor generalización.
- Opcionalmente, `1 - pérdida_modelo / pérdida_referencia` expresa una mejora relativa si el denominador es positivo. No es un coeficiente de correlación ni un porcentaje de varianza explicada; puede ser negativo.
- ROC-AUC debe calcularse sobre la **probabilidad predicha**, no sobre `Y` sin transformar si el patrón es no lineal. A diferencia del AUC descriptivo de `Y` de la sección 4, este AUC depende del modelo entrenado.
- No usar accuracy como medida principal: predecir siempre «presente» puede acertar casi todas las filas cuando faltan pocos datos. Tampoco usar importancia de features o puntuación de entrenamiento como evidencia principal.
- No existe conversión universal de estas métricas a Cohen d, delta de Cliff o un coeficiente con signo. Para comparaciones sustantivas, complementar con cambios de probabilidad; para logística lineal, con odds ratios.

### 7.4. Validación y precauciones durante el EDA

1. **Conservar la preparación de la sección 1:** construir `M` antes de imputar y analizar solo `Y` observada, documentando la selección. Aunque una librería admita entradas faltantes, incluirlas aquí mezclaría la pregunta sobre los valores de `Y` con la de su propia ausencia.
2. **Evaluar fuera de muestra:** con observaciones independientes, usar validación cruzada estratificada, por ejemplo [`StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html). Verificar ambas clases en entrenamiento y evaluación. El número de particiones no debe superar el tamaño de la clase minoritaria; cumplirlo no garantiza una evaluación estable si queda solo uno o muy pocos casos por partición.
3. **Evitar fugas:** ajustar escalado, nudos de splines y demás transformaciones solo dentro del entrenamiento. Usar [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) cuando los componentes sean compatibles. Si se eligen hiperparámetros o el mejor modelo, usar validación interna y evaluación externa, o una muestra reservada que no intervenga en la elección; no reportar como independiente el mismo resultado usado para seleccionar.
4. **Respetar dependencia y desbalance:** con clústeres o tiempo, usar particiones apropiadas en lugar de una estratificación aleatoria. No aplicar sobremuestreo antes de dividir. Ponderar clases puede ser útil para ciertos objetivos, pero cambia el ajuste y puede descalibrar las probabilidades de ausencia respecto a su frecuencia real; no activar `class_weight="balanced"` automáticamente.
5. **No sobreinterpretar fluctuaciones:** comparar modelos en las mismas particiones, reportar estabilidad e incertidumbre con procedimientos apropiados. Los resultados de folds comparten entrenamiento y no son observaciones independientes para construir un IC ingenuo. La ausencia muy rara puede impedir una evaluación fiable aunque el total sea grande.
6. **Si se quiere un contraste de independencia:** puede calibrarse una puntuación predictiva permutando `M` bajo intercambiabilidad. [`sklearn.model_selection.permutation_test_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html) ofrece este marco para estimadores compatibles. Toda transformación, selección de hiperparámetros y elección de modelo basada en los datos debe repetirse dentro de cada permutación; si hubo selección, permutar solo el modelo ganador con ajustes fijos no reproduce el procedimiento completo. No confundir esta prueba con la importancia por permutación de una feature.
7. **Mantener las cautelas estadísticas:** una puntuación predictiva no es un valor p. Si se contrastan muchas parejas o se seleccionan hallazgos, considerar multiplicidad y sesgo de selección. No concluir causalidad ni un mecanismo MAR/MNAR.
8. **Si `Y` es el objetivo final del proyecto:** realizar este EDA solo con el conjunto de entrenamiento del proyecto. Su validación cruzada interna no autoriza a consultar el objetivo de validación o test para elegir features o reglas de imputación.

**Recomendación práctica:** empezar con logística; añadir un GAM si interesa una curva no lineal, o un árbol pequeño si interesan umbrales. Usar Random Forest solo como comprobación adicional si mejora de forma estable fuera de muestra. Documentar pareja analizada, clases, filas excluidas, particiones, modelo, regularización, métricas del modelo y de referencia, forma de la curva y limitaciones.

Las librerías enlazadas son opciones de implementación, no una afirmación de que estén instaladas en el proyecto. Consultar la documentación de la versión utilizada; esta ampliación documental no requiere instalar dependencias.

## 8. Referencias técnicas

- [SciPy: t de dos muestras, Welch y recorte de Yuen](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html).
- [SciPy: Mann-Whitney U, empates y métodos de inferencia](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html).
- [SciPy: Brunner-Munzel y recomendaciones para muestras pequeñas](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.brunnermunzel.html).
- [SciPy: Kolmogorov-Smirnov de dos muestras y supuesto de continuidad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).
- [SciPy: Levene y variante de Brown-Forsythe](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html).
- [SciPy: pruebas de permutación e hipótesis de intercambiabilidad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
- [SciPy: correlación punto-biserial y equivalencia con Pearson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pointbiserialr.html).