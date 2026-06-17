# Manual de Usuario — Dashboard Zomato Analytics

## Acceso

Una vez que el sistema esté corriendo, abre tu navegador y ve a:

```
http://localhost:8501
```

## Estructura del dashboard

El dashboard tiene dos pestañas en la parte superior:

- **📊 Vista de Negocio** — para gerentes y tomadores de decisión
- **🔬 Vista Técnica** — para científicos de datos e ingenieros

---

## 📊 Vista de Negocio

### Sección 1: Segmentos de Restaurantes

Al abrir el dashboard verás tarjetas con el perfil de cada segmento identificado por el modelo:

- Número de restaurantes en el segmento
- Rating promedio del segmento
- Votos promedio
- Costo promedio para dos personas

Los segmentos van desde económico hasta alta gama según las características del restaurante.

### Sección 2: Gráficos de distribución

- **Gráfico de torta:** muestra qué porcentaje del total representa cada segmento
- **Gráfico de barras:** compara el rating promedio entre segmentos

### Sección 3: Explorador de Restaurantes

Filtra los restaurantes usando los controles:

1. **Filtrar por segmento:** selecciona un segmento específico o "Todos"
2. **Rating mínimo:** desliza para ver solo restaurantes sobre cierto rating

La tabla muestra hasta 50 resultados con nombre, ubicación, rating, votos y costo.

### Sección 4: Simulador de Rating 🎮

Predice el rating que tendría un restaurante hipotético:

1. Ingresa el número de votos esperados
2. Ingresa el costo aproximado para dos personas (en rupias indias)
3. Indica si acepta pedidos online y reservas de mesa
4. Selecciona la ubicación y tipo de servicio
5. Haz clic en **🔮 Predecir Rating**

El resultado muestra:
- El rating predicho sobre 5.0
- Una interpretación en texto (Excelente, Muy bueno, Bueno, Regular)
- Un indicador visual tipo velocímetro con colores semafóricos

> **Nota:** el simulador requiere que la API esté corriendo en el puerto 8000.

---

## 🔬 Vista Técnica

### Sección 1: Proyección PCA 2D

Scatter plot que muestra los restaurantes proyectados en dos dimensiones (PCA), coloreados por cluster. Permite visualizar la separación entre grupos.

- Cada punto es un restaurante
- El color indica el cluster al que pertenece
- Los ejes muestran el porcentaje de varianza explicada

### Sección 2: Métricas de Clustering

Cuatro métricas del modelo K-Means:

| Métrica | Qué indica |
|---|---|
| **Silhouette Score** | Qué tan bien separados están los clusters (cercano a 1 es mejor) |
| **Calinski-Harabasz** | Densidad y separación (más alto es mejor) |
| **Davies-Bouldin** | Similitud entre clusters (cercano a 0 es mejor) |
| **Clusters** | Número de grupos encontrados |

### Sección 3: Comparativa de Modelos de Regresión

Comparación entre Ridge Regression y Random Forest:

- **R²:** qué porcentaje de la varianza del rating explica el modelo
- **RMSE:** error promedio en unidades de rating
- **MAE:** error absoluto promedio

El gráfico de barras agrupadas permite comparar ambos modelos visualmente.

### Sección 4: Distribuciones de Variables

- **Histograma de ratings:** distribución de calificaciones en el dataset
- **Histograma de votos:** distribución de popularidad (sin outliers extremos)
- **Scatter rating vs costo:** relación entre precio y calificación, coloreado por cluster
