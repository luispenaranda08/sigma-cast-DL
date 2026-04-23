# Sigma-Cast: Bitcoin Volatility Forecasting API

## Deep Learning Project

---

## 👥 Integrantes
- Luis David Peñaranda Pérez  
- David Márquez  
- Johan David Díaz López  
- Cristian Linero  

---

## 🎯 Objetivos del Proyecto

1. Predecir la volatilidad realizada de Bitcoin a 7 días usando histórico multi-resolución  
2. Comparar arquitecturas de Deep Learning: **MLP, RNN y LSTM** con grid search y validación temporal  
3. Validar estadísticamente contra un baseline naive usando **test de Diebold-Mariano**  
4. Implementar monitoreo de drift en runtime  
5. Industrializar el modelo en una **API REST con FastAPI + Docker + CI/CD**

---

## 📊 Resultados Clave

- 🧠 Mejor modelo: **LSTM (lag=21)**  
- 📉 RMSE en test: **0.1896**  
- 📈 Validación estadística: **p-value = 0.018**  
- ⚙️ API: **34/34 tests pasando en CI/CD**

---

## 🗂️ Estructura del Proyecto

- `1_eda_volatility_final.ipynb` → Análisis exploratorio multi-resolución  
- `2_feature_engineering.ipynb` → Creación de lags y splits temporales  
- `3_model_training.ipynb` → Grid search (MLP, RNN, LSTM)  
- `4_model_evaluation.ipynb` → Métricas, test BDS, curvas  
- `volatility_api/` → API en FastAPI + modelo empaquetado  

---

## 🔗 Enlaces

- 🔥 [Repositorio en GitHub](https://github.com/luispenaranda08/sigma-cast-DL)