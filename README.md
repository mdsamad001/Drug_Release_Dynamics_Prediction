# Prediction of drug release dynamics from polymer microparticle characteristics 
<img width="733" height="267" alt="image" src="https://github.com/user-attachments/assets/3b459a47-4c72-4479-8a19-9685d1f072de" />


We developed a novel approach to predict drug release dynamics from material characteristics in three tasks:
- Prediction of drug release magnitude (area under the curve of drug release profile)
- Prediction of drug release type (burst vs. delayed) defined based on area under the curve of the drug release profile
- Prediction of the complete drug release profile without the conventional time-input feature

## Notebook Structure
### Model Development & Evaluation
#### Prediction of Drug Release Magnitude (AUC)
| Notebook | Model | Description |
|----------|-------|-------------|
| `Drug release AUC LinR.ipynb` | Linear Regression | Linear regression model training and validation |
| `Drug release AUC RF.ipynb` | Random Forest | Random forest model training and validation |
| `Drug release AUC XGB.ipynb` | XGBoost | Extreme gradient-boosted model training and validation |
#### Prediction of Drug Release Type (Burst vs. Delayed)
| Notebook | Model | Technique |
|----------|-------|-----------|
| `Drug release type LR.ipynb` | Logistic Regression | Standard classification |
| `Drug release type RF.ipynb` | Random Forest | Standard classification |
| `Drug release type XGB.ipynb` | XGBoost | Standard classification |
| `Drug release type LR-SMOTE.ipynb` | Logistic Regression | With SMOTE oversampling |
| `Drug release type RF-SMOTE.ipynb` | Random Forest | With SMOTE oversampling |
| `Drug release type XGB-SMOTE.ipynb` | XGBoost | With SMOTE oversampling |
#### Prediction of Complete Drug Release Profile
| Notebook | Model | Input Features |
|----------|-------|---------------|
| `Drug release profile FC-NN-LSTM.ipynb` | LSTM Neural Network | Without time feature |
| `Drug release profile FC-NN-GRU.ipynb` | GRU Neural Network | Without time feature |
| `Drug release profile XGB-Time.ipynb` | XGBoost | With time feature |
| `Drug release profile XGB-No Time.ipynb` | XGBoost | Without time feature |

### Analysis & Comparison
| Notebook | Purpose |
|----------|---------|
| `Drug release AUC analysis.ipynb` | AUC calculation and distribution of burst/delayed classes |
| `Drug release profile analysis.ipynb` | Compares all model performances for complete profile prediction |

## How to run
### Execution Order
| Order | Task | Notebooks to Run |
|-------|------|------------------|
| 1 | Data Understanding | `Drug release AUC analysis.ipynb` |
| 2 | AUC Magnitude Prediction | All notebooks in AUC magnitude section |
| 3 | Release Type Classification | Standard models, then SMOTE variants |
| 4 | Complete Profile Prediction | XGB-Time, XGB-No Time, RNNs |
| 5 | Final Analysis | `Drug release profile analysis.ipynb` |

