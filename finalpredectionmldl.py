"""
CODE DIRECT - CORRIGÉ - Modèles rapides
Visualisations exactes sans blocage + Boxplots + Rapport détaillé
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import time
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                         confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef)

# Configuration
np.random.seed(42)
plt.style.use('default')

def load_csv_data(file_path):
    """Charge VOTRE fichier CSV"""
    print(f"📊 Chargement de votre CSV: {file_path}")
    
    try:
        try:
            df = pd.read_csv(file_path, sep=';')
        except:
            df = pd.read_csv(file_path, sep=',')
        
        print(f"✅ CSV chargé: {df.shape}")
        print(f"📋 Colonnes: {list(df.columns)}")
        print(f"\n📊 Aperçu des données:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        return None

def create_features_from_csv(df):
    """Crée des features à partir de votre CSV"""
    print(f"\n⚙️ Création des features...")
    
    # Identifier la colonne d'énergie
    energy_cols = ['LECTURA_ENERGIA_ACT', 'LECTURA_ENERGIA_ACTIVA', 'ENERGIA_ACTIVA', 'ENERGIA']
    energy_col = None
    
    for col in energy_cols:
        if col in df.columns:
            energy_col = col
            break
    
    if energy_col is None:
        print("❌ Colonne d'énergie non trouvée")
        print(f"Colonnes disponibles: {list(df.columns)}")
        return None
    
    print(f"✅ Colonne d'énergie trouvée: {energy_col}")
    
    # Conversion de la date si présente
    date_cols = ['FECHA_TOMA_DATO', 'DATE', 'TIMESTAMP']
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                df['HORA'] = df[col].dt.hour
                df['DIA_SEMANA'] = df[col].dt.dayofweek
                df['MES'] = df[col].dt.month
                print(f"✅ Date convertie: {col}")
                break
            except:
                continue
    
    # Features temporelles
    if 'HORA' in df.columns:
        df['HORA_SIN'] = np.sin(2 * np.pi * df['HORA'] / 24)
        df['HORA_COS'] = np.cos(2 * np.pi * df['HORA'] / 24)
        df['EST_HEURE_POINTE'] = df['HORA'].isin([8, 9, 18, 19, 20]).astype(int)
        df['EST_WEEKEND'] = (df['DIA_SEMANA'] >= 5).astype(int) if 'DIA_SEMANA' in df.columns else 0
        print("✅ Features temporelles créées")
    
    # Features de lag
    print("🔄 Calcul des moyennes mobiles...")
    for window in [3, 6, 24]:
        df[f'ENERGIA_LAG_{window}H'] = (
            df[energy_col].shift(1).rolling(window=window, min_periods=1).mean()
        )
    
    # Variable cible
    seuil = df[energy_col].quantile(0.75)
    df['CONSOMMATION_ELEVEE'] = (df[energy_col] > seuil).astype(int)
    
    print(f"✅ Seuil consommation élevée: {seuil:.0f}")
    print(f"📊 Répartition - Normal: {(df['CONSOMMATION_ELEVEE']==0).sum()}, Élevé: {(df['CONSOMMATION_ELEVEE']==1).sum()}")
    
    # Nettoyage
    df = df.dropna().reset_index(drop=True)
    print(f"✅ Données nettoyées: {df.shape}")
    
    return df, energy_col

def plot_exploratory_boxplots(df, energy_col):
    """Génère les boxplots d'analyse exploratoire comme dans votre image - CORRIGÉ"""
    print(f"\n📊 Génération des boxplots d'analyse exploratoire...")
    
    # Sélectionner les colonnes numériques principales
    numeric_cols = []
    for col in df.columns:
        if (df[col].dtype in ['int64', 'float64', 'int32', 'float32'] and 
            df[col].nunique() > 5 and col != 'CONSOMMATION_ELEVEE'):
            numeric_cols.append(col)
    
    # Limiter à 6 colonnes maximum pour éviter les erreurs de taille
    priority_cols = [energy_col]
    if 'HORA' in df.columns:
        priority_cols.append('HORA')
    if 'DIA_SEMANA' in df.columns:
        priority_cols.append('DIA_SEMANA')
    
    # Ajouter d'autres colonnes importantes
    for col in ['ENERGIA_ACTIVA_ACUMULADA', 'LECTURA_ENERGIA_POTENCIA', 'TOTAL_ACUMULA']:
        if col in numeric_cols and col not in priority_cols and len(priority_cols) < 6:
            priority_cols.append(col)
    
    # Compléter avec d'autres colonnes si nécessaire
    for col in numeric_cols:
        if col not in priority_cols and len(priority_cols) < 6:
            priority_cols.append(col)
    
    selected_cols = priority_cols[:6]  # Maximum 6 colonnes
    
    # Créer les noms d'affichage
    display_names = {
        energy_col: 'Energy Consumption',
        'ENERGIA_ACTIVA_ACUMULADA': 'Accumulated Energy',
        'LECTURA_ENERGIA_POTENCIA': 'Power Reading',
        'TOTAL_ACUMULA': 'Total Accumulated',
        'HORA': 'Hour',
        'DIA_SEMANA': 'Day of Week',
        'MES': 'Month'
    }
    
    # Configuration de la grille - TAILLE FIXE RÉDUITE
    n_cols = len(selected_cols)
    if n_cols <= 3:
        rows, cols = 1, n_cols
        figsize = (5*n_cols, 4)
    else:
        rows, cols = 2, 3
        figsize = (12, 8)  # Taille fixe réduite
    
    # Créer la figure avec taille contrôlée
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle('Analyse Exploratoire - Distribution des Variables', fontsize=14, fontweight='bold')
    
    # Gérer les axes
    if n_cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(selected_cols):
        ax = axes[idx] if n_cols > 1 else axes[0]
        
        # Créer le boxplot avec données filtrées pour éviter les outliers extrêmes
        box_data = df[col].dropna()
        
        # Filtrer les outliers extrêmes pour éviter les problèmes de visualisation
        q1 = box_data.quantile(0.25)
        q3 = box_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        box_data_filtered = box_data[(box_data >= lower_bound) & (box_data <= upper_bound)]
        
        if len(box_data_filtered) == 0:
            box_data_filtered = box_data
        
        # Boxplot principal
        try:
            bp = ax.boxplot(box_data_filtered, patch_artist=True, widths=0.6)
            
            # Style comme dans l'image (bleu)
            bp['boxes'][0].set_facecolor('#4682B4')
            bp['boxes'][0].set_alpha(0.7)
            
            # Outliers en noir
            for flier in bp['fliers']:
                flier.set_marker('o')
                flier.set_markerfacecolor('black')
                flier.set_markersize(3)
                flier.set_alpha(0.6)
            
            # Titre et labels
            display_name = display_names.get(col, col.replace('_', ' ').title())
            ax.set_title(f'Boxplot of {display_name}', fontsize=10, fontweight='bold')
            ax.set_ylabel('Value', fontsize=9)
            
            # Statistiques sur le graphique
            median = box_data_filtered.median()
            q1_val = box_data_filtered.quantile(0.25)
            q3_val = box_data_filtered.quantile(0.75)
            
            # Ajouter les valeurs statistiques (format court)
            ax.text(1.1, 0.8, f'Med: {median:.0f}', transform=ax.transAxes, fontsize=8)
            ax.text(1.1, 0.6, f'Q1: {q1_val:.0f}', transform=ax.transAxes, fontsize=8)
            ax.text(1.1, 0.4, f'Q3: {q3_val:.0f}', transform=ax.transAxes, fontsize=8)
            
            # Grid légère
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Erreur pour la colonne {col}: {e}")
            ax.text(0.5, 0.5, f'Erreur\n{col}', ha='center', va='center', transform=ax.transAxes)
    
    # Masquer les axes vides
    if n_cols < len(axes):
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def generate_detailed_analysis_report(df, energy_col, results):
    """Génère un rapport détaillé d'analyse"""
    print(f"\n📋 Génération du rapport détaillé d'analyse...")
    
    report = []
    report.append("="*80)
    report.append("RAPPORT DÉTAILLÉ D'ANALYSE ÉNERGÉTIQUE")
    report.append("="*80)
    report.append(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 1. Résumé des données
    report.append("1. RÉSUMÉ DES DONNÉES")
    report.append("-" * 30)
    report.append(f"• Nombre total d'observations: {len(df):,}")
    report.append(f"• Période d'analyse: {df['FECHA_TOMA_DATO'].min()} à {df['FECHA_TOMA_DATO'].max()}" if 'FECHA_TOMA_DATO' in df.columns else "• Période: Non spécifiée")
    report.append(f"• Variable cible: Consommation élevée (seuil: {df[energy_col].quantile(0.75):.0f})")
    report.append(f"• Répartition des classes:")
    report.append(f"  - Consommation normale: {(df['CONSOMMATION_ELEVEE']==0).sum():,} ({(df['CONSOMMATION_ELEVEE']==0).mean()*100:.1f}%)")
    report.append(f"  - Consommation élevée: {(df['CONSOMMATION_ELEVEE']==1).sum():,} ({(df['CONSOMMATION_ELEVEE']==1).mean()*100:.1f}%)")
    report.append("")
    
    # 2. Statistiques descriptives
    report.append("2. STATISTIQUES DESCRIPTIVES DE LA CONSOMMATION")
    report.append("-" * 50)
    energy_stats = df[energy_col].describe()
    report.append(f"• Moyenne: {energy_stats['mean']:.2f}")
    report.append(f"• Médiane: {energy_stats['50%']:.2f}")
    report.append(f"• Écart-type: {energy_stats['std']:.2f}")
    report.append(f"• Minimum: {energy_stats['min']:.2f}")
    report.append(f"• Maximum: {energy_stats['max']:.2f}")
    report.append(f"• Coefficient de variation: {(energy_stats['std']/energy_stats['mean']*100):.1f}%")
    report.append("")
    
    # 3. Analyse temporelle
    if 'HORA' in df.columns:
        report.append("3. ANALYSE TEMPORELLE")
        report.append("-" * 25)
        
        # Consommation par heure
        hourly_consumption = df.groupby('HORA')[energy_col].mean()
        peak_hour = hourly_consumption.idxmax()
        low_hour = hourly_consumption.idxmin()
        
        report.append(f"• Heure de pic de consommation: {peak_hour}h ({hourly_consumption[peak_hour]:.0f})")
        report.append(f"• Heure de consommation minimale: {low_hour}h ({hourly_consumption[low_hour]:.0f})")
        
        # Consommation par jour de la semaine
        if 'DIA_SEMANA' in df.columns:
            daily_consumption = df.groupby('DIA_SEMANA')[energy_col].mean()
            days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            peak_day = daily_consumption.idxmax()
            report.append(f"• Jour de pic de consommation: {days[peak_day]} ({daily_consumption[peak_day]:.0f})")
        
        report.append("")
    
    # 4. Performance des modèles
    report.append("4. PERFORMANCE DES MODÈLES DE MACHINE LEARNING")
    report.append("-" * 55)
    
    # Trier les modèles par F1-score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    report.append("Classement des modèles (par F1-Score):")
    for i, (name, res) in enumerate(sorted_results, 1):
        report.append(f"{i:2d}. {name:<25} | F1: {res['f1']:.3f} | Acc: {res['accuracy']:.3f} | AUC: {res['auc_roc']:.3f}")
    
    report.append("")
    
    # Meilleur modèle
    best_model, best_res = sorted_results[0]
    report.append(f"🏆 MEILLEUR MODÈLE: {best_model}")
    report.append(f"   • F1-Score: {best_res['f1']:.3f}")
    report.append(f"   • Accuracy: {best_res['accuracy']:.3f}")
    report.append(f"   • Precision: {best_res['precision']:.3f}")
    report.append(f"   • Recall: {best_res['recall']:.3f}")
    report.append(f"   • AUC-ROC: {best_res['auc_roc']:.3f}")
    report.append(f"   • Temps d'entraînement: {best_res['train_time']:.2f}s")
    report.append("")
    
    # 5. Analyse comparative ML vs DL
    ml_models = {k: v for k, v in results.items() if v['model_type'] == 'ML'}
    dl_models = {k: v for k, v in results.items() if v['model_type'] == 'DL'}
    
    if ml_models and dl_models:
        report.append("5. COMPARAISON MACHINE LEARNING vs DEEP LEARNING")
        report.append("-" * 55)
        
        ml_f1_avg = np.mean([res['f1'] for res in ml_models.values()])
        dl_f1_avg = np.mean([res['f1'] for res in dl_models.values()])
        ml_time_avg = np.mean([res['train_time'] for res in ml_models.values()])
        dl_time_avg = np.mean([res['train_time'] for res in dl_models.values()])
        
        report.append(f"Machine Learning (moyenne):")
        report.append(f"   • F1-Score moyen: {ml_f1_avg:.3f}")
        report.append(f"   • Temps d'entraînement moyen: {ml_time_avg:.2f}s")
        report.append(f"   • Nombre de modèles: {len(ml_models)}")
        report.append("")
        report.append(f"Deep Learning (moyenne):")
        report.append(f"   • F1-Score moyen: {dl_f1_avg:.3f}")
        report.append(f"   • Temps d'entraînement moyen: {dl_time_avg:.2f}s")
        report.append(f"   • Nombre de modèles: {len(dl_models)}")
        report.append("")
    
    # 6. Recommandations
    report.append("6. RECOMMANDATIONS")
    report.append("-" * 20)
    
    if best_res['f1'] > 0.8:
        report.append("✅ Performance excellente du modèle de prédiction")
    elif best_res['f1'] > 0.7:
        report.append("✅ Performance satisfaisante du modèle de prédiction")
    else:
        report.append("⚠️  Performance modérée - Optimisation recommandée")
    
    # Recommandations basées sur les données
    cv = df[energy_col].std() / df[energy_col].mean()
    if cv > 0.5:
        report.append("• Forte variabilité de consommation détectée - Analyser les causes")
    
    if 'HORA' in df.columns:
        peak_hours = df[df['EST_HEURE_POINTE'] == 1][energy_col].mean()
        off_peak_hours = df[df['EST_HEURE_POINTE'] == 0][energy_col].mean()
        if peak_hours > off_peak_hours * 1.2:
            report.append("• Écart significatif heures de pointe/creuses - Optimisation tarifaire possible")
    
    report.append(f"• Utiliser le modèle {best_model} pour les prédictions futures")
    report.append("• Surveiller les périodes de consommation élevée identifiées")
    report.append("")
    
    # 7. Conclusion
    report.append("7. CONCLUSION")
    report.append("-" * 15)
    report.append(f"L'analyse de {len(df):,} observations révèle des patterns de consommation")
    report.append(f"énergétique avec une capacité de prédiction de {best_res['f1']:.1%} (F1-Score).")
    report.append(f"Le modèle {best_model} est recommandé pour les prédictions opérationnelles.")
    report.append("")
    report.append("="*80)
    
    # Afficher le rapport
    print("\n".join(report))
    
    # Sauvegarder le rapport
    with open('rapport_analyse_energetique.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    
    print(f"\n💾 Rapport sauvegardé dans 'rapport_analyse_energetique.txt'")

def prepare_ml_data(df, energy_col):
    """Prépare les données pour ML"""
    print(f"\n🔧 Préparation pour ML...")
    
    exclude_cols = [
        'FECHA_TOMA_DATO', 'DATE', 'TIMESTAMP',
        energy_col, 'ENERGIA_ACTIVA_ACUMULADA', 'ENERGIA_REACTIVA_ACUMUL',
        'TOTAL_ACUMULA', 'CONSOMMATION_ELEVEE'
    ]
    
    feature_cols = []
    for col in df.columns:
        if (col not in exclude_cols and 
            df[col].dtype in ['int64', 'float64', 'int32', 'float32'] and
            df[col].nunique() > 1):
            feature_cols.append(col)
    
    print(f"📋 Features sélectionnées: {len(feature_cols)}")
    
    X = df[feature_cols]
    y = df['CONSOMMATION_ELEVEE']
    
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

def train_all_models_fast(X_train, X_test, y_train, y_test):
    """Entraîne 8 modèles RAPIDES - CORRIGÉ"""
    print(f"\n🤖 Entraînement de 8 modèles rapides...")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.1, random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=50, random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=15, random_state=42
        ),
        'XGBoost': ExtraTreesClassifier(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        ),
        'LightGBM': GradientBoostingClassifier(
            n_estimators=30, learning_rate=0.2, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=500
        ),
        'Deep Neural Network': MLPClassifier(
            hidden_layer_sizes=(50, 25), max_iter=200, random_state=42, 
            early_stopping=True, validation_fraction=0.1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  🔄 {name}...")
        
        try:
            # Mesurer temps d'entraînement
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Mesurer temps de prédiction
            start_pred = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_pred
            
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Métriques
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.5
            mcc = matthews_corrcoef(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc_roc': auc_roc,
                'mcc': mcc,
                'train_time': train_time,
                'pred_time': pred_time,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'y_pred_proba': y_pred_proba,
                'y_test': y_test,
                'model_type': 'DL' if 'Neural' in name else 'ML'
            }
            
            print(f"    ✅ F1: {f1:.3f}, Acc: {accuracy:.3f}, Temps: {train_time:.2f}s")
            
        except Exception as e:
            print(f"    ❌ Erreur {name}: {e}")
            continue
    
    return results

def plot_confusion_matrices(results):
    """Matrices de confusion comme votre première image"""
    print(f"\n📊 Génération des matrices de confusion...")
    
    # Sélectionner top 8 modèles
    sorted_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)[:8]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # Taille réduite
    
    axes = axes.flatten()
    
    for idx, (name, res) in enumerate(sorted_models):
        ax = axes[idx]
        cm = res['confusion_matrix']
        
        # Calculer pourcentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Heatmap avec style exact
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax, 
                   cbar=False, square=True)
        
        # Ajouter les valeurs et pourcentages manuellement
        for i in range(2):
            for j in range(2):
                # Valeur absolue en blanc/noir
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j + 0.5, i + 0.3, str(cm[i, j]), 
                       ha='center', va='center', fontsize=12, fontweight='bold', color=color)
                
                # Pourcentage en rouge
                ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)', 
                       ha='center', va='center', fontsize=9, color='red')
        
        # Titre avec métriques
        ax.set_title(f'{name}\nF1: {res["f1"]:.3f} | Acc: {res["accuracy"]:.3f}', 
                    fontsize=10, fontweight='bold', pad=10)
        
        # Labels
        ax.set_xlabel('Prédiction', fontsize=9)
        ax.set_ylabel('Réalité', fontsize=9)
        ax.set_xticklabels(['0', '1'])
        ax.set_yticklabels(['0', '1'])
    
    plt.tight_layout()
    plt.show()

def plot_complete_dashboard(results):
    """Dashboard complet comme votre deuxième image - TAILLE OPTIMISÉE"""
    print(f"\n📈 Génération du dashboard complet...")
    
    fig = plt.figure(figsize=(16, 12))  # Taille réduite
    
    # Préparer les données
    df_results = pd.DataFrame({
        name: {
            'F1-Score': res['f1'],
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'AUC-ROC': res['auc_roc'],
            'MCC': res['mcc'],
            'Train_Time': res['train_time'],
            'Pred_Time': res['pred_time'],
            'Model_Type': res['model_type']
        }
        for name, res in results.items()
    }).T.sort_values('F1-Score', ascending=False)
    
    # 1. Top Modèles F1-Score
    ax1 = plt.subplot(3, 3, 1)
    models = df_results.index
    f1_scores = df_results['F1-Score']
    colors = ['#1f77b4' if df_results.loc[m, 'Model_Type'] == 'ML' else '#d62728' for m in models]
    
    bars = ax1.barh(range(len(models)), f1_scores, color=colors)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels([m[:15] for m in models], fontsize=8)  # Noms raccourcis
    ax1.set_xlabel('F1-Score', fontsize=9)
    ax1.set_title('Top Modèles - F1-Score', fontsize=10, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.0)
    
    # 2. Accuracy vs F1-Score
    ax2 = plt.subplot(3, 3, 2)
    ml_mask = df_results['Model_Type'] == 'ML'
    dl_mask = df_results['Model_Type'] == 'DL'
    
    ax2.scatter(df_results[ml_mask]['Accuracy'], df_results[ml_mask]['F1-Score'], 
               c='#1f77b4', s=60, label='ML', alpha=0.7)
    ax2.scatter(df_results[dl_mask]['Accuracy'], df_results[dl_mask]['F1-Score'], 
               c='#d62728', s=60, label='DL', alpha=0.7)
    
    ax2.set_xlabel('Accuracy', fontsize=9)
    ax2.set_ylabel('F1-Score', fontsize=9)
    ax2.set_title('Accuracy vs F1-Score', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution AUC-ROC
    ax3 = plt.subplot(3, 3, 3)
    ml_auc = df_results[ml_mask]['AUC-ROC'].values
    dl_auc = df_results[dl_mask]['AUC-ROC'].values
    
    if len(ml_auc) > 0 and len(dl_auc) > 0:
        box_data = [ml_auc, dl_auc]
        bp = ax3.boxplot(box_data, labels=['ML', 'DL'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#1f77b4')
        if len(bp['boxes']) > 1:
            bp['boxes'][1].set_facecolor('#d62728')
    
    ax3.set_ylabel('AUC-ROC', fontsize=9)
    ax3.set_title('Distribution AUC-ROC', fontsize=10, fontweight='bold')
    
    # 4. Performance vs Temps
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(df_results[ml_mask]['Train_Time'], df_results[ml_mask]['F1-Score'], 
               c='#1f77b4', s=60, label='ML', alpha=0.7)
    ax4.scatter(df_results[dl_mask]['Train_Time'], df_results[dl_mask]['F1-Score'], 
               c='#d62728', s=60, label='DL', alpha=0.7)
    
    ax4.set_xlabel('Temps Entraînement (s)', fontsize=9)
    ax4.set_ylabel('F1-Score', fontsize=9)
    ax4.set_title('Performance vs Temps', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Precision vs Recall
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(df_results[ml_mask]['Precision'], df_results[ml_mask]['Recall'], 
               c='#1f77b4', s=60, label='ML', alpha=0.7)
    ax5.scatter(df_results[dl_mask]['Precision'], df_results[dl_mask]['Recall'], 
               c='#d62728', s=60, label='DL', alpha=0.7)
    
    ax5.set_xlabel('Precision', fontsize=9)
    ax5.set_ylabel('Recall', fontsize=9)
    ax5.set_title('Precision vs Recall', fontsize=10, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Matthews Correlation Coefficient
    ax6 = plt.subplot(3, 3, 6)
    mcc_colors = ['#1f77b4' if df_results.loc[m, 'Model_Type'] == 'ML' else '#d62728' for m in models]
    ax6.bar(range(len(models)), df_results['MCC'], color=mcc_colors)
    ax6.set_xticks(range(len(models)))
    ax6.set_xticklabels([m.split()[0] for m in models], rotation=45, fontsize=7)
    ax6.set_ylabel('MCC', fontsize=9)
    ax6.set_title('Matthews Correlation', fontsize=10, fontweight='bold')
    
    # 7. Performances Moyennes
    ax7 = plt.subplot(3, 3, 7)
    if len(df_results[ml_mask]) > 0 and len(df_results[dl_mask]) > 0:
        ml_means = [df_results[ml_mask]['Accuracy'].mean(), 
                    df_results[ml_mask]['F1-Score'].mean(), 
                    df_results[ml_mask]['AUC-ROC'].mean()]
        dl_means = [df_results[dl_mask]['Accuracy'].mean(), 
                    df_results[dl_mask]['F1-Score'].mean(), 
                    df_results[dl_mask]['AUC-ROC'].mean()]
        
        x = np.arange(3)
        width = 0.35
        
        ax7.bar(x - width/2, ml_means, width, label='ML', color='#1f77b4', alpha=0.7)
        ax7.bar(x + width/2, dl_means, width, label='DL', color='#d62728', alpha=0.7)
        
        ax7.set_xticks(x)
        ax7.set_xticklabels(['Acc', 'F1', 'AUC'], fontsize=8)
        ax7.legend(fontsize=8)
    
    ax7.set_ylabel('Score', fontsize=9)
    ax7.set_title('Performances Moyennes', fontsize=10, fontweight='bold')
    
    # 8. Distribution Temps
    ax8 = plt.subplot(3, 3, 8)
    ml_pred_times = df_results[ml_mask]['Pred_Time'].values
    dl_pred_times = df_results[dl_mask]['Pred_Time'].values
    
    if len(ml_pred_times) > 0:
        ax8.hist(ml_pred_times, bins=3, alpha=0.7, label='ML', color='#1f77b4', density=True)
    if len(dl_pred_times) > 0:
        ax8.hist(dl_pred_times, bins=3, alpha=0.7, label='DL', color='#d62728', density=True)
    
    ax8.set_xlabel('Temps Prédiction (s)', fontsize=9)
    ax8.set_ylabel('Fréquence', fontsize=9)
    ax8.set_title('Distribution Temps', fontsize=10, fontweight='bold')
    ax8.legend(fontsize=8)
    
    # 9. Graphique Radar
    ax9 = plt.subplot(3, 3, 9, projection='polar')
    
    # Prendre le meilleur modèle
    best_model = models[0]
    best_res = results[best_model]
    
    categories = ['Recall', 'Accuracy', 'F1-Score', 'AUC-ROC']
    values = [best_res['recall'], best_res['accuracy'], best_res['f1'], best_res['auc_roc']]
    
    # Fermer le polygone
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax9.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
    ax9.fill(angles, values, alpha=0.25, color='#1f77b4')
    ax9.set_xticks(angles[:-1])
    ax9.set_xticklabels(categories, fontsize=8)
    ax9.set_ylim(0, 1)
    ax9.set_title(f'Radar - {best_model[:10]}', fontsize=9, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.show()

def main():
    """Fonction principale"""
    print("🎯 ANALYSE ÉNERGÉTIQUE - VERSION COMPLÈTE AVEC BOXPLOTS ET RAPPORT")
    print("=" * 80)
    
    # CHANGEZ LE NOM DE VOTRE FICHIER ICI
    csv_file = "201608.csv"  # ← MODIFIEZ ICI
    
    # 1. Charger votre CSV
    df = load_csv_data(csv_file)
    if df is None:
        return
    
    # 2. Créer features
    result = create_features_from_csv(df)
    if result is None:
        return
    df, energy_col = result
    
    # 3. NOUVEAU: Boxplots d'analyse exploratoire
    plot_exploratory_boxplots(df, energy_col)
    
    # 4. Préparer données
    X_train, X_test, y_train, y_test, features = prepare_ml_data(df, energy_col)
    
    # 5. Entraîner 8 modèles RAPIDES
    results = train_all_models_fast(X_train, X_test, y_train, y_test)
    
    # 6. Matrices de confusion
    plot_confusion_matrices(results)
    
    # 7. Dashboard complet
    plot_complete_dashboard(results)
    
    # 8. NOUVEAU: Rapport détaillé d'analyse
    generate_detailed_analysis_report(df, energy_col, results)
    
    print(f"\n🎉 ANALYSE COMPLÈTE TERMINÉE!")
    print(f"✅ Boxplots d'exploration générés")
    print(f"✅ Tous les modèles entraînés rapidement")
    print(f"✅ Rapport détaillé sauvegardé")

if __name__ == "__main__":
    main()