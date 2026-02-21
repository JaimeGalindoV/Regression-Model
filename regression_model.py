import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

class DataPreprocessor:
    
    def __init__(self):
        # StandardScaler normaliza las características para que tengan media=0 y desviación estándar=1
        # Esto ayuda al modelo a converger más rápido y evita que características con valores grandes dominen
        self.scaler = StandardScaler()
            
    def load_data(self, filepath='./insurance.csv'):
        """
        Carga el dataset desde un archivo CSV.
        """
        return pd.read_csv(filepath)
    
    def clean_data(self, df):
        """
        Convierte variables categóricas a numéricas.
        """
        df_clean = df.copy()
        # Convertir 'sex' de texto a número:
        # Aplicar One-Hot Encoding para crear columnas nuevas para definir el sexo de la persona
        df_clean = pd.get_dummies(df_clean, columns=['sex'], prefix='sex')
        # Convertir 'smoker' de texto a número: no=0, yes=1
        df_clean['smoker'] = df_clean['smoker'].map({'no': 0, 'yes': 1})
        # Convertir 'region' de texto a número: 
        # Aplicar One-Hot Encoding para crear columnas nuevas para definir la region de la persona
        df_clean = pd.get_dummies(df_clean, columns=['region'], prefix='region')
        return df_clean
    
    def create_features(self, df):
        """
        Crea nuevas características combinando las existentes.
        Los costos médicos NO suben linealmente, sino en curva.
        Estas características capturan esas relaciones no-lineales:
        - smoker_age = smoker * age (fumadores mayores cuestan exponencialmente más, no el doble)
        - bmi_smoker = bmi * smoker (sobrepeso + fumar tiene efecto multiplicativo en costos)
        - age_bmi = age * bmi (edad y peso juntos afectan de forma combinada)
        - bmi_squared = bmi² (sobrepeso severo cuesta mucho más que el doble)
        - age_squared = age² (un anciano de 60 no cuesta el doble que uno de 30, sino 4x o más)
        """
        df['smoker_age'] = df['smoker'] * df['age']
        df['bmi_smoker'] = df['bmi'] * df['smoker']
        df['age_bmi'] = df['age'] * df['bmi']
        df['bmi_squared'] = df['bmi'] ** 2
        df['age_squared'] = df['age'] ** 2
        return df
    
    def split_data(self, df, test_size=0.3, random_state=100):
        """
        Divide los datos en conjuntos de entrenamiento (70%) y prueba (30%).        
        """
        # X contiene todas las columnas excepto "charges" 
        X = df.drop("charges", axis=1)
        # y contiene solo "charges" que es lo que se va a predecir
        y = df["charges"]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)    
    
    def scale_and_transform(self, X_train, X_test):
        """
        Normaliza X (age, sex, bmi, children, smoker, region) para que tengan la misma escala.
        Solo se escala X, no Y (charges) porque queremos predecir dólares reales.
        """
        # fit_transform: Aprende la escala de X_train y la aplica
        X_train_scaled = self.scaler.fit_transform(X_train)
        # transform: Aplica la misma escala de X_train a X_test 
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled


class ModelTrainer:
    def __init__(self, alpha =1.0):
        self.model = Ridge(alpha)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Modelo entrenado correctamente.")

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"MSE: {mse}")
        return mse

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('./insurance.csv')
    df = preprocessor.clean_data(df)
    df = preprocessor.create_features(df)
    #Dividir en conjuntos de Train (70%) y Test (30%)
    X_train, X_test, y_train, y_test = preprocessor.split_data(df)
    
    # Escalar solo X y no Y
    # X_train_scaled, X_test_scaled tienen las características normalizadas
    # y_train, y_test mantienen los costos reales
    X_train_scaled, X_test_scaled = preprocessor.scale_and_transform(X_train, X_test)
    
    minAlpha = 0
    minMse = 9999999999
    for i in np.arange(0, 3, 0.1):
        print(i)
        trainer = ModelTrainer(alpha=i)
        trainer.train(X_train_scaled, y_train)
        mse = trainer.evaluate(X_test_scaled, y_test)
        if mse < minMse:
            minAlpha = i
            minMse = mse

        print("\n")

    print("El mejor resultado fue: \nAlpha: " + str(minAlpha) + "\nMSE: " + str(minMse))    


