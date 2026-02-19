import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge


class DataPreprocessor:
    
    def __init__(self):
        # StandardScaler normaliza las características para que tengan media=0 y desviación estándar=1
        # Esto ayuda al modelo a converger más rápido y evita que características con valores grandes dominen
        self.scaler = StandardScaler()
            
    def load_data(self, filepath='insurance.csv'):
        """
        Carga el dataset desde un archivo CSV.
        """
        return pd.read_csv(filepath)
    
    def clean_data(self, df):
        """
        Convierte variables categóricas a numéricas.
        """
        df_clean = df.copy()
        # Convertir 'sex' de texto a número: female=0, male=1
        df_clean['sex'] = df_clean['sex'].map({'female': 0, 'male': 1})
        # Convertir 'smoker' de texto a número: no=0, yes=1
        df_clean['smoker'] = df_clean['smoker'].map({'no': 0, 'yes': 1})
        # Convertir 'region' de texto a número: cada región tiene un código único
        df_clean['region'] = df_clean['region'].map({'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3})
        return df_clean
    
    def create_features(self, df):
        """
        Crea nuevas características combinando las existentes.
        Los costos médicos NO suben linealmente, sino en curva.
        Estas características capturan esas relaciones no-lineales:
        - smoker_age = smoker × age (fumadores mayores cuestan exponencialmente más, no el doble)
        - bmi_smoker = bmi × smoker (sobrepeso + fumar tiene efecto multiplicativo en costos)
        - age_bmi = age × bmi (edad y peso juntos afectan de forma combinada)
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
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
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

    def __init__(self):
        pass


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('insurance.csv')
    df = preprocessor.clean_data(df)
    df = preprocessor.create_features(df)
    #Dividir en conjuntos de Train (70%) y Test (30%)
    X_train, X_test, y_train, y_test = preprocessor.split_data(df)
    
    # Escalar solo X y no Y
    # X_train_scaled, X_test_scaled tienen las características normalizadas
    # y_train, y_test mantienen los costos reales
    X_train_scaled, X_test_scaled = preprocessor.scale_and_transform(X_train, X_test)
    
    #TODO: Entrenar el modelo con X_train_scaled y y_train, luego evaluar con X_test_scaled y y_test
    trainer = ModelTrainer()
    trainer.train(X_train_scaled, y_train)
    


