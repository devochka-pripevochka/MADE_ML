import joblib

class Classifier(object):
    def __init__(self):
        self.vectorizer = joblib.load("/Users/karina/Desktop/ml/HWs/hw4/genres_vectorizer_dump.pkl")
        self.model = joblib.load("/Users/karina/Desktop/ml/HWs/hw4/genres_model_dump.pkl")
        self.target_names = joblib.load("/Users/karina/Desktop/ml/HWs/hw4/genres_target_dump.pkl")
    
    def get_name_by_label(self, label):
        try:
            return self.target_names[label]
        except:
            return "label error"

    def predict_text(self, text):
        try:
            vectorized = self.vectorizer.transform([text])
            return self.model.predict(vectorized)[0] 
        except:
            print("prediction error")
            return None 

    def get_result_message(self, text):
        prediction = self.predict_text(text)
        return self.get_name_by_label(prediction)
