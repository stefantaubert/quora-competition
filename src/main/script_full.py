import sys
import time
import feature_extraction
import model_training
import prediction
import feature_selection

if __name__ == '__main__':
    start = time.time()

    # Features extrahieren
    #feature_extraction.extract_features(root, True, True)

    feature_selection.select_features_at(969)

    # Modell trainieren
    model_training.train_and_save_model()

    # Test-Daten klassifizieren
    prediction.predict_and_write_testdata()

    #print('overall duration: ', str(round(time.time()-start, 2) / 60), 'min')

