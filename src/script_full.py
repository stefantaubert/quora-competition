import sys
import time
import feature_extraction
import model_training
import prediction
import feature_selection

if __name__ == '__main__':
    start = time.time()
    # Root-Verzeichnis aus Parametern lesen und Pfade initialisieren
    root = sys.argv[1]

    # Features extrahieren
    #feature_extraction.extract_features(root, True, True)

    feature_selection.select_features_at(root, 969)

    # Modell trainieren
    model_training.train_and_save_model(root)

    # Test-Daten klassifizieren
    prediction.predict_and_write_testdata(root)

    #print('overall duration: ', str(round(time.time()-start, 2) / 60), 'min')

