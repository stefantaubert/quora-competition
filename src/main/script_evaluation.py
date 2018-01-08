import sys
import time
import model_training
import prediction
import feature_selection
import evaluation
import feature_extraction

def full_run():
    start_at(1)

def run_only(run_only_iterations):
    # Root-Verzeichnis aus Parametern lesen und Pfade initialisieren

    all_start = time.time()

    for iteration in run_only_iterations:
        start = time.time()
        print("Starting new iteration... Current:", str(iteration))

        feature_selection.select_features_at(iteration)

        # Modell trainieren
        #model_training.train_and_save_model()

        # Evaluation ausführen
        evaluation.write_evaluation(iteration, True)

        print('Iteration ', str(iteration), 'finished. Duration: ', str(round((time.time() - start) / 60, 2)), 'min')

    print("Overall duration:", str(round((time.time() - all_start) / 60, 2)), 'min')


def start_at(iteration):
    # Root-Verzeichnis aus Parametern lesen und Pfade initialisieren
    count_of_iterations = iteration - 1
    max_iterations = 1000
    all_start = time.time()

    while not count_of_iterations == max_iterations:

        start = time.time()
        count_of_iterations = count_of_iterations + 1
        print("Starting new iteration... Current:", str(count_of_iterations))

        if count_of_iterations == iteration:
            feature_selection.select_features_at(iteration)
        else:
            feature_selection.select_features()

        # Modell trainieren
        model_training.train_and_save_model()

        # Evaluation ausführen
        evaluation.write_evaluation(count_of_iterations, False)

        print('Iteration ', str(count_of_iterations), 'finished. Duration: ', str(round((time.time() - start) / 60, 2)),
              'min')

    print("all", str(max_iterations), "iterations finished.")
    print("Overall duration:", str(round((time.time() - all_start) / 60, 2)), 'min')

# Vorraussetzung ist, dass alle Features die unten verwendet werden in der Feature-Backup Datei vorhanden sind.
if __name__ == '__main__':
    fullrun = True
    #fullrun = False
    if fullrun:
        #feature_extraction.extract_features(False, True)
        full_run()
        # run_only([969, 349, 515])
    else:
        feature_extraction.extract_features(False, True)
        start_at(175)
        # run_only([969, 349, 515])


