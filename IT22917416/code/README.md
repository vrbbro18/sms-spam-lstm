# SMS spam detection with one LSTM

This is a one-model learning project, not a complete response to the assignment's minimum model-comparison requirement. Obtain your lecturer's approval for the reduced scope. No performance results are claimed before running training.

## Windows / Python 3.12 setup
Extract this folder to your Downloads folder. Open Command Prompt inside this folder (type cmd in File Explorer's address bar).

    py -3.12 -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements.txt
    python project.py train
    python project.py predict

Internet is needed for dependencies and the first dataset download. CPU is sufficient; training time depends on your laptop. Do not work inside C:\Windows\System32. No GPU setup is needed.

If download fails, use https://archive.ics.uci.edu/dataset/228/sms+spam+collection , download and extract its ZIP, and put the file named SMSSpamCollection in the project's data folder. Re-run training.

To reopen later, open cmd in the project folder, activate .venv, then run predict. Train only when you intend to create a new experiment; training overwrites outputs. Preserve outputs before experimenting. Do not repeatedly tune based on test results.

## Method
Binary supervised text classification: ham=0, spam=1. Tokenize lowercase text, preserve punctuation and numbers, discard empty inputs and conflicting labels, deduplicate identical token sequences before splitting. Stratified split: approximately 70% training, 15% validation, 15% test, random seed 42. Vocabulary is fitted ONLY on training data, with minimum count 2 and at most 8,000 entries including padding and unknown tokens. Pad/truncate to 100 tokens. Packed sequences stop padding from influencing the final LSTM state.

Architecture: trainable 64-dimensional embedding -> one 64-unit LSTM -> dropout 0.3 -> linear output. The output is a logit during training; sigmoid converts it to a spam score for evaluation. Weighted BCEWithLogitsLoss uses a training-derived negative/positive class ratio. Adam learning rate 0.001, batch size 64, gradient clipping at 1.0, maximum 15 epochs. Early stopping patience 3 uses validation loss. Restore the best validation checkpoint. Fixed decision threshold 0.5; no test-set tuning. CPU training with two threads.

## Generated outputs
- model.pt: saved weights and vocabulary for demonstration.
- results.json: train, validation, and test accuracy, precision, recall, F1, ROC-AUC.
- config.json: actual dataset counts, split sizes, best epoch, parameter count and training duration.
- history.csv and learning_curve.png: training and validation losses.
- confusion_matrix.png: final test confusion matrix.
- class_distribution.png: cleaned dataset class counts.
- test_predictions.csv: inspect errors; actual/predicted 1 means spam.
- train_split.csv, validation_split.csv, test_split.csv: reproduce the experiment.

After successful installation, run `python -m pip freeze > installed_versions.txt` to record exact dependency versions. The requirements file uses compatibility ranges, not a fully pinned tested Windows environment.

## Dataset and attribution
UCI SMS Spam Collection, creators Tiago A. Almeida and Jose Maria Gomez Hidalgo (2011).
Dataset source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
DOI: https://doi.org/10.24432/C5CC84
Use the original dataset page for licensing and source details in your report; retain the dataset's original documentation. Actual row counts are computed at runtime; cleaning reduces the original count.
PyTorch installation reference: https://docs.pytorch.org/get-started/locally/

This code and accompanying explanations were prepared with ChatGPT assistance. Review, understand, test, adapt and acknowledge that assistance as required by your course. Describe only contributions you actually made.

## Limitations and validation status
Python syntax was checked during preparation. The student completed local Windows training and supplied metrics, configuration and charts in ../evidence/. Independent rerun and interactive prediction have not been verified. The saved checkpoint remains in the original laptop project. This is an older English SMS benchmark; performance does not establish reliability on modern scams, Sinhala, Singlish or other languages. Similar message templates can still cross splits despite exact normalized deduplication. A single split/seed gives limited robustness evidence. Class-weighted sigmoid scores are not calibrated probabilities. One architecture cannot support a model comparison. A classical baseline and additional distinct deep models are sensible extensions if scope permits.
