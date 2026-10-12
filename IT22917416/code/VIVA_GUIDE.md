# Understand this project before your viva

## A short introduction
This project classifies English SMS messages as spam or legitimate messages using an LSTM neural network. It learns word embeddings and sequential features from labeled examples. I evaluate it on a held-out test split using precision, recall, F1, accuracy and ROC-AUC. Fill in your actual findings only after training.

## Questions to practise
1. **Why supervised learning?** Every training message has a known ham/spam label.
2. **What is an embedding?** A learned dense vector for each vocabulary token; similar task-relevant tokens can acquire useful representations.
3. **Why LSTM?** It processes ordered tokens using a memory cell and input, forget and output gates. This provides a way to learn dependencies across a message. It is a design choice, not proof that it outperforms simpler methods.
4. **Why no softmax?** Binary classification needs one logit; sigmoid maps that logit to a score between zero and one.
5. **Why BCEWithLogitsLoss?** It combines sigmoid and binary cross-entropy numerically stably. We do not apply sigmoid before this loss.
6. **Why class weighting?** Spam is less frequent. Training-derived positive weighting increases the loss contribution of spam examples. It may change the precision/recall tradeoff.
7. **Why not accuracy alone?** Predicting only the majority ham class can look accurate while missing spam.
8. **Precision versus recall?** Spam precision is TP/(TP+FP); spam recall is TP/(TP+FN). A false positive blocks a legitimate message; a false negative lets spam through.
9. **F1?** The harmonic mean of precision and recall: 2PR/(P+R). Spam F1 is useful for the minority class; macro F1 weights both classes equally.
10. **ROC-AUC?** A ranking measure across decision thresholds, calculated from scores rather than hard labels. PR-AUC would be another useful metric for imbalance.
11. **What are the three data splits?** Training learns parameters; validation chooses the stopping point; test estimates final generalization. Test data must not guide preprocessing or hyperparameters.
12. **How is leakage reduced?** Deduplicate normalized token sequences before splitting and fit vocabulary only on training data. Similar templates may still cause optimistic results.
13. **What is overfitting?** The model learns training-specific patterns that do not generalize. Falling training loss with rising validation loss is a warning.
14. **What does dropout do?** Randomly zeros some hidden features during training to reduce reliance on particular features. Disabled in evaluation mode.
15. **What is early stopping?** Stop after three epochs without validation-loss improvement and restore the best checkpoint.
16. **What are epochs and batches?** An epoch traverses the training set; a batch is a subset used for one gradient update.
17. **What does Adam do?** Updates weights using running estimates of gradients and squared gradients. Learning rate controls update scale.
18. **Why gradient clipping?** Limits gradient norm to reduce unstable large updates in recurrent training.
19. **How does unseen text work?** Known tokens map to learned IDs; unknown tokens map to UNK. Messages are truncated at 100 tokens. Packed sequences avoid processing padding as content.
20. **What would improve the study?** Additional distinct models, a classical baseline, repeated seeds, template-grouped splits, modern external data, error analysis, and validation-based threshold selection.
21. **What did you contribute?** Answer honestly using the work you actually performed. Acknowledge AI-assisted implementation.

## Code walkthrough
- tokens / encode: text -> integer IDs and true sequence lengths.
- SpamLSTM: embedding, packed LSTM, dropout and output logit.
- load_data: download, parse, clean and deduplicate.
- train: split, training-only vocabulary, optimize, early stop, save and evaluate.
- evaluate: evaluation mode, disable gradients, gather loss and scores.
- predict: load the saved model and use the same preprocessing.

## Five-minute demonstration
1. Explain the problem, source and actual cleaned dataset size from config.json.
2. Show the architecture and explain the train/validation/test split.
3. Show learning_curve.png and explain whether it suggests overfitting.
4. Show the confusion matrix and actual spam precision, recall and F1 from results.json.
5. Run `python project.py predict`. Try a legitimate message and a promotional message. These hand-written examples are demonstrations, not test evidence.
6. Explain one false positive or false negative from test_predictions.csv, limitations and your actual contribution.

## Record after training
Actual raw/clean rows: ____ / ____
Train/validation/test sizes: ____ / ____ / ____
Best epoch: ____
Test accuracy: ____
Spam precision/recall/F1: ____ / ____ / ____
Test ROC-AUC: ____
Training time: ____
Observed error example and plausible cause: ____

Do not claim a metric or superiority without the experiment supporting it. This package is one model and does not satisfy the brief's multi-model comparison requirement by itself.
