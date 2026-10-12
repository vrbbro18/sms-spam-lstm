# Ten-minute demonstration and narration guide
Lakshan RHR - IT22917416
Rehearse and use your own words. These time windows include screen navigation and demonstration; the text alone will not fill ten minutes. Do not claim unperformed contributions.

## 0:00-1:00 | Problem and scope
Show Report.pdf cover. Introduce yourself and the goal: classify English SMS messages into ham (legitimate) and spam. Explain why missing spam and flagging legitimate messages are different errors. State that this implementation investigates one LSTM architecture and does not establish superiority over other models.

## 1:00-2:00 | Dataset
Show Report.pdf dataset table. Explain the UCI source and its attribution. The local parser read 5,574 rows; preprocessing retained 5,159 unique usable messages. The clean set has 4,517 ham and 642 spam messages. Spam is only about 12.44%, so accuracy alone is inadequate. Distinguish local recorded counts from any repository headline count.

## 2:00-3:00 | Preprocessing and leakage
Show tokens(), encode() and load_data(). Explain lowercasing, tokens, normalization-based deduplication, conflicting-label removal and empty-message filtering. Identical normalized inputs are removed before splitting. Explain train/validation/test sizes: 3,611/774/774. The vocabulary is fitted only on training data. State that near-duplicate templates may still cross splits.

## 3:00-4:00 | Architecture
Show SpamLSTM. Trace a message through integer IDs, 64-dimensional embeddings, a 64-unit LSTM, dropout 0.3 and one output logit. Explain LSTM memory and gates in plain language. Sigmoid produces a spam score at inference. Packed sequences prevent padding from being processed as meaningful text. Total parameters: 235,713. Unknown tokens map to UNK; maximum length is 100 tokens.

## 4:00-5:00 | Training
Show train(). Explain Adam at 0.001, batches of 64, weighted binary cross-entropy and gradient clipping at 1.0. The positive weight, about 7.04, comes only from training labels. Explain why BCEWithLogitsLoss takes logits directly. Mention seed 42, CPU execution and measured training-loop time of 21.73 seconds; hardware specifications were not recorded.

## 5:00-6:00 | Learning curve
Show evidence/learning_curve.png. Explain that lower loss is better. Validation loss was lowest at epoch 6, about 0.2652. Training loss kept falling but validation loss failed to improve for epochs 7-9. Early stopping restored epoch 6. This is evidence of overfitting risk, not proof of perfect generalization. Training and validation loss are computed under different dropout modes.

## 6:00-7:00 | Results
Show Report.pdf results and confusion matrix. Accuracy is 96.90%, spam precision 81.58%, recall 96.88%, F1 88.57%, ROC-AUC 0.9872. Explain 657 correct ham, 21 false alarms, 3 missed spam and 93 detected spam. Precision is 93/114; recall is 93/96. A high recall is useful but legitimate-message false alarms matter.

## 7:00-8:00 | Live prediction
In the ORIGINAL trained project run `python project.py predict`. Type a normal meeting message and a promotional prize message. Explain actual outputs, including mistakes if any. The score is not a calibrated probability. These examples illustrate operation and are not additional unbiased test evidence. Type quit. If prediction fails, explain the issue; do not present a staged output as a live prediction.

## 8:00-9:00 | Critical analysis
Explain that always predicting ham yields 87.60% test accuracy but zero spam recall. Our accuracy is 9.30 percentage points higher; this is an analytical baseline, not a trained competing model. Discuss train/test spam F1 gap (96.60% vs 88.57%), older English data, unknown words and one random seed. Explain that class weighting may contribute to recall/precision behavior but no unweighted ablation was performed.

## 9:00-10:00 | Limitations, contributions and conclusion
State that one architecture does not meet the brief's four-model comparison requirement without an approved exception. Explain useful future work: distinct architectures, a TF-IDF baseline, grouped template splits, multiple seeds, PR-AUC and validation-selected thresholds. Describe your actual local execution and review. Acknowledge ChatGPT assistance with code and writing. Conclude with the measured result and its limits, not a claim of production readiness.
