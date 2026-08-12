# Capstone: Spam Screening for an Email Service Provider

Notebook: [data-exploration.ipynb](data-exploration.ipynb)

## Research Question

Email Service Providers (such as SendGrid, Mailchimp) need to maintain and protect the trust of the email channel; can the subject and body of an outbound message be scanned and used to detect and block spam before delivery?

## Data Source

Kaggle dataset: `nitishabharathi/email-spam-dataset`, fetched via `kagglehub` at [https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset/](https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset/).

**Change from the Module 16 proposal:** the original plan named a single combined Enron + TREC 2007 corpus. The actual Kaggle dataset turned out to contain **three separate real email sources**, not one combined set:

- **Enron**; data set extracted from emails obtained from the collapse of Enron in 2001, constructed by V. Metsis, I. Androutsopoulos and G. Paliouras and described in [Spam Filtering with Naive Bayes - Which Naive Bayes?](https://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf)
- **Spam Assassin**; Data set, collated by Justin Mason between 2002-2003 as a **testing** [resource](https://spamassassin.apache.org/old/publiccorpus/readme.html) for spam filter developers.
- **LingSpam**;  data set collated by researchers at NCSR Demokritos in 2000 from a linguistics mailing list as a benchmark for anti-spam filtering.

Each source has a `Body` column (raw text, generally starting with a `Subject:` header line) and a `Label` column (`spam/1` or `ham/0`).

## Methodology

This project followed the CRISP-DM framework.

**Data cleaning.** Each message's raw text starts with `Subject: <text>` on the first line. This pattern was checked and confirmed 100% consistent, across all three sources, both case-sensitive and case-insensitive. A plain string split on the first newline (no regex needed) separates the subject from the body. Rows with a missing body were dropped. Exact duplicate messages (313 out of roughly 10,000 in Enron) were dropped before splitting into train and test, to avoid the same message appearing in both splits and inflating the test score.

**Feature engineering.** Built structural features on both the subject and the body: character length, word count, capitalization ratio, digit ratio, exclamation/dollar-sign/asterisk/question-mark counts, and boolean flags for embedded links (`http://` and bare `www.` links checked separately). These features are not yet used in the baseline model below — held in reserve so the first baseline is a clean, simple comparison point (likely next step for Module 24).

**Exploratory data analysis.** Two separate passes, answering two different questions:

- *Does this feature separate spam from ham?*  Compared each engineered feature's values to see if they matched across the label sets (e.g. if a feature's values are evenly distributed across spam and ham, they do not add value to the model).

- *Are there outlier values in this feature, regardless of label?*  For the numeric features outliers were identified but not dropped.  Spam messages are likely to exhibit some anomalous content, so removing outliers could throw away signal not noise.

**Key modeling decision: the three sources were kept fully separate through modeling, not combined.**

- The Enron data set was split into train and test sets (80/20 distribution) stratified on the `Label` column for a consistent distribution.
- The Spam Assassin and LingSpam data sets were purely used as validation sets.  They were never used to fit or train a model, only to evaluate model performance on unseen email sources.
- The vectorizer (TF-IDF) was only fit on the Enron training set.  The Spam Assassin and LingSpam sets were only transformed with this vectorizer (avoiding leaking their vocabulary into the model).

This was an intentional setup to mirror the case of highly divergent email sources (across platform customers) and the need to effectively detect spam on novel streams of emails.

**Baseline model.** The baseline model used TF-IDF (with a single consolidated subject+body field) and a [Multinomial Naive Bayes classifier](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes).

## Outlier Analysis

Evaluated each engineered feature for outliers using IQR fencing for reporting only (these features are not used in the baseline model). For most features, outliers skew heavily towards spam:

- `subject_dollar_count` outliers are 97% spam
- `subject_exclaim_count` outliers are 90% spam
- `subject_length` outliers are 88% spam
- `body_exclaim_count` outliers are 84% spam

This indicates that these structural features are meaningful as indicators of spam, making a strong case for incorporating the engineered features into a future model.

## Results

| Dataset | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Enron (test, in distribution) | 0.993 | 0.981 | 0.987 | 0.999 |
| Spam Assassin (out of distribution) | 0.335 | 0.974 | 0.498 | 0.903 |
| LingSpam (out of distribution) | 0.542 | 0.957 | 0.692 | 0.973 |

Key observations:

- **Excellent baseline in distribution**.  On the held out Enron test set, the model finds spam with very high precision and recall.
- **Recall generalizes, precision does not**.  On the Spam Assassin and LingSpam sets recall stays high (model catches most real spam), but precision collapses (false positives).  This would render the model ineffective in a commercial context (as it would be blocking substantial amounts of legitimate customer traffic, leading to customer churn).
- **Why this happens**.  The TF-IDF vocabulary was learned from Enron employees' writing style.  Legitimate email from the Spam Assassin and LingSpam data sets does not look like that corporate style, resulting in the model treating it as more spam-like.  The model appears to have learned "not corporate is more like spam", not what spam looks like in the general sense.
- **Verified this isn't a data-quality artifact**.  Spam Assassin's raw text was not prefixed with `Subject:`, which an earlier version of the cleaning step didn't account for (silently dropping the first line of every message).  After fixing that bug, the metrics didn't substantially change (precision `0.335` either way), confirming the precision collapse is a genuine generalization gap.
- **Positive ROC-AUC**.  ROC-AUC stays reasonably high even when precision collapses; suggesting that the model's underlying ranking is fairly sound, but needs threshold tuning (and possibly a more diverse training set).
- **Why this matters**.  An email service provider with many customers would observe the same failure mode in practice - a filter trained on a subset of customer's "normal" writing styles may misclassify (and block) too much of a customer's legitimate traffic.