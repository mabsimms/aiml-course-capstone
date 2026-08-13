# Spam Screening for an Email Service Provider

[Full technical notebook →](./capstone-results.ipynb)

Email Service Providers (such as SendGrid, Mailchimp) need to maintain and protect the trust of the email channel; can the subject and body of an outbound message be scanned and used to detect and block spam before delivery?

An email service provider's traffic is inherently "open" in that customers (or potentially customers of their customers) send mail that is potentially spam.  A spam filter trained narrowly on a focused data set or singular writing style risks two failure modes on emails that don't resemble the training set - missing spam that doesn't resemble the training examples, or misclassifying legitimate email as spam.

This project asks whether the subject and body text of an outbound email can be used to reliably screen for spam against this business context.

**Goals**:
- Build a binary spam classifier from email subject/body text
- Evaluate for production readiness, weighing quality against inference cost

**Challenges**:
- The open-world nature of email service provider traffic means the model will always see senders and styles absent from training
- False positives are costlier than false negatives

**Potential benefits**:
- Protects channel trust (and ability to operate) without overly blocking legitimate customer email

## What We Built

This project built out a spam classifier; a system that reads the subject and body of an email and produces a score for how likely it is to be spam.  The classifier was trained and tested against ~18,600 historical emails pulled from three different sources.

Two technical approaches were evaluated; a simpler/faster statistical approach, and a more complex deep-learning approach.  Each model was evaluated for accuracy (including false positives and false negatives), and the efficiency of running each model in a production context (receiving messages over an API call to score in real-time).

## What We Found

- **The best model is somewhat reliable on writing patterns it has seen before**.  On email traffic similar to the training set the model correctly identifies about 99% of spam, and mistakenly blocks roughly 1.6% of legitimate traffic.

- **The best performing model is also the cheapest to serve**.  The simpler, faster model won on accuracy, but was also an order of magnitude cheaper to serve and required no specialized GPU hardware.

- **A filter trained on a narrow data set does not reliably protect against the broad range of potential traffic**.  Initial validation of a model trained on a single-sourced data set resulted in high (>50%) block rates on legitimate traffic from other sources.

## Is This Ready for Production?

This model is **NOT** ready for production deployment.  A 1.6% false positive rate sounds small in isolation, but at the production scale of a large Email Service Provider (billions of messages per day), that adds up to tens of millions of legitimate emails blocked every day.

**The training data set is too small, too narrow, and likely too dated**.  The project used ~18,600 messages from older public sources.  This is enough to explore and compare technical approaches, but far short of the diversity required to achieve sufficient model accuracy.  In addition to model training challenges, this also compounds the difficulty of determining how well the model might generalize to novel data sources and patterns.

**The model only reads message content**.  The training data set did not contain any message metadata, or sender context.  It does not have access to how long a customer has been active on the platform, their sending patterns, or the ability to identify any anomalies in their behavior.

## Recommendations and Next Steps

**Grow and diversify the training data**.  Invest in obtaining more high-quality labelled data sets, preferably with additional metadata (customer context, how long the sender has been active, email headers), before considering introducing this to customer traffic.

**Tune the tradeoff between missing spam and blocking legitimate email**.  Every result in this project used a default setting that treats false positives and false negatives as equally bad.  This should be revisited in light of balancing per-customer impact with the Email Service Provider's reputation/trust in the ecosystem.

