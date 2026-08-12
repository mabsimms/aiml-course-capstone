#### Problem Statement

Email Service Providers (such as SendGrid, Mailchimp) need to maintain and protect the trust of the email channel; can the subject and body of an outbound message be scanned and used to detect and block spam before delivery?

An email service provider's traffic is inherently "open" in that customers (or potentially customers of their customers) send mail that is potentially spam.  A spam filter trained narrowly on a focused data set or singular writing style risks two failure modes on emails that don't resemble the training set - missing spam that doesn't resemble the training examples, or misclassifying legitmate email as spam.

This project asks whether the subject and body text of an outbound email can be used to reliably screen for spam against this business context.

**Goals**:
- Build a binary spam classifier from email subject/body text
- Evaluate for production readiness, weighing quality against inference cost

**Challenges**:
- The open-world nature of email service provider traffic means the model will always see senders and styles absent from training
- False positives are costlier than false negatives

**Potential benefits**:
- Protects channel trust (and ability to operate) without overly blocking legitmate customer email
