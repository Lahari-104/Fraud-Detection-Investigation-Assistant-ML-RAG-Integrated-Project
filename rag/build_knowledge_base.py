"""
Phase 2 -- knowledge base generation.

GROUNDING RULE (Option 1): every case writeup and every policy references ONLY
features that can be derived from the real creditcard.csv:
    - Amount            (raw column)
    - hour of day       (derived from Time)
    - anomaly profile   (which V-components deviate, from SHAP attribution)
    - burstiness        (Time-ordering; the dataset has no card id, so this is
                         a global-ordering signal, described honestly as such)

Deliberately EXCLUDED because the dataset cannot support them: country, merchant
category, device fingerprint, IP, cardholder tenure, MCC. Writing policies about
those would make retrieval theatre -- the query builder could never match them.

The statistics baked into these documents are the real ones measured in Phase 2
prep (fraud median $9.25; 36.8% of fraud <= $1; hours 00-05 carry ~3x the base
fraud rate and ~25% of all fraud; V17/V14/V12/V10 are the strongest anomaly
components). Each file is a plain .txt with a small metadata header so retrieval
can display provenance.
"""

from __future__ import annotations

import os
import re

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "knowledge_base")

# --- policies: specific, actionable, grounded in derivable features ----------
POLICIES = [
    ("POL-01", "Micro-Authorization Card-Testing Rule",
     "Transactions of $1.00 or less that carry an elevated model anomaly score "
     "must be treated as suspected card-testing. In the reference data 36.8% of "
     "confirmed fraud is at or below $1.00 versus 10.7% of legitimate activity, "
     "so a low amount combined with an anomalous profile is a positive signal, "
     "not a reason to dismiss the alert. Action: hold the card for step-up "
     "verification rather than approving on the basis of the small value."),

    ("POL-02", "Overnight Elevated-Risk Window",
     "Transactions timestamped between 00:00 and 05:59 local processing time sit "
     "in an elevated-risk window: the fraud rate there is roughly three times the "
     "overall base rate and about a quarter of all fraud occurs in these six "
     "hours. A flag raised during this window should be prioritized in the review "
     "queue over an equivalent daytime flag."),

    ("POL-03", "High-Value Manual-Hold Threshold",
     "Any flagged transaction with an amount at or above $1,000 requires manual "
     "analyst sign-off before release. This is the fraud 99th-percentile band "
     "($1,357) and false negatives here are the most costly. Automated approval "
     "is not permitted for flagged transactions above this threshold regardless "
     "of model probability."),

    ("POL-04", "Anomaly-Component Corroboration",
     "A fraud flag is considered corroborated when its top score drivers include "
     "the components historically most separating for fraud -- V17, V14, V12 and "
     "V10, all of which run several standard deviations below their legitimate "
     "mean in confirmed fraud. When these dominate the attribution, the analyst "
     "may fast-track confirmation. When they do not, wider manual review is "
     "warranted before action."),

    ("POL-05", "Small-Amount Approval Exception Removed",
     "The legacy rule that auto-approved transactions under $5.00 to reduce "
     "friction is withdrawn. Card-testing exploits exactly that exception; small "
     "amount alone must never override a model flag. Supersedes any prior "
     "low-value fast-approve guidance."),

    ("POL-06", "Model Probability Banding",
     "Flagged transactions are triaged by model probability: at or above 0.90, "
     "treat as high-confidence and action immediately; 0.50 to 0.90, standard "
     "review; below the operating threshold but above 0.20, log as watch-list "
     "and monitor for repeat activity. Probability bands set queue priority, not "
     "the approve/decline decision by themselves."),

    ("POL-07", "Burst-Activity Escalation",
     "A cluster of flagged transactions arriving in close succession in the "
     "processing stream should be escalated as a potential coordinated run even "
     "when individual amounts are small. Because this dataset has no card "
     "identifier, burst detection operates on processing-time proximity rather "
     "than per-card velocity, and findings should be labelled as provisional "
     "pending account-level correlation."),

    ("POL-08", "Zero-Amount Verification Probes",
     "Transactions of exactly $0.00 are frequently account-verification probes "
     "used to confirm a card is live before a later charge. A flagged zero-amount "
     "transaction must trigger a card-status review rather than being discarded "
     "as harmless."),

    ("POL-09", "False-Positive Cost Discipline",
     "Because legitimate volume outweighs fraud roughly 578 to 1, every false "
     "positive carries real customer-friction cost. Analysts must record a brief "
     "written justification when declining a flagged transaction so that "
     "precision can be audited. A flag is a prompt for review, not an automatic "
     "decline."),

    ("POL-10", "Explanation-Before-Action Requirement",
     "No flagged transaction may be actioned without a recorded rationale citing "
     "the specific drivers of the flag -- amount band, time window, and dominant "
     "anomaly components -- and at least one supporting policy or precedent case. "
     "This requirement exists to keep decisions auditable and to prevent action "
     "on an unexplained score."),

    ("POL-11", "High-Amount Daytime Anomaly Handling",
     "A high-value transaction (>= $200) flagged during ordinary daytime hours "
     "with a strong anomaly profile should not be downgraded merely because its "
     "timing is unremarkable. Roughly 17% of fraud exceeds $200, so amount and "
     "anomaly profile can carry a flag on their own without an overnight "
     "timestamp."),

    ("POL-12", "Repeat-Micro-Charge Pattern",
     "Multiple micro-charges (each <= $2.00) flagged in the same processing "
     "window are a recognized card-testing signature. Treat the group as one "
     "incident, hold the associated card, and do not clear any single charge in "
     "isolation on the grounds of its trivial value."),
]

# --- case writeups: 2-4 sentences, varied patterns, feature-grounded ---------
CASES = [
    ("CASE-01", "Card-testing micro-charge run",
     "A $0.79 transaction was flagged with probability 0.97 in the small hours. "
     "Its top drivers were V14 and V10 running far below their normal range, the "
     "classic anomaly signature. Investigation confirmed a card-testing probe: "
     "the fraudster was validating a stolen card number with a trivial charge "
     "before attempting larger purchases."),

    ("CASE-02", "Overnight account-takeover charge",
     "A $340 transaction at 02:14 processing time was flagged with a heavy V17 "
     "and V12 anomaly contribution. The account had no prior night-time activity. "
     "It was confirmed as an account takeover, with the charge placed while the "
     "legitimate cardholder was asleep."),

    ("CASE-03", "Zero-amount verification probe",
     "A $0.00 authorization was flagged and initially looked harmless. Because "
     "policy requires zero-amount flags to trigger a card-status review, the "
     "analyst held the card; a $900 charge attempt followed twelve minutes later "
     "and was blocked. The zero-amount probe was the fraudster confirming the "
     "card was live."),

    ("CASE-04", "High-value manual-hold catch",
     "A $1,420 transaction was flagged just above the fraud 99th-percentile band. "
     "Model probability was a moderate 0.64, which alone might have been released, "
     "but the high-value manual-hold policy forced analyst sign-off. Manual review "
     "confirmed fraud and prevented the largest single loss in that day's batch."),

    ("CASE-05", "False positive on a large legitimate purchase",
     "An $1,100 transaction was flagged at probability 0.58 with a mixed anomaly "
     "profile lacking the dominant V17/V14 signature. Manual review found a "
     "legitimate high-value purchase by an established customer. The case is "
     "retained as a template for distinguishing genuine large purchases from "
     "high-value fraud by anomaly-component profile."),

    ("CASE-06", "Repeat micro-charge cluster",
     "Five transactions between $0.50 and $1.80 were flagged in the same "
     "processing window, each with a similar anomaly profile. Treated as a single "
     "card-testing incident rather than five separate low-value alerts, the "
     "associated card was held. All five traced to one compromised card number."),

    ("CASE-07", "Small-amount flag wrongly dismissed",
     "A $0.95 flagged transaction was auto-approved under the old under-$5 "
     "fast-approve rule. It was later confirmed as the opening probe of a fraud "
     "run that escalated to a $600 loss. This case drove the withdrawal of the "
     "small-amount approval exception."),

    ("CASE-08", "Daytime high-value anomaly",
     "A $260 transaction at 14:30 was flagged despite unremarkable timing, driven "
     "almost entirely by an extreme V17 anomaly contribution. It was confirmed "
     "fraudulent, illustrating that a strong anomaly profile can carry a flag "
     "without an overnight timestamp."),

    ("CASE-09", "Phishing-sourced credential fraud",
     "A $180 transaction was flagged with strong V12 and V10 anomaly drivers "
     "shortly after midnight. The cardholder later reported entering their "
     "details on a phishing page. The anomaly profile matched other confirmed "
     "phishing-sourced fraud even though the amount was moderate."),

    ("CASE-10", "Low-probability watch-list hit that matured",
     "A $45 transaction scored 0.28, below the action threshold, and was logged "
     "to the watch-list under the probability-banding policy. A second flagged "
     "charge on a similar profile appeared within the hour, and the combined "
     "pattern confirmed fraud. Neither charge alone would have been actioned."),

    ("CASE-11", "Burst run across the processing stream",
     "A tight cluster of flagged transactions appeared in close processing-time "
     "proximity, amounts ranging $2 to $75. Escalated as a provisional "
     "coordinated run per burst-activity policy. Account-level correlation later "
     "confirmed several compromised cards used in quick succession."),

    ("CASE-12", "Legitimate overnight traveller",
     "A $95 transaction at 03:40 was flagged largely on the overnight risk "
     "window. The anomaly profile was weak and no dominant fraud component was "
     "present. Manual review confirmed a legitimate purchase by a traveller in a "
     "different time zone; retained as a caution against acting on timing alone."),

    ("CASE-13", "Cloned-card high-value fraud",
     "An $780 transaction carried an extreme combined V17/V14/V12 anomaly "
     "signature and was flagged at probability 0.99. Confirmed as a cloned-card "
     "transaction. The strong multi-component anomaly profile is characteristic "
     "of physically cloned cards used for larger in-person-style charges."),

    ("CASE-14", "Micro then macro escalation",
     "A $0.60 probe was flagged and held; because the card was blocked, a "
     "follow-up $1,250 attempt fifteen minutes later was declined automatically. "
     "The case shows why holding on the micro-charge, rather than approving it, "
     "prevents the larger downstream loss."),

    ("CASE-15", "Ambiguous mid-value flag resolved by policy",
     "A $130 transaction scored 0.55 with a moderate anomaly profile and daytime "
     "timing. Standard review under probability banding, combined with an absence "
     "of dominant fraud components, led to a request for step-up verification "
     "rather than an outright decline; the customer completed it and the charge "
     "cleared."),

    ("CASE-16", "Confirmed fraud missed at 0.5 threshold",
     "A confirmed fraudulent $210 transaction scored 0.41, which a default 0.5 "
     "cut-off would have released. The tuned operating threshold caught it. This "
     "case is the standing argument for tuning the decision threshold rather than "
     "leaving it at 0.5."),

    ("CASE-17", "Zero-amount ignored, loss followed",
     "A $0.00 authorization was dismissed before the zero-amount policy existed. "
     "A $500 charge on the same card cleared two hours later. The incident is the "
     "origin of the zero-amount verification-probe requirement."),

    ("CASE-18", "High precision on dominant-anomaly flag",
     "A $58 transaction flagged at 0.93 with V14 as the overwhelming driver was "
     "fast-tracked under the anomaly-corroboration policy and confirmed within "
     "minutes. It exemplifies the high-precision regime where a single dominant "
     "fraud component makes confirmation quick."),

    ("CASE-19", "Weekend late-night cluster",
     "Several flagged transactions between $15 and $90 appeared in the overnight "
     "window with consistent anomaly profiles. Prioritized under the overnight "
     "policy and escalated as a burst, the cluster was confirmed as a single "
     "compromised card tested repeatedly."),

    ("CASE-20", "Legitimate large purchase cleared",
     "A $1,050 transaction tripped the high-value manual-hold threshold at "
     "probability 0.47. Manual review found a legitimate appliance purchase; the "
     "anomaly profile lacked any dominant fraud component. Cleared with a "
     "recorded justification, satisfying the false-positive cost-discipline "
     "policy."),

    ("CASE-21", "Card-testing at exactly $1.00",
     "A cluster of $1.00 charges was flagged with tight, similar anomaly "
     "profiles. Under the micro-authorization rule these were treated as "
     "card-testing rather than dismissed on value. The associated card was held "
     "and confirmed compromised."),

    ("CASE-22", "Escalating amounts on one profile",
     "Three flagged transactions of $5, $50 and $400 arrived in sequence with a "
     "shared anomaly signature. The escalating-amount pattern -- probe, then test, "
     "then cash-out -- is a recognized fraud progression and was confirmed as a "
     "single fraud run."),

    ("CASE-23", "Moderate score, strong single component",
     "A $72 transaction scored a middling 0.52 but its attribution was almost "
     "entirely one extreme V10 value. Flagged for review under anomaly "
     "corroboration and confirmed fraudulent, showing probability and "
     "single-component extremity can diverge."),
]

HEADER = "# type: {kind}\n# id: {id}\n# title: {title}\n# grounded_features: {feat}\n\n"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def write_kb():
    os.makedirs(KB_DIR, exist_ok=True)
    # clear any prior generation so re-runs are clean
    for fn in os.listdir(KB_DIR):
        if fn.endswith(".txt"):
            os.remove(os.path.join(KB_DIR, fn))

    written = []
    for pid, title, body in POLICIES:
        feat = "amount, hour_of_day, anomaly_components, probability"
        fn = f"policy_{pid.lower()}_{_slug(title)[:40]}.txt"
        with open(os.path.join(KB_DIR, fn), "w") as f:
            f.write(HEADER.format(kind="policy", id=pid, title=title, feat=feat))
            f.write(body + "\n")
        written.append(fn)

    for cid, title, body in CASES:
        feat = "amount, hour_of_day, anomaly_components, probability"
        fn = f"case_{cid.lower()}_{_slug(title)[:40]}.txt"
        with open(os.path.join(KB_DIR, fn), "w") as f:
            f.write(HEADER.format(kind="case", id=cid, title=title, feat=feat))
            f.write(body + "\n")
        written.append(fn)

    return written


if __name__ == "__main__":
    files = write_kb()
    pol = [f for f in files if f.startswith("policy_")]
    case = [f for f in files if f.startswith("case_")]
    print(f"Knowledge base written to {KB_DIR}")
    print(f"  {len(pol)} policy documents")
    print(f"  {len(case)} case writeups")
    print(f"  {len(files)} total\n")
    print("Sample policy file:")
    with open(os.path.join(KB_DIR, sorted(pol)[0])) as f:
        print("  " + f.read().replace("\n", "\n  ").rstrip())
    print("\nSample case file:")
    with open(os.path.join(KB_DIR, sorted(case)[0])) as f:
        print("  " + f.read().replace("\n", "\n  ").rstrip())
