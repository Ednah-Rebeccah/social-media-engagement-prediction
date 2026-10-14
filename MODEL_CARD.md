# Model card

A short, plain-language record kept beside the model (paper, Section 10.5). Fill it in after each study run and update it at every retraining.

| Field | What to record | Your entry |
| --- | --- | --- |
| Purpose | The planning decision the model supports | |
| Outcome predicted | Exact metric and time window | Engagements within 48 hours |
| Inputs used | Every predictor, each confirmed as known before posting | See `features.FEATURE_COLUMNS` |
| Data and period | Source, accounts, date range, number of posts | |
| How it was tested | Split method, baseline, results | Chronological 80/20 + rolling origin; see `outputs/REPORT.md` |
| Gate result | Did a model beat the best baseline by at least 5%? | |
| Known weaknesses | Groups with the largest errors or bias | See `outputs/error_by_group.csv` |
| Not suitable for | Uses the model must not be put to | Deciding whether strategically important content is published; judging staff performance; predicting totals |
| Owner | Who maintains it | |
| Reviewer | Who challenges it | |
| Last trained | Date | |
| Next review | Date (at least quarterly) | |

## Decision log

| Date | Decision | Why |
| --- | --- | --- |
| | | |
