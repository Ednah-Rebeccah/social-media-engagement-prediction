# Data dictionary

One row per **published** post. Save as CSV (UTF-8). Column names must match exactly, or be changed in `src/engagement_model/config.py`.

## Required columns

| Column | Type | Known before posting? | Example | Notes |
| --- | --- | --- | --- | --- |
| `post_id` | text | Identifier only | `FB-2025-0412` | Unique per post |
| `published_at` | date-time | Yes | `2025-04-12 10:30` | Local time, consistent across the file |
| `platform` | category | Yes | `facebook` | Labels are lower-cased automatically |
| `account` | category | Yes | `main_page` | Which page or handle published it |
| `content_format` | category | Yes | `carousel` | e.g. image, carousel, reel, video, text, link |
| `topic` | category | Yes | `grant_call` | Your team's own label, assigned when planning |
| `paid` | category | Yes | `organic` | `paid` only if the decision to boost is made **before** posting |
| `followers_at_post` | number | Yes | `18450` | Audience size on the morning of publication |
| `caption` | text | Yes | `Applications are open...` | Used for simple text features |
| `engagement_48h` | number | **No: this is the outcome** | `87` | Reactions + comments + shares + saves within 48 hours |

## Columns you may have but must NEVER use as predictors

These are measured after publication. They can stay in the file; the pipeline ignores them as inputs and refuses to run if one is added to the features.

`lifetime_reach`, `lifetime_impressions`, `lifetime_engagement`, `reach_48h`, `impressions_48h`, `video_views`, `first_hour_reach`, `first_hour_engagement`, `likes`, `comments`, `shares`, `saves`

The full list lives in `config.FORBIDDEN_PREDICTORS`.

## Planned posts

The planned-posts CSV used by `scripts/predict.py` has the same columns **except** `engagement_48h`.

## Data you should NOT collect

Names, profile details or messages of individual followers or commenters. They are not needed for engagement prediction, and leaving them out reduces legal and ethical risk.

## Tips

- Record the outcome at a **fixed** time after posting (48 hours). Lifetime totals favour older posts.
- Keep definitions stable. If "engagement" changed definition midway, note the date and consider analysing the periods separately.
- Fill in `topic` and `content_format` when planning, not afterwards from memory.
