"""
Central configuration for the engagement prediction study.

Every decision the paper asks a team to make explicitly lives here:
what is predicted, over what window, which columns may be used as
predictors, and which columns must never be used.

Edit this file to match your own data export. Nothing else in the
package should need to change.
"""

# ---------------------------------------------------------------------
# 1. The outcome (paper, Section 5.1)
# ---------------------------------------------------------------------
# The single number the model predicts for each post. It must be
# measured over a FIXED window after publication so that old and new
# posts are compared fairly.
TARGET = "engagement_48h"
OUTCOME_WINDOW_HOURS = 48       # must match how TARGET was measured

# ---------------------------------------------------------------------
# 2. Identifiers and time (not predictors)
# ---------------------------------------------------------------------
POST_ID = "post_id"
PUBLISHED_AT = "published_at"   # ISO datetime, e.g. 2025-04-12 10:30
ACCOUNT = "account"             # which page or handle published the post

# ---------------------------------------------------------------------
# 3. Raw columns known BEFORE publication (paper, Section 5.2)
# ---------------------------------------------------------------------
# These are the inputs a communications officer can fill in while
# planning a post. Feature engineering turns them into model features.
CATEGORICAL_INPUTS = [
    "platform",        # Facebook, Instagram, LinkedIn, X ...
    "account",
    "content_format",  # image, carousel, reel, video, text, link
    "topic",           # team label: grant_call, partner_story, event ...
    "paid",            # "paid" or "organic", decided before posting
]

NUMERIC_INPUTS = [
    "followers_at_post",  # audience size on the morning of publication
]

TEXT_INPUT = "caption"

# ---------------------------------------------------------------------
# 4. Columns that must NEVER be used as predictors (the leaking cup)
# ---------------------------------------------------------------------
# Anything measured after the post went live. If one of these slips
# into the feature set, the model will look brilliant in testing and
# be useless for planning. The pipeline refuses to run if any of these
# (or the target) appears among the features.
FORBIDDEN_PREDICTORS = [
    TARGET,
    "lifetime_reach",
    "lifetime_impressions",
    "lifetime_engagement",
    "reach_48h",
    "impressions_48h",
    "video_views",
    "first_hour_reach",
    "first_hour_engagement",
    "comments",
    "shares",
    "likes",
    "saves",
]

# ---------------------------------------------------------------------
# 5. History feature (paper, Section 9.2)
# ---------------------------------------------------------------------
# "The account's typical recent performance" is a strong and legitimate
# predictor ONLY if computed from posts published BEFORE the one being
# predicted. This sets how many earlier posts are used.
HISTORY_WINDOW = 20

# ---------------------------------------------------------------------
# 6. Evaluation (paper, Sections 8 and 9)
# ---------------------------------------------------------------------
TEST_FRACTION = 0.20          # most recent 20% of posts held back
ROLLING_TEST_MONTHS = 6       # rolling-origin rounds, one month each
MIN_TRAIN_POSTS = 100         # do not train on fewer posts than this

RANDOM_STATE = 42
