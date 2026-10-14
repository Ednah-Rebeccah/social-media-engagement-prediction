"""
Generate a SYNTHETIC dataset so the pipeline can be tried end to end.

The numbers are invented. They are built to behave roughly like real
organisational accounts (format, topic, timing and audience size all
matter; there is plenty of noise; a few posts "go viral" by chance) so
that every step of the study can be demonstrated. Do not quote any
result produced from this file as evidence about real audiences.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --posts 900 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PLATFORMS = {
    # platform: (base log-engagement offset, formats available)
    "facebook": (0.0, ["image", "carousel", "video", "link", "text"]),
    "instagram": (0.3, ["image", "carousel", "reel"]),
    "linkedin": (-0.2, ["image", "carousel", "video", "link", "text"]),
    "x": (-0.7, ["image", "video", "link", "text"]),
}
ACCOUNTS = {"main_page": 1.0, "programme_page": 0.35}  # audience size multiplier
FORMAT_EFFECT = {"image": 0.0, "carousel": 0.25, "reel": 0.35, "video": 0.15, "link": -0.35, "text": -0.2}
TOPIC_EFFECT = {
    "grant_call": 0.45, "job_opportunity": 0.4, "partner_story": 0.1, "event": 0.05,
    "celebration": 0.3, "research": -0.15, "policy": -0.25,
}
CAPTIONS = {
    "grant_call": "Applications are now open for our {year} grants for community organisations.",
    "job_opportunity": "We are hiring! Join our team as a programme officer.",
    "partner_story": "Meet one of our partners and hear how their work is changing lives in {place}.",
    "event": "Highlights from this week's convening in {place} with civil society leaders.",
    "celebration": "Today we celebrate {years} years of working alongside communities.",
    "research": "New research: what we learned about local philanthropy in East Africa.",
    "policy": "Our statement on proposed changes to the regulatory framework for NGOs.",
}
PLACES = ["Kampala", "Gulu", "Mbarara", "Nairobi", "Arua", "Jinja"]


def make_posts(n_posts: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-06-30")
    days = rng.integers(0, (end - start).days, n_posts)
    hours = rng.choice(np.arange(7, 22), n_posts, p=None)
    minutes = rng.choice([0, 15, 30, 45], n_posts)
    published = start + pd.to_timedelta(days, "D") + pd.to_timedelta(hours, "h") + pd.to_timedelta(minutes, "m")

    rows = []
    for i, ts in enumerate(sorted(published)):
        platform = rng.choice(list(PLATFORMS), p=[0.35, 0.3, 0.2, 0.15])
        offset, formats = PLATFORMS[platform]
        account = rng.choice(list(ACCOUNTS), p=[0.75, 0.25])
        fmt = rng.choice(formats)
        topic = rng.choice(list(TOPIC_EFFECT))
        paid = "paid" if rng.random() < 0.08 else "organic"

        # Audiences grow over time.
        months = (ts - start).days / 30.4
        followers = int(12000 * ACCOUNTS[account] * (1 + 0.025 * months) * rng.uniform(0.95, 1.05))
        if platform == "x":
            followers = int(followers * 0.6)

        caption = CAPTIONS[topic].format(year=ts.year + 1, place=rng.choice(PLACES), years=rng.integers(5, 12))
        has_q = rng.random() < 0.3
        has_cta = rng.random() < 0.5
        if has_q:
            caption += " What do you think?"
        if has_cta:
            caption += " " + rng.choice(["Apply via the link in bio.", "Read more on our website.", "Share with your network."])
        n_tags = int(rng.integers(0, 6))
        caption += " " + " ".join(f"#{t}" for t in rng.choice(["CivilSociety", "Uganda", "Philanthropy", "Community", "EastAfrica", "Youth"], n_tags, replace=False))

        weekend = ts.dayofweek >= 5
        evening = 18 <= ts.hour <= 21
        log_mu = (
            0.9 * np.log10(followers) - 0.1 + offset
            + FORMAT_EFFECT[fmt] + TOPIC_EFFECT[topic]
            + (-0.2 if weekend else 0.0) + (0.12 if evening else 0.0)
            + (0.08 if has_q else 0.0) + (0.1 if has_cta else 0.0)
            + (0.5 if paid == "paid" else 0.0)
            - 0.03 * max(0, n_tags - 3)
        )
        noise = rng.normal(0, 0.55)
        viral = rng.random() < 0.02
        engagement = int(np.expm1(log_mu + noise + (1.8 if viral else 0.0)))
        reach = int(engagement * rng.uniform(15, 40))

        rows.append({
            "post_id": f"P{i + 1:04d}",
            "published_at": ts.strftime("%Y-%m-%d %H:%M"),
            "platform": platform,
            "account": account,
            "content_format": fmt,
            "topic": topic,
            "paid": paid,
            "followers_at_post": followers,
            "caption": caption.strip(),
            "engagement_48h": engagement,
            # Post-publication columns: present in real exports, NEVER predictors.
            "lifetime_reach": int(reach * rng.uniform(1.1, 1.6)),
            "lifetime_impressions": int(reach * rng.uniform(1.5, 2.5)),
        })
    return pd.DataFrame(rows)


def make_planned(seed: int) -> pd.DataFrame:
    """A week of planned posts to demonstrate forecasting."""
    return pd.DataFrame([
        {"post_id": "PLAN-01", "published_at": "2026-07-06 10:00", "platform": "instagram", "account": "main_page",
         "content_format": "carousel", "topic": "grant_call", "paid": "organic", "followers_at_post": 21500,
         "caption": "Applications are now open for our 2027 Community Media Fellowship! Are you a young storyteller? Apply via the link in bio. #Fellowship #Uganda"},
        {"post_id": "PLAN-02", "published_at": "2026-07-07 19:00", "platform": "facebook", "account": "main_page",
         "content_format": "video", "topic": "partner_story", "paid": "organic", "followers_at_post": 21900,
         "caption": "Meet one of our partners in Gulu and hear how their work is changing lives. #CivilSociety"},
        {"post_id": "PLAN-03", "published_at": "2026-07-10 08:30", "platform": "linkedin", "account": "main_page",
         "content_format": "link", "topic": "policy", "paid": "organic", "followers_at_post": 21700,
         "caption": "Our statement on proposed changes to the regulatory framework for NGOs. Read more on our website."},
        {"post_id": "PLAN-04", "published_at": "2026-07-11 16:00", "platform": "x", "account": "programme_page",
         "content_format": "image", "topic": "event", "paid": "organic", "followers_at_post": 4600,
         "caption": "Highlights from this week's convening in Kampala with civil society leaders. #EastAfrica #Community"},
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--posts", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    make_posts(args.posts, args.seed).to_csv(out / "sample_posts.csv", index=False)
    make_planned(args.seed).to_csv(out / "sample_planned_posts.csv", index=False)
    print(f"Wrote {out / 'sample_posts.csv'} ({args.posts} synthetic posts) and {out / 'sample_planned_posts.csv'}.")


if __name__ == "__main__":
    main()
