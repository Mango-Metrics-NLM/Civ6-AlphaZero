"""Reconstruct tournament results from Cloud Logging for a cancelled job."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

from google.cloud import aiplatform, logging_v2

from alpha_zero.tournament.analysis import TournamentAnalyzer
from alpha_zero.tournament.config import MatchResult
from alpha_zero.tournament.elo import ELORatingSystem
from alpha_zero.tournament.history import MatchHistory

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_logs(project_id: str, job_id: str, start_time: str | None = None) -> list[str]:
    """Fetch logs for the given job ID, resuming on quota errors."""
    logger.info(f"Fetching logs for job {job_id} in project {project_id}...")
    client = logging_v2.Client(project=project_id)

    current_start = start_time
    log_messages = []
    last_timestamp = None

    while True:
        filter_str = f'resource.type="ml_job" AND resource.labels.job_id="{job_id}"'
        if current_start:
            filter_str += f' AND timestamp >= "{current_start}"'

        # Order by timestamp to make resumption easy
        entries = client.list_entries(filter_=filter_str, page_size=1000, order_by="timestamp asc")

        fetch_count = 0
        try:
            for entry in entries:
                payload = entry.payload
                if isinstance(payload, dict):
                    text = payload.get("message", str(payload))
                else:
                    text = str(payload)

                log_messages.append(text)
                last_timestamp = entry.timestamp
                fetch_count += 1

                if len(log_messages) % 5000 == 0:
                    logger.info(f"Fetched {len(log_messages)} entries total...")

            # If we reach here, we've exhausted this iterator
            logger.info("Reached end of logs.")
            break

        except Exception as e:
            if ("Quota exceeded" in str(e) or "429" in str(e)) and last_timestamp:
                logger.warning(
                    f"Quota exceeded. Fetched {fetch_count} in this batch. Waiting 65s..."
                )
                # Shift start_time slightly to avoid re-fetching the last entry
                # last_timestamp is a datetime object
                next_start_dt = last_timestamp + timedelta(microseconds=1)
                current_start = next_start_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                time.sleep(65)
                continue
            else:
                logger.error(f"Error fetching logs: {e}")
                break

    logger.info(f"Fetched {len(log_messages)} log entries total.")
    return log_messages


def parse_match_results(logs: list[str]) -> list[dict]:
    """Parse matchup results from logs."""
    # Pattern: Result: v000 vs v001 -> W:8 D:1 L:1 (win_rate=80.0%)
    match_pattern = re.compile(r"Result:\s+(v\d+)\s+vs\s+(v\d+)\s+->\s+W:(\d+)\s+D:(\d+)\s+L:(\d+)")

    matchups = []
    seen_match_keys = set()

    for log in logs:
        match = match_pattern.search(log)
        if match:
            player_a = match.group(1)
            player_b = match.group(2)
            wins = int(match.group(3))
            draws = int(match.group(4))
            losses = int(match.group(5))

            # Use a key to avoid duplicates if logs overlaps
            match_key = f"{player_a}_{player_b}_{wins}_{draws}_{losses}"
            if match_key not in seen_match_keys:
                matchups.append(
                    {
                        "player_a": player_a,
                        "player_b": player_b,
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                        "total_games": wins + draws + losses,
                    }
                )
                seen_match_keys.add(match_key)

    logger.info(f"Parsed {len(matchups)} unique matchups from logs.")
    return matchups


def reconstruct_tournament(matchups: list[dict], output_db: str):
    """Reconstruct tournament history and run analysis."""
    Path(output_db).parent.mkdir(parents=True, exist_ok=True)
    if Path(output_db).exists():
        Path(output_db).unlink()

    history = MatchHistory(output_db)
    elo_system = ELORatingSystem()

    for matchup in matchups:
        player_a = matchup["player_a"]
        player_b = matchup["player_b"]

        stats_a = history.get_player_statistics(player_a)
        stats_b = history.get_player_statistics(player_b)

        rating_a = stats_a.current_rating if stats_a else 1500.0
        rating_b = stats_b.current_rating if stats_b else 1500.0

        for k in range(matchup["wins"]):
            change_a, change_b = elo_system.calculate_rating_changes(rating_a, rating_b, 1.0, 0.0)
            rating_a_new = rating_a + change_a
            rating_b_new = rating_b + change_b

            res = MatchResult(
                match_id=f"{player_a}_{player_b}_w{k}_{time.time()}",
                player1_id=player_a,
                player2_id=player_b,
                player1_score=1.0,
                player2_score=0.0,
                player1_rating_before=rating_a,
                player2_rating_before=rating_b,
                player1_rating_change=change_a,
                player2_rating_change=change_b,
                winner_id=player_a,
                duration_seconds=0,
            )
            history.record_match(res, rating_a_new, rating_b_new)
            rating_a, rating_b = rating_a_new, rating_b_new

        for k in range(matchup["draws"]):
            change_a, change_b = elo_system.calculate_rating_changes(rating_a, rating_b, 0.5, 0.5)
            rating_a_new = rating_a + change_a
            rating_b_new = rating_b + change_b

            res = MatchResult(
                match_id=f"{player_a}_{player_b}_d{k}_{time.time()}",
                player1_id=player_a,
                player2_id=player_b,
                player1_score=0.5,
                player2_score=0.5,
                player1_rating_before=rating_a,
                player2_rating_before=rating_b,
                player1_rating_change=change_a,
                player2_rating_change=change_b,
                winner_id=None,
                duration_seconds=0,
            )
            history.record_match(res, rating_a_new, rating_b_new)
            rating_a, rating_b = rating_a_new, rating_b_new

        for k in range(matchup["losses"]):
            change_a, change_b = elo_system.calculate_rating_changes(rating_a, rating_b, 0.0, 1.0)
            rating_a_new = rating_a + change_a
            rating_b_new = rating_b + change_b

            res = MatchResult(
                match_id=f"{player_a}_{player_b}_l{k}_{time.time()}",
                player1_id=player_a,
                player2_id=player_b,
                player1_score=0.0,
                player2_score=1.0,
                player1_rating_before=rating_a,
                player2_rating_before=rating_b,
                player1_rating_change=change_a,
                player2_rating_change=change_b,
                winner_id=player_b,
                duration_seconds=0,
            )
            history.record_match(res, rating_a_new, rating_b_new)
            rating_a, rating_b = rating_a_new, rating_b_new

    logger.info("Tournament reconstruction complete.")
    return history


def main():
    parser = argparse.ArgumentParser(description="Reconstruct tournament from logs")
    parser.add_argument("--project", default="hrmtrain", help="GCP Project ID")
    parser.add_argument("--job_id", default="356684784686071808", help="Vertex AI Job ID")
    parser.add_argument(
        "--output", default="data/tournament_reconstructed.db", help="Output DB path"
    )
    parser.add_argument(
        "--report", default="reports/tournament_analysis.md", help="Output report path"
    )

    args = parser.parse_args()

    aiplatform.init(project=args.project)
    job = aiplatform.CustomJob.get(args.job_id)
    start_time = job.create_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info(f"Job created at {start_time}, using as filter.")

    logs = fetch_logs(args.project, args.job_id, start_time=start_time)
    matchups = parse_match_results(logs)

    if not matchups:
        logger.error("No matchups found in logs. Check job ID and project.")
        sys.exit(1)

    history = reconstruct_tournament(matchups, args.output)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    analyzer = TournamentAnalyzer(match_history=history)
    analyzer.export_report(args.report, format="markdown")
    analyzer.export_report(args.report.replace(".md", ".json"), format="json")

    logger.info(f"Analysis report generated at {args.report}")


if __name__ == "__main__":
    main()
