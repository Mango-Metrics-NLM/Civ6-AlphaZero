"""Example usage of TournamentOrchestrator.

This example demonstrates how to use the tournament orchestrator to:
- Create different tournament formats
- Run tournaments with custom arena runners
- Track player statistics and ELO ratings
- Generate tournament reports
"""

import uuid
from typing import Any

from alpha_zero.tournament import (
    MatchResult,
    TournamentConfig,
    TournamentFormat,
    TournamentOrchestrator,
)


def simple_arena_runner(player_a: str, player_b: str, num_games: int) -> MatchResult:
    """Simple arena runner that simulates matches.

    Args:
        player_a: First player ID
        player_b: Second player ID
        num_games: Number of games to play

    Returns:
        MatchResult with simulated outcome.
    """
    # Simulate match (in real use, would run actual games)
    # For this example, player with lower ID wins more often
    if player_a < player_b:
        player1_score = num_games * 0.6
        player2_score = num_games * 0.4
        winner_id = player_a
    else:
        player1_score = num_games * 0.4
        player2_score = num_games * 0.6
        winner_id = player_b

    return MatchResult(
        match_id=str(uuid.uuid4())[:8],
        player1_id=player_a,
        player2_id=player_b,
        player1_score=player1_score,
        player2_score=player2_score,
        winner_id=winner_id,
        duration_seconds=num_games * 10.0,
    )


def run_round_robin_tournament() -> None:
    """Example: Run a round-robin tournament."""
    print("\n=== Round-Robin Tournament ===")

    # Create config and orchestrator
    config = TournamentConfig(games_per_match=10)
    orchestrator = TournamentOrchestrator(
        config=config,
        arena_runner=simple_arena_runner,
        games_per_match=10,
    )

    # Create players
    players = [f"player_{i}" for i in range(1, 5)]

    # Create bracket
    bracket = orchestrator.create_bracket(
        players,
        format=TournamentFormat.ROUND_ROBIN,
    )

    print(f"Players: {len(players)}")
    print(f"Rounds: {bracket.total_rounds}")
    print(f"Total matches: {bracket.total_matches}")

    # Run tournament
    result = orchestrator.run_tournament(bracket)

    # Display results
    print(f"\nWinner: {result.winner}")
    print(f"Total games: {result.total_games}")
    print(f"Duration: {result.duration_seconds:.1f}s")

    print("\nFinal Standings:")
    for standing in result.final_standings:
        print(
            f"  {standing.rank}. {standing.player_id:12s} - "
            f"{standing.wins}W/{standing.losses}L/{standing.draws}D "
            f"(Rating: {standing.elo_rating:.0f}, Points: {standing.points})"
        )


def run_single_elimination_tournament() -> None:
    """Example: Run a single-elimination tournament."""
    print("\n=== Single-Elimination Tournament ===")

    orchestrator = TournamentOrchestrator(
        arena_runner=simple_arena_runner,
        games_per_match=5,
    )

    # Create 8 players (power of 2)
    players = [f"player_{i}" for i in range(1, 9)]

    # Create bracket with seeding
    seeding = {player: i + 1 for i, player in enumerate(players)}

    bracket = orchestrator.create_bracket(
        players,
        format=TournamentFormat.SINGLE_ELIMINATION,
        seeding=seeding,
    )

    print(f"Players: {len(players)}")
    print(f"Rounds: {bracket.total_rounds}")
    print(f"Total matches: {bracket.total_matches}")

    # Run tournament
    result = orchestrator.run_tournament(bracket)

    # Display results
    print(f"\nChampion: {result.winner}")
    print(f"Total games: {result.total_games}")

    print("\nTop 4:")
    for standing in result.final_standings[:4]:
        print(
            f"  {standing.rank}. {standing.player_id:12s} - "
            f"{standing.wins}W/{standing.losses}L"
        )


def run_swiss_tournament() -> None:
    """Example: Run a Swiss-system tournament."""
    print("\n=== Swiss Tournament ===")

    orchestrator = TournamentOrchestrator(
        arena_runner=simple_arena_runner,
        games_per_match=5,
    )

    # Create players
    players = [f"player_{i}" for i in range(1, 9)]

    # Create bracket (Swiss starts with no rounds)
    bracket = orchestrator.create_bracket(
        players,
        format=TournamentFormat.SWISS,
    )

    print(f"Players: {len(players)}")

    # Run 4 Swiss rounds
    num_rounds = 4
    for round_num in range(num_rounds):
        # Generate next round
        orchestrator._generate_swiss_round(bracket)

        # Run round
        round_results = orchestrator.run_round(bracket, round_num)
        bracket.results[round_num] = round_results

        # Display standings after round
        standings = orchestrator.get_standings(bracket)
        print(f"\nAfter Round {round_num + 1}:")
        for i, standing in enumerate(standings[:3], 1):
            print(f"  {i}. {standing.player_id:12s} - Points: {standing.points}")

        bracket.current_round += 1

    # Final results
    final_standings = orchestrator.get_standings(bracket)
    print(f"\nWinner: {final_standings[0].player_id}")


def demonstrate_player_tracking() -> None:
    """Example: Track player statistics across multiple tournaments."""
    print("\n=== Player Statistics Tracking ===")

    orchestrator = TournamentOrchestrator(
        arena_runner=simple_arena_runner,
        games_per_match=5,
    )

    # Same players across multiple tournaments
    players = ["alice", "bob", "charlie", "diana"]

    # Run 3 round-robin tournaments
    for tournament_num in range(1, 4):
        print(f"\nTournament {tournament_num}:")
        bracket = orchestrator.create_bracket(
            players,
            format=TournamentFormat.ROUND_ROBIN,
        )
        result = orchestrator.run_tournament(bracket)
        print(f"  Winner: {result.winner}")

    # Check accumulated statistics
    print("\nCumulative Statistics:")
    for player_id in players:
        stats = orchestrator._get_or_create_player_stats(player_id)
        history_stats = orchestrator.match_history.get_player_statistics(player_id)

        if history_stats:
            print(
                f"  {player_id:8s}: {history_stats.total_games} games, "
                f"{history_stats.wins}W/{history_stats.losses}L, "
                f"Rating: {stats.rating:.0f}"
            )


if __name__ == "__main__":
    # Run examples
    run_round_robin_tournament()
    run_single_elimination_tournament()
    run_swiss_tournament()
    demonstrate_player_tracking()

    print("\n" + "=" * 60)
    print("Examples complete!")
