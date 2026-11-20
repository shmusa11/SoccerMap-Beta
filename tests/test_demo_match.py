import math

from soccer_analyzer.workspace_analysis import DemoMatch


def test_demo_match_is_deterministic_continuous_and_has_passes():
    first, second = DemoMatch(105, 68, 60), DemoMatch(105, 68, 60)
    assert first.legs == second.legs
    # Passes between teammates and some interceptions, so the coach tools have events.
    assert len(first.legs) >= 15
    assert len(first.changes) - 1 >= 2
    previous = None
    fast = total = 0
    for step in range(600):
        players, ball = first.state(step / 10)
        assert len(players) == 23
        for _, _, _, x, y in players:
            assert -1 <= x <= 106 and -12 <= y <= 69
        if previous is not None:
            # The ball never jumps faster than a 15 m/s pass between 0.1 s samples.
            assert math.dist(ball, previous[1]) <= 1.6
            for now, before in zip(players, previous[0]):
                total += 1
                fast += math.dist(now[3:], before[3:]) > 0.95
        previous = (players, ball)
    assert fast / total < 0.01


def test_demo_goalkeepers_and_referee_use_pipeline_labels():
    players, _ = DemoMatch(105, 68, 10).state(3.0)
    kinds = {(team, kind) for _, team, kind, _, _ in players}
    assert (-3, "goalkeeper") in kinds
    assert (-2, "referee") in kinds
    assert (0, "player") in kinds and (1, "player") in kinds
