"""Headless tests for the desktop app, driven through Textual's pilot.

The app is where a blocking call silently freezes the window rather than
raising, so these drive it the way a user would -- type, press enter, wait
for the worker -- and assert that the result actually arrived.

Everything runs in offline mode: the tests must not depend on a market
being open or on a vendor being reachable.
"""

from __future__ import annotations

from textual.widgets import TabbedContent

from research_cli.app import ResearchApp


def _status(app: ResearchApp) -> str:
    return str(app.query_one("#status").content)


async def _run_query(pilot, app: ResearchApp, query: str, limit: int = 200) -> None:
    """Type a query, submit it, and wait for the worker to finish."""
    field = app.query_one("#query")
    field.focus()
    await pilot.pause()
    field.value = query
    await pilot.press("enter")
    for _ in range(limit):
        await pilot.pause(0.05)
        if not app._busy:
            return
    raise AssertionError(f"worker did not finish for {query!r}")


def test_app_composes() -> None:
    """Constructing the app must not touch the network or the filesystem."""
    app = ResearchApp(mock=True)
    assert app.TITLE == "research"
    assert app.mock is True


def test_attribution_query_renders_and_validates() -> None:
    import asyncio

    async def scenario() -> None:
        app = ResearchApp(mock=True)
        async with app.run_test(size=(120, 45)) as pilot:
            await _run_query(pilot, app, "why did SPY fall today")
            status = _status(app)
            assert "traced to the evidence" in status
            assert "VALIDATION FAILED" not in status

    asyncio.run(scenario())


def test_snapshot_query_renders_and_validates() -> None:
    import asyncio

    async def scenario() -> None:
        app = ResearchApp(mock=True)
        async with app.run_test(size=(120, 45)) as pilot:
            await _run_query(pilot, app, "is KO good to invest")
            assert "traced to the evidence" in _status(app)
            assert "VALIDATION FAILED" not in _status(app)

    asyncio.run(scenario())


def test_empty_query_is_refused_without_crashing() -> None:
    import asyncio

    async def scenario() -> None:
        app = ResearchApp(mock=True)
        async with app.run_test(size=(120, 45)) as pilot:
            field = app.query_one("#query")
            field.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert not app._busy

    asyncio.run(scenario())


def test_clear_binding_resets_the_view() -> None:
    import asyncio

    async def scenario() -> None:
        app = ResearchApp(mock=True)
        async with app.run_test(size=(120, 45)) as pilot:
            await _run_query(pilot, app, "why did SPY fall today")
            app.query_one("#query").focus()
            await pilot.press("ctrl+l")
            await pilot.pause(0.1)
            assert app.query_one("#query").value == ""

    asyncio.run(scenario())


def test_backtest_tab_runs() -> None:
    import asyncio

    async def scenario() -> None:
        app = ResearchApp(mock=True)
        async with app.run_test(size=(120, 45)) as pilot:
            app.query_one(TabbedContent).active = "backtest-tab"
            await pilot.pause()
            app.query_one("#ticker").value = "SPY"
            await pilot.click("#run-backtest")
            for _ in range(600):
                await pilot.pause(0.05)
                if not app._busy:
                    break
            assert not app._busy, "backtest worker did not finish"

    asyncio.run(scenario())


def test_backtest_without_a_ticker_is_refused() -> None:
    import asyncio

    async def scenario() -> None:
        app = ResearchApp(mock=True)
        async with app.run_test(size=(120, 45)) as pilot:
            app.query_one(TabbedContent).active = "backtest-tab"
            await pilot.pause()
            await pilot.click("#run-backtest")
            await pilot.pause(0.1)
            assert not app._busy

    asyncio.run(scenario())
