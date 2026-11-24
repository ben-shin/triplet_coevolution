"""Process-local progress routing for normal native commands and programmatic callers."""

import json
import os
import time
from contextlib import ExitStack, redirect_stdout
from contextvars import ContextVar
from datetime import UTC, datetime

import typer

quiet_mode = ContextVar("triplet_quiet_mode", default=False)

_emit = ContextVar("triplet_command_progress", default=None)


def event(name, **details):
    writer = _emit.get()
    if writer is not None:
        writer(name, details)


def configure(ctx, *, quiet=False, verbose=False, json_log=False):
    if quiet and (verbose or json_log):
        raise typer.BadParameter("--quiet conflicts with --verbose/--json-log")
    if not (quiet or verbose or json_log):
        return
    resources = ExitStack()
    ctx.call_on_close(resources.close)
    if quiet:
        resources.callback(quiet_mode.reset, quiet_mode.set(True))
        sink = resources.enter_context(open(os.devnull, "w"))  # noqa: SIM115 -- Click closes the stack at command exit.
        resources.enter_context(redirect_stdout(sink))
        return
    command = ctx.invoked_subcommand
    start = time.monotonic()

    def write(name, details):
        row = {
            "event": name,
            "command": command,
            "time_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - start,
            **details,
        }
        typer.echo(
            json.dumps(row, allow_nan=False)
            if json_log
            else f"{command}: {name} ({row['elapsed_seconds']:.3f} s)",
            err=True,
        )

    token = _emit.set(write)
    resources.callback(_emit.reset, token)
    resources.callback(event, "command_ended")
    event("command_started")
