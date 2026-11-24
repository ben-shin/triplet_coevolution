"""Bounded local processes and verified, atomically published search-result directories."""

import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from tempfile import TemporaryDirectory

from triplet_coevolution.io.profiles import check_profile, read_sequences


def checksum(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def copy_model(source, target):
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    with opener(source, "rb") as inp, target.open("wb") as out:
        shutil.copyfileobj(inp, out)


def _hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tools_for(comp):
    back = comp["backend"]
    programs = {
        "jackhmmer": ["jackhmmer"],
        "hmmer": ["hmmsearch" if comp["molecule"] == "protein" else "nhmmer"],
        "infernal": ["cmsearch"],
        "psiblast": ["makeblastdb", "psiblast"],
        "reference": [],
        "import": [],
    }[back]
    if comp["alignment_backend"] in ("hmmalign", "cmalign"):
        programs.append(comp["alignment_backend"])
    out = []
    for name in programs:
        path = shutil.which(name)
        if not path:
            raise ValueError(f"missing search executable {name!r}")
        flag = "-version" if name in ("makeblastdb", "psiblast") else "-h"
        r = subprocess.run(
            [path, flag],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=15,
            check=False,
            env={**os.environ, "BLAST_USAGE_REPORT": "false"},
            text=True,
            errors="replace",
        )
        pat = rf"^{name}:\s+(\S+)" if flag == "-version" else r"^# (?:HMMER|INFERNAL)\s+(\S+)"
        version = re.search(pat, r.stdout, re.MULTILINE)
        if r.returncode or not version:
            raise ValueError(f"unable to verify the {name} version banner")
        out.append(
            {
                "name": name,
                "path": str(Path(path).absolute()),
                "version": version.group(1),
                "sha256": checksum(path),
            }
        )
    return out


def run_tool(args, work, comp, programs, log, stdout=None):
    program = next(p for p in programs if p["name"] == args[0])
    args = [str(x) for x in args]
    cmd = [program["path"], *args[1:]]
    limit = comp.get("memory_mb")
    if limit is not None:
        if os.name != "posix":
            raise ValueError("hard external-tool memory limits need POSIX")
        script = "import os,resource,sys; n=int(sys.argv[1]); resource.setrlimit(resource.RLIMIT_AS,(n,n)); os.execv(sys.argv[2],sys.argv[2:])"
        cmd = [sys.executable, "-c", script, str(limit * 1024**2), *cmd]
    idx = len(log) + 1
    stderr = work / f"command-{idx}.stderr"
    out = stdout or f"command-{idx}.stdout"
    log.append(
        {
            "program": args[0],
            "argv": args,
            "stderr": stderr.name,
            "stdout": out,
            "timeout_seconds": comp["timeout_seconds"],
            "memory_mb": limit,
        }
    )
    with (
        stderr.open("wb") as err,
        (work / out).open("wb") as dest,
        subprocess.Popen(
            cmd,
            cwd=work,
            stdin=subprocess.DEVNULL,
            stdout=dest,
            stderr=err,
            start_new_session=os.name == "posix",
            env={
                **os.environ,
                "BLAST_USAGE_REPORT": "false",
                "NCBI_CONFIG__BLAST__DATA_LOADERS": "none",
            },
        ) as process,
    ):
        try:
            try:
                status = process.wait(timeout=comp["timeout_seconds"])
            except BaseException:
                if process.poll() is None:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                raise
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"{args[0]} exceeded timeout_seconds") from exc
    if status:
        with stderr.open("rb") as f:
            f.seek(max(0, stderr.stat().st_size - 1600))
            detail = f.read().decode(errors="replace")
        raise ValueError(f"{args[0]} exited with {status}: {detail.strip()}")


def _run_search(comp, work, programs, logs, ref_only=False, targets=0):
    back = comp["backend"]
    db = "probe.fa" if ref_only else "database.fa"
    suffix = "reference" if ref_only else "hits"
    table = f"{suffix}.tbl"
    cpu = comp["cpu"]
    ev = comp["evalue"]
    if back == "jackhmmer":
        args = [
            "jackhmmer",
            "--cpu",
            cpu,
            "--seed",
            comp["seed"],
            "-N",
            comp["iterations"],
            "-E",
            ev,
            "--domE",
            comp["domain_evalue"],
            "--incE",
            comp["include_evalue"],
            "--incdomE",
            comp["include_domain_evalue"],
            "--chkhmm",
            "round",
            "--chkali",
            "round",
            "--domtblout",
            table,
            "--tblout",
            "sequences.tbl",
            "-o",
            "rounds.txt",
            "-A",
            "final.sto",
            "--qformat",
            "fasta",
            "--tformat",
            "fasta",
            "seed.fa",
            db,
        ]
        fmt = "hmmsearch_domtblout"
    elif back in ("hmmer", "infernal"):
        tool = (
            "cmsearch"
            if back == "infernal"
            else "hmmsearch"
            if comp["molecule"] == "protein"
            else "nhmmer"
        )
        args = [tool, "--cpu", cpu, "--noali", "-E", ev]
        if tool == "hmmsearch":
            args.extend(
                ["--seed", comp["seed"], "--domE", comp["domain_evalue"], "--domtblout", table]
            )
            fmt = "hmmsearch_domtblout"
        else:
            if tool == "nhmmer":
                args.extend(["--seed", comp["seed"]])
            args.extend(["--tblout", table])
            fmt = "cmsearch_tblout" if tool == "cmsearch" else "nhmmer_tblout"
        args.extend(["model", db])
    elif back == "psiblast":
        idx = f"db-{suffix}"
        run_tool(
            ["makeblastdb", "-in", db, "-dbtype", "prot", "-parse_seqids", "-out", idx],
            work,
            comp,
            programs,
            logs,
        )
        args = [
            "psiblast",
            *(["-in_pssm", "model"] if "profile" in comp else ["-query", "seed.fa"]),
            "-db",
            idx,
            "-num_iterations",
            comp["iterations"],
            "-num_threads",
            cpu,
            "-evalue",
            ev,
            "-max_target_seqs",
            max(1, targets),
            "-outfmt",
            "7 sseqid sstart send bitscore evalue qstart qend",
            "-out",
            table,
        ]
        fmt = "psiblast7"
    else:
        raise ValueError("unknown external search")
    try:
        run_tool(args, work, comp, programs, logs)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{args[0]} exceeded timeout_seconds") from exc
    if back == "psiblast":
        original = work / f"{suffix}.all-iterations.tsv"
        (work / table).rename(original)
        with original.open() as f, (work / table).open("w+") as dest:
            for line in f:
                if line.startswith("# Iteration:"):
                    dest.seek(0)
                    dest.truncate()
                dest.write(line)
    return table, fmt


def _build(raw, comp, reference, programs):
    raw.mkdir()
    logs = []
    back = comp["backend"]
    if back == "import":
        table = "import.tbl" + (".gz" if comp["hit_table"].suffix == ".gz" else "")
        shutil.copyfile(comp["hit_table"], raw / table)
        ref = None
        if "reference_hits" in comp:
            ref = "reference.tbl" + (".gz" if comp["reference_hits"].suffix == ".gz" else "")
            shutil.copyfile(comp["reference_hits"], raw / ref)
        return {
            "table": table,
            "format": comp["hit_format"],
            "reference_table": ref,
            "commands": [],
            "aliases": False,
            "iterations": [],
        }
    if "profile" in comp:
        check_profile(comp)
        copy_model(comp["profile"], raw / "model")
    (raw / "seed.fa").write_text(">seed\n" + reference + "\n", encoding="ascii")
    count = 0
    with (
        (raw / "database.fa").open("w", encoding="ascii") as db,
        (raw / "aliases.tsv").open("w", encoding="utf-8") as mapping,
    ):
        for count, (seq_id, seq) in enumerate(
            read_sequences(comp["database"], comp["molecule"]), 1
        ):
            name = f"s{count:09d}"
            db.write(f">{name}\n{seq}\n")
            mapping.write(f"{name}\t{seq_id}\t{len(seq)}\n")
    table, fmt = _run_search(comp, raw, programs, logs, targets=count)
    ref = None
    if "profile" in comp:
        (raw / "probe.fa").write_text(">seed\n" + reference + "\n", encoding="ascii")
        ref, _ = _run_search(comp, raw, programs, logs, True, 1)
    rounds = []
    converged = None
    if back == "jackhmmer":
        converged = False
        ids = sorted(int(p.stem.split("-")[-1]) for p in raw.glob("round-*.sto"))
        if not ids or ids != list(range(1, len(ids) + 1)) or len(ids) > comp["iterations"]:
            raise ValueError("incomplete JACKHMMER round checkpoints")

        def numbered(lines):
            current = 1
            for line in lines:
                if line.startswith("@@ Round:"):
                    current = int(line.split()[-1])
                yield current, line

        with (raw / "rounds.txt").open(errors="replace") as lines:
            for i, section in groupby(numbered(lines), key=itemgetter(0)):
                if i != len(rounds) + 1 or i not in ids:
                    raise ValueError("JACKHMMER checkpoints disagree with per-round report")
                sto, hmm, report = (f"round-{i}.{ext}" for ext in ("sto", "hmm", "txt"))
                if not (raw / hmm).is_file():
                    raise ValueError(f"missing JACKHMMER model checkpoint for iteration {i}")
                with (raw / report).open("w", encoding="utf-8") as dest:
                    for _, line in section:
                        dest.write(line)
                        converged |= line.startswith("@@ CONVERGED")
                rounds.append({"iteration": i, "stockholm": sto, "model": hmm, "report": report})
        if len(rounds) != len(ids):
            raise ValueError("JACKHMMER checkpoint count differs from report")
        if not (raw / "final.sto").is_file():
            raise ValueError("missing final JACKHMMER Stockholm")
    # Raw reports and aliases suffice for replay. Temporary normalized input and indexes are excluded.
    for p in tuple(raw.iterdir()):
        if p.name in ("model", "seed.fa", "database.fa", "probe.fa") or p.name.startswith(
            ("db-reference.", "db-hits.")
        ):
            p.unlink()
    return {
        "table": table,
        "format": fmt,
        "reference_table": ref,
        "commands": logs,
        "aliases": True,
        "database_records": count,
        "converged": converged,
        "iterations": rounds,
    }


@contextmanager
def _lock(path):
    with path.open("a+b") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError(
                "search cache key has an active writer; retry after completion"
            ) from exc
        yield


def raw_search(comp, ref, programs, out, cache=None):
    keys = (
        "backend",
        "molecule",
        "iterations",
        "seed",
        "evalue",
        "include_evalue",
        "include_domain_evalue",
        "domain_evalue",
        "cpu",
        "timeout_seconds",
        "memory_mb",
        "database_release",
        "hit_format",
        "reference_id",
    )
    paths = {
        key: comp[key]
        for key in (
            "database",
            "reference",
            "profile",
            "hit_table",
            "reference_hits",
            "alignment_profile",
        )
        if key in comp
    }
    digests = {key: checksum(path) for key, path in paths.items()}
    ident = {
        "schema": "triplet.raw-search.v1",
        "recipe_sha256": checksum(__file__),
        "settings": {k: comp[k] for k in keys if k in comp},
        "sources": digests,
        "programs": [{k: v for k, v in p.items() if k != "path"} for p in programs],
        "seed_sequence_sha256": hashlib.sha256(ref.encode()).hexdigest(),
    }
    key = _hash(ident)
    cache = Path(cache) if cache is not None else None
    base = cache or out
    base.mkdir(parents=True, exist_ok=True)
    final = base / (key if cache else "raw")
    lock = base / (key + ".lock")
    with _lock(lock):
        reused = final.exists()
        if not reused:
            with TemporaryDirectory(dir=base, prefix=".building-") as tmp:
                stage = Path(tmp) / "raw"
                report = _build(stage, comp, ref, programs)
                if any(checksum(path) != digests[k] for k, path in paths.items()):
                    raise ValueError("search inputs changed during execution")
                files = {
                    p.name: {"sha256": checksum(p), "bytes": p.stat().st_size}
                    for p in sorted(stage.iterdir())
                }
                record = {"identity": ident, "key": key, "result": report, "files": files}
                record["manifest_sha256"] = _hash(record)
                (stage / "manifest.json").write_text(
                    json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
                stage.rename(final)
        obj = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        digest = obj.pop("manifest_sha256", None)
        if digest != _hash(obj) or obj.get("identity") != ident or obj.get("key") != key:
            raise ValueError("search cache manifest checksum or identity mismatch")
        obj["manifest_sha256"] = digest
        if {p.name for p in final.iterdir()} != set(obj["files"]) | {"manifest.json"}:
            raise ValueError("search cache artifact inventory mismatch")
        for leaf, meta in obj["files"].items():
            p = final / leaf
            if (
                p.is_symlink()
                or not p.is_file()
                or p.stat().st_size != meta["bytes"]
                or checksum(p) != meta["sha256"]
            ):
                raise ValueError(f"search cache artifact checksum mismatch: {leaf}")
        if cache:
            shutil.copytree(final, out / "raw")
    return obj, reused
