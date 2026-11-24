"""Accessible, fixed-size archival score and calibration graphics."""

from html import escape

import numpy as np

from triplet_coevolution.reporting.explore_data import artifact

PALETTES = {
    "accessible": (("#eff7f4", "#148279", "#123c54"), ("#315c91", "#faf7ee", "#a64216")),
    "amber-blue": (("#fff5d8", "#d8973e", "#49334b"), ("#175b70", "#faf7ee", "#953359")),
}


def colors(palette, signed):
    stops = PALETTES[palette][int(signed)]
    rgb = np.array([[int(c[i : i + 2], 16) for i in (1, 3, 5)] for c in stops])
    ramp = []
    for x in np.linspace(0, 2, 17):
        k = min(1, int(x))
        t = x - k
        ramp.append(
            "#"
            + "".join(f"{v:02x}" for v in np.rint((1 - t) * rgb[k] + t * rgb[k + 1]).astype(int))
        )
    return ramp


def val(value, digits=4):
    return "Unavailable" if value is None else f"{value:.{digits}g}"


def grid_svg(grid, *, title, units, maximum, signed, palette, identifier, overlay=None):
    data = grid["values"]
    nr, nc = data.shape
    ramp = colors(palette, signed)
    paths = [[] for _ in range(17)]
    for i, j in zip(*np.nonzero(np.isfinite(data))):
        level = data[i, j] / (maximum or 1.0)
        index = int(np.clip(round(8 * (1 + level) if signed else 16 * level), 0, 16))
        paths[index].append(f"M{j} {i}h1v1h-1z")
    bucket = "div" if signed else "seq"
    missing_id = "missing-" + identifier
    desc = (
        f"{title}. Rows and columns are retained sites grouped by original alignment column. "
        f"{grid['finite']:,} of {grid['total']:,} cells have finite values. "
        "Color shows the mean among finite, eligible cells. Hatching marks bins without a value."
    )
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 250" role="img" aria-labelledby="title-{identifier} desc-{identifier}">',
        f'<title id="title-{identifier}">{escape(title)}</title><desc id="desc-{identifier}">{escape(desc)}</desc>',
        f'<defs><pattern id="{missing_id}" patternUnits="userSpaceOnUse" width="6" height="6"><rect width="6" height="6" fill="#ecedf0"/><path d="M0 6L6 0" stroke="#91949d" stroke-width="1.2"/></pattern></defs>',
        f'<rect x="39" y="16" width="275" height="158" fill="url(#{missing_id})"/>',
        f'<g transform="translate(39,16) scale({275 / nc},{158 / nr})" shape-rendering="crispEdges">',
    ]
    for i, path in enumerate(paths):
        if path:
            pieces.append(
                f'<path fill="{ramp[i]}" style="fill:var(--{bucket}{i},{ramp[i]})" d="{"".join(path)}"/>'
            )
    pieces.append("</g>")
    if overlay is not None:
        for i, j in zip(*np.nonzero(overlay)):
            xpos = 39 + (j + 0.5) * 275 / nc
            ypos = 16 + (i + 0.5) * 158 / nr
            pieces.append(
                f'<circle cx="{xpos:.2f}" cy="{ypos:.2f}" r="2.4" fill="none" stroke="#15212a" stroke-width="1.4"/>'
            )
    pieces.append('<g fill="#25323c" font-family="sans-serif" font-size="11">')
    for x, y, s, align in (
        (39, 189, grid["y"][0], "start"),
        (314, 189, grid["y"][-1], "end"),
        (36, 24, grid["x"][0], "end"),
        (36, 173, grid["x"][-1], "end"),
    ):
        pieces.append(f'<text x="{x}" y="{y}" text-anchor="{align}">{escape(s)}</text>')
    if maximum:
        for k, color in enumerate(ramp):
            pieces.append(
                f'<rect x="{39 + 12 * k}" y="201" width="12" height="8" fill="{color}" style="fill:var(--{bucket}{k},{color})"/>'
            )
        pieces.append(
            f'<text x="39" y="224">{val(-maximum if signed else 0)}</text>'
            f'<text x="243" y="224" text-anchor="end">{val(maximum)}</text>'
        )
        if signed:
            pieces.append('<text x="141" y="224" text-anchor="middle">0</text>')
    else:
        label = "All eligible scores are zero" if grid["finite"] else "No eligible finite scores"
        pieces.append(f'<text x="39" y="209">{label}</text>')
    pieces.append(
        f'<text x="39" y="242">{escape(units)}</text><rect x="258" y="200" width="12" height="10" fill="url(#{missing_id})"/><text x="274" y="209">missing</text></g></svg>'
    )
    return "".join(pieces)


def xy_svg(
    points, *, title, x_label, y_label, xmax=1.0, ymax=1.0, line=False, diagonal=False, origin=0.0
):
    def x(val):
        return 43 + 249 * (val - origin) / max(1e-12, xmax - origin)

    def y(val):
        return 160 - 141 * (val - origin) / max(1e-12, ymax - origin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 318 202" role="img" aria-label="{escape(title, quote=True)}"><title>{escape(title)}</title>',
        '<path d="M43 19v141h249" stroke="#566573" fill="none"/>',
    ]
    if diagonal:
        parts.append('<path d="M43 160L292 19" stroke="#a7b4b5" stroke-dasharray="3 4"/>')
    if line and points:
        parts.append(
            '<polyline fill="none" stroke="#087c78" stroke-width="2.4" points="'
            + " ".join(f"{x(a):.2f},{y(b):.2f}" for a, b in points)
            + '"/>'
        )
    else:
        for a, b in points:
            parts.append(
                f'<circle cx="{x(a):.2f}" cy="{y(b):.2f}" r="2.7" fill="#17696e" fill-opacity=".7"/>'
            )
    for xx, yy, anc, text in [
        (43, 175, "middle", val(origin)),
        (292, 175, "middle", val(xmax)),
        (38, 23, "end", val(ymax)),
        (38, 160, "end", val(origin)),
        (167, 195, "middle", x_label),
    ]:
        parts.append(
            f'<text x="{xx}" y="{yy}" text-anchor="{anc}" font-family="sans-serif" font-size="11">{escape(text)}</text>'
        )
    parts.append(
        f'<text transform="translate(12 94) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="11">{escape(y_label)}</text></svg>'
    )
    return "".join(parts)


def export_maps(
    directory,
    name,
    grids,
    *,
    maximum,
    units,
    signed,
    palette,
    formats,
    overlays=None,
    threshold=None,
):
    """Write explicitly requested compact maps with the same normalization and data as HTML."""
    if not formats:
        return []
    destination = artifact(directory, "report_assets")
    destination.mkdir(exist_ok=True)
    files = []
    for pair, g in grids.items():
        title = f"{pair} · {name}"
        overlay = (overlays or {}).get(pair)
        if "svg" in formats:
            filename = f"{name}_{pair}.svg"
            svg = grid_svg(
                g,
                title=title,
                units=units,
                maximum=maximum,
                signed=signed,
                palette=palette,
                identifier="figure",
                overlay=overlay,
            )
            caption = f"{title} · original columns (rows {pair[0]}, columns {pair[1]})"
            if threshold and overlay is not None:
                caption += " · " + threshold
            svg = svg.replace('viewBox="0 0 340 250"', 'viewBox="0 -22 340 272"').replace(
                "<defs>",
                f'<text x="5" y="-5" font-size="9" font-family="sans-serif">{escape(caption)}</text><defs>',
                1,
            )
            artifact(destination, filename).write_text(svg, encoding="utf-8")
            files.append("report_assets/" + filename)
        other = [f for f in formats if f in ("png", "pdf")]
        if not other:
            continue
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
        except ImportError as exc:
            raise ValueError("PNG/PDF report exports require the plots extra") from exc
        fig, ax = plt.subplots(figsize=(5.5, 4.2), layout="constrained")
        try:
            cmap = ListedColormap(colors(palette, signed)).with_extremes(bad="#9ba3ac")
            image = ax.imshow(
                np.ma.masked_invalid(g["values"]),
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=-(maximum or 1) if signed else 0,
                vmax=maximum or 1,
            )
            if overlay is not None and np.any(overlay):
                a, b = np.nonzero(overlay)
                ax.scatter(b, a, facecolors="none", edgecolors="black", s=12, label=threshold)
                ax.legend(fontsize=7)
            ax.set_title(title)
            ax.set_xlabel(pair[1] + " original alignment column")
            ax.set_ylabel(pair[0] + " original alignment column")
            for setter, labels in [(ax.set_xticks, g["y"]), (ax.set_yticks, g["x"])]:
                positions = np.unique(
                    np.linspace(0, len(labels) - 1, min(4, len(labels)), dtype=int)
                )
                setter(positions, [labels[i] for i in positions], fontsize=8)
            if maximum:
                fig.colorbar(image, ax=ax, label=units)
            else:
                status = (
                    "All eligible scores are zero" if g["finite"] else "No eligible finite scores"
                )
                ax.set_title(f"{title}\n{status} · {units}")
            fig.text(
                0.01, 0.01, "Gray = missing · bin color = mean eligible finite score", fontsize=7
            )
            for fmt in other:
                filename = f"{name}_{pair}.{fmt}"
                fig.savefig(artifact(destination, filename), dpi=150)
                files.append("report_assets/" + filename)
        finally:
            plt.close(fig)
    return files
