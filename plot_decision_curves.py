"""Build the vector plot for the recorded matched-FPR audit."""

import argparse
import json
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas


METHODS = (
    ("targeted", "Balanced targeting", "#146769", False),
    ("uncertain", "Uncertainty", "#9c4326", True),
)
PANELS = (
    ("absent", "A   All verified absences"),
    ("near_absent", "B   Adjacent verified absences"),
)
INK = HexColor("#17212a")
GUIDE = HexColor("#dce1e5")


def label(plot, x, y, text, size=8, *, align="left", bold=False):
    plot.setFillColor(INK)
    plot.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    draw = {"left": plot.drawString, "center": plot.drawCentredString, "right": plot.drawRightString}
    draw[align](x, y, text)


def panel(plot, x, title, results, fpr_points):
    y, width, height = 39, 132, 97
    label(plot, x, y + height + 11, title, 8.2, bold=True)
    for value in (0, 20, 40, 60, 80, 100):
        row = y + height * value / 100
        plot.setStrokeColor(GUIDE)
        plot.setLineWidth(0.35)
        plot.line(x, row, x + width, row)
        label(plot, x - 4, row - 2.5, str(value), 7, align="right")
    plot.setStrokeColor(INK)
    plot.setLineWidth(0.6)
    plot.line(x, y, x + width, y)
    for value in (5, 10, 20, 30, 40):
        col = x + width * (value - 5) / 35
        plot.line(col, y, col, y - 3)
        label(plot, col, y - 12, str(value), 7, align="center")
    for method, _, color, dashed in METHODS:
        xs = [x + width * (100 * value - 5) / 35 for value in fpr_points]
        ys = [y + height * value for value in results["methods"][method]["recall"]]
        plot.setStrokeColor(HexColor(color))
        plot.setFillColor(HexColor(color))
        plot.setLineWidth(1.3)
        plot.setDash([4, 2.5] if dashed else [])
        for left, right in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
            plot.line(*left, *right)
        plot.setDash([])
        for cx, cy in zip(xs, ys, strict=True):
            if dashed:
                plot.rect(cx - 1.6, cy - 1.6, 3.2, 3.2, stroke=0, fill=1)
            else:
                plot.circle(cx, cy, 1.85, stroke=0, fill=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit["false_positive_rates"] != [0.05, 0.1, 0.2, 0.3, 0.4]:
        raise ValueError("Expected the recorded 5 to 40 percent FPR grid")
    plot = Canvas(str(args.output), pagesize=(336, 177), invariant=True)
    plot.setTitle("Patient-macro recall at matched false-positive rates")
    for (key, title), x in zip(PANELS, (33, 193), strict=True):
        panel(plot, x, title, audit["groups"][key], audit["false_positive_rates"])
    label(plot, 174, 12, "Matched false-positive rate (%)", 8, align="center")
    plot.saveState()
    plot.translate(8, 87)
    plot.rotate(90)
    label(plot, 0, 0, "Small-slice recall (%)", 8, align="center")
    plot.restoreState()
    for (_, text, color, dashed), x in zip(METHODS, (85, 185), strict=True):
        plot.setStrokeColor(HexColor(color))
        plot.setLineWidth(1.3)
        plot.setDash([4, 2.5] if dashed else [])
        plot.line(x, 164, x + 15, 164)
        plot.setDash([])
        label(plot, x + 19, 161, text, 8)
    plot.showPage()
    plot.save()


if __name__ == "__main__":
    main()
