"""Generate invented example documents without downloading a dataset."""
import argparse
import csv
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("sample.csv"))
    parser.add_argument("--documents", type=int, default=180)
    args = parser.parse_args()
    if args.documents < 4:
        parser.error("--documents must be at least four")
    themes = [
        (["astronomers", "space researchers", "observatory scientists"],
         ["measured the orbit of an exoplanet", "analyzed light from a distant galaxy", "studied the atmosphere of a new planet"],
         "The telescope collected images and spectral measurements for the astronomy project."),
        (["urban planners", "public transport engineers", "city officials"],
         ["proposed a new bus route", "studied subway service frequency", "improved connections between transit stations"],
         "Residents discussed commuting times, accessible stations, and affordable public transportation."),
        (["gardeners", "horticulture students", "community farmers"],
         ["tested soil moisture in vegetable beds", "planted seasonal herbs", "compared compost treatments for seedlings"],
         "The garden team recorded plant growth, watering schedules, and harvest quality."),
    ]
    rng = random.Random(42)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text"])
        for i in range(args.documents):
            subjects, actions, detail = themes[i % len(themes)]
            text = f"In report {i}, {rng.choice(subjects)} {rng.choice(actions)}. {detail} The discussion included observations from week {1 + i % 12}."
            writer.writerow([f"sample-{i:04}", text])
    print(f"Wrote {args.documents} synthetic documents to {args.output}")


if __name__ == "__main__":
    main()
