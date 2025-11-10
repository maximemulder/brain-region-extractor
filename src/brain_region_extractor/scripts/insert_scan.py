#!/usr/bin/env python

import argparse
import json
from pathlib import Path

from sqlalchemy.orm import Session

from brain_region_extractor.database.engine import get_engine
from brain_region_extractor.database.insertion import insert_scan
from brain_region_extractor.scan import Scan


def read_scan_json(path: Path) -> Scan:
    with open(path) as file:
        scan_data = json.load(file)

    return Scan(**scan_data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Insert a scan JSON into the database.'
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input JSON file containing the scan data.'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Loading Scan data from {args.input}")

    # Load Scan from JSON
    scan = read_scan_json(args.input)

    print(f"Loaded scan: {scan.file_name}")
    print(f"Number of regions: {len(scan.regions)}")

    # Insert into database
    print("Inserting scan into database...")

    session = Session(get_engine())

    db_scan = insert_scan(session, scan)

    print(f"Successfully inserted scan with ID: {db_scan.id}")


if __name__ == '__main__':
    main()
