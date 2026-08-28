import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

try:
    from upstash_redis import Redis
except ImportError:
    Redis = None


app = FastAPI()

SAFE_INT_MAX = 9007199254740991

TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def utf8_key(value):
    return str(value).encode("utf-8")


def safe_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


def finite_number(value):
    return (
        isinstance(value, (int, float, Decimal))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def parse_timestamp(value):
    if not isinstance(value, str) or not TS_RE.fullmatch(value):
        raise ValueError()

    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        return datetime.fromisoformat(value).astimezone(timezone.utc)

    except Exception:
        raise ValueError()


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def make_digest(train_ids, eval_ids, feature_names):
    payload = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
    }

    return hashlib.sha256(
        compact_json(payload).encode("utf-8")
    ).hexdigest()


def canonical_request_hash(body):
    return hashlib.sha256(
        compact_json(body).encode("utf-8")
    ).hexdigest()


def get_redis():
    if Redis is None:
        return None

    url = os.environ.get("KV_REST_API_URL")
    token = os.environ.get("KV_REST_API_TOKEN")

    if not url or not token:
        return None

    return Redis(url=url, token=token)


redis = get_redis()


def storage_key(run_id):
    return "bqml:run:" + run_id


def load_run(run_id):
    if redis is None:
        return None

    return redis.get(storage_key(run_id))


def save_run(run_id, value):
    if redis is None:
        raise RuntimeError("Persistent storage is not configured")

    redis.set(storage_key(run_id), value)


def invalid_input():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )


def validate_selection(body):
    reasons = []

    run_id = body.get("runId")

    if not isinstance(run_id, str) or not (1 <= len(run_id) <= 128):
        reasons.append("INVALID_INPUT")

    forbidden = body.get("forbiddenFeatures")

    if not isinstance(forbidden, list):
        reasons.append("INVALID_INPUT")

    limit = body.get("numTrialsLimit")

    if not safe_int(limit) or limit <= 0:
        reasons.append("INVALID_INPUT")

    rows = body.get("rows")

    if not isinstance(rows, list) or not rows:
        reasons.append("INVALID_INPUT")

    trials = body.get("trials")

    if not isinstance(trials, list):
        reasons.append("INVALID_INPUT")

    if reasons:
        return run_id, {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": sorted(set(reasons), key=utf8_key),
        }

    if len(trials) > limit:
        reasons.append("TRIAL_LIMIT_EXCEEDED")

    if any(not isinstance(x, str) for x in forbidden):
        reasons.append("INVALID_INPUT")

    parsed_rows = []
    row_ids = set()

    for row in rows:

        if not isinstance(row, dict):
            reasons.append("INVALID_INPUT")
            continue

        required = (
            "id",
            "entity",
            "eventTime",
            "predictionTime",
            "version",
            "split",
            "features",
        )

        if any(k not in row for k in required):
            reasons.append("INVALID_INPUT")
            continue

        row_id = row["id"]

        if (
            not isinstance(row_id, str)
            or not row_id
            or row_id in row_ids
        ):
            reasons.append("INVALID_INPUT")

        row_ids.add(row_id)

        if not isinstance(row["entity"], str):
            reasons.append("INVALID_INPUT")

        try:
            event_time = parse_timestamp(row["eventTime"])
            prediction_time = parse_timestamp(row["predictionTime"])
        except ValueError:
            reasons.append("INVALID_INPUT")
            continue

        if not safe_int(row["version"]):
            reasons.append("INVALID_INPUT")

        if row["split"] not in ("TRAIN", "EVAL"):
            reasons.append("INVALID_INPUT")

        features = row["features"]

        if not isinstance(features, dict):
            reasons.append("INVALID_INPUT")
            continue

        parsed_features = {}

        for name, feature in features.items():

            if (
                not isinstance(name, str)
                or not isinstance(feature, dict)
                or "value" not in feature
                or "availableAt" not in feature
            ):
                reasons.append("INVALID_INPUT")
                continue

            try:
                available_at = parse_timestamp(
                    feature["availableAt"]
                )
            except ValueError:
                reasons.append("INVALID_INPUT")
                continue

            parsed_features[name] = available_at

        parsed_rows.append(
            {
                "row": row,
                "event": event_time,
                "prediction": prediction_time,
                "features": parsed_features,
            }
        )

    parsed_trials = []
    trial_ids = set()

    for trial in trials:

        if (
            not isinstance(trial, dict)
            or any(
                k not in trial
                for k in (
                    "trialId",
                    "status",
                    "evalMetric",
                )
            )
        ):
            reasons.append("INVALID_INPUT")
            continue

        trial_id = trial["trialId"]

        if not safe_int(trial_id) or trial_id in trial_ids:
            reasons.append("INVALID_INPUT")

        trial_ids.add(trial_id)

        if trial["status"] not in ("SUCCEEDED", "FAILED"):
            reasons.append("INVALID_INPUT")

        parsed_trials.append(trial)

    if reasons:
        return run_id, {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": sorted(set(reasons), key=utf8_key),
        }

    # Deduplicate by entity + UTC eventTime.
    retained = {}

    for item in parsed_rows:

        row = item["row"]

        key = (
            row["entity"],
            item["event"],
        )

        existing = retained.get(key)

        if (
            existing is None
            or row["version"] > existing["row"]["version"]
            or (
                row["version"] == existing["row"]["version"]
                and utf8_key(row["id"])
                < utf8_key(existing["row"]["id"])
            )
        ):
            retained[key] = item

    kept = list(retained.values())

    common_features = set(kept[0]["features"])

    for item in kept[1:]:
        common_features &= set(item["features"])

    forbidden_set = set(forbidden)

    feature_names = []

    for feature in common_features:

        if feature in forbidden_set:
            continue

        if all(
            item["features"][feature]
            <= item["prediction"]
            for item in kept
        ):
            feature_names.append(feature)

    feature_names.sort(key=utf8_key)

    train_ids = sorted(
        [
            item["row"]["id"]
            for item in kept
            if item["row"]["split"] == "TRAIN"
        ],
        key=utf8_key,
    )

    eval_ids = sorted(
        [
            item["row"]["id"]
            for item in kept
            if item["row"]["split"] == "EVAL"
        ],
        key=utf8_key,
    )

    successful = [
        trial
        for trial in parsed_trials
        if (
            trial["status"] == "SUCCEEDED"
            and finite_number(trial["evalMetric"])
        )
    ]

    if not successful:

        return run_id, {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": feature_names,
            "datasetDigest": None,
            "reasonCodes": ["NO_SUCCESSFUL_TRIAL"],
        }

    selected = max(
        successful,
        key=lambda trial: (
            float(trial["evalMetric"]),
            -trial["trialId"],
        ),
    )

    dataset_digest = make_digest(
        train_ids,
        eval_ids,
        feature_names,
    )

    return run_id, {
        "runId": run_id,
        "selectedTrialId": selected["trialId"],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": dataset_digest,
        "reasonCodes": [],
    }


def evaluate(body, stored):

    reasons = []

    run_id = body.get("runId")
    selected_id = body.get("selectedTrialId")
    supplied_digest = body.get("datasetDigest")

    metric_floor = body.get("metricFloor")
    required_slices = body.get("requiredSlices")
    rows = body.get("rows")

    bytes_processed = body.get("bytesProcessed")
    max_bytes = body.get("maxBytes")

    if not isinstance(run_id, str) or not (1 <= len(run_id) <= 128):
        reasons.append("INVALID_INPUT")

    if not safe_int(selected_id):
        reasons.append("INVALID_INPUT")

    if (
        not isinstance(supplied_digest, str)
        or not DIGEST_RE.fullmatch(supplied_digest)
    ):
        reasons.append("INVALID_INPUT")

    if (
        not finite_number(metric_floor)
        or not 0 <= float(metric_floor) <= 1
    ):
        reasons.append("INVALID_INPUT")

    if not isinstance(required_slices, dict):
        reasons.append("INVALID_INPUT")
    else:
        for name, floor in required_slices.items():

            if (
                not isinstance(name, str)
                or not name
                or not finite_number(floor)
                or not 0 <= float(floor) <= 1
            ):
                reasons.append("INVALID_INPUT")

    if not isinstance(rows, list):
        reasons.append("INVALID_INPUT")

    if not safe_int(bytes_processed):
        reasons.append("INVALID_INPUT")

    if not safe_int(max_bytes):
        reasons.append("INVALID_INPUT")

    if reasons:
        return sorted(set(reasons), key=utf8_key), None, False

    lineage_ok = (
        stored is not None
        and stored["selectedTrialId"] is not None
        and stored["runId"] == run_id
        and stored["selectedTrialId"] == selected_id
        and stored["datasetDigest"] == supplied_digest
        and not stored["reasonCodes"]
    )

    if not lineage_ok:
        reasons.append("INVALID_LINEAGE")

    # Empty test data.
    if not rows:

        if bytes_processed > max_bytes:
            reasons.append("BYTE_LIMIT")

        return (
            sorted(set(reasons), key=utf8_key),
            None,
            False,
        )

    correct = 0
    slices = {}
    invalid_row = False

    for row in rows:

        valid = (
            isinstance(row, dict)
            and isinstance(row.get("label"), int)
            and not isinstance(row.get("label"), bool)
            and row.get("label") in (0, 1)
            and isinstance(row.get("prediction"), int)
            and not isinstance(row.get("prediction"), bool)
            and row.get("prediction") in (0, 1)
            and isinstance(row.get("slice"), str)
            and bool(row.get("slice"))
        )

        if not valid:
            invalid_row = True
            continue

        prediction_correct = int(
            row["label"] == row["prediction"]
        )

        correct += prediction_correct

        bucket = slices.setdefault(
            row["slice"],
            [0, 0],
        )

        bucket[0] += prediction_correct
        bucket[1] += 1

    if invalid_row:

        reasons.append("INVALID_TEST_ROW")

        if bytes_processed > max_bytes:
            reasons.append("BYTE_LIMIT")

        return (
            sorted(set(reasons), key=utf8_key),
            None,
            False,
        )

    test_metric = round(
        correct / len(rows),
        12,
    )

    if test_metric < float(metric_floor):
        reasons.append("AGGREGATE_FLOOR")

    slice_pass = True

    for name in sorted(
        required_slices.keys(),
        key=utf8_key,
    ):

        if name not in slices:

            reasons.append(
                "MISSING_SLICE:" + name
            )

            slice_pass = False

            continue

        slice_metric = round(
            slices[name][0] / slices[name][1],
            12,
        )

        if slice_metric < float(
            required_slices[name]
        ):

            reasons.append(
                "SLICE_FLOOR:" + name
            )

            slice_pass = False

    if bytes_processed > max_bytes:
        reasons.append("BYTE_LIMIT")

    if not lineage_ok:
        slice_pass = False

    return (
        sorted(set(reasons), key=utf8_key),
        test_metric,
        slice_pass,
    )


@app.post("/bqml")
async def bqml(request: Request):

    try:
        body = await request.json()
    except Exception:
        return invalid_input()

    if not isinstance(body, dict):
        return invalid_input()

    phase = body.get("phase")

    if phase not in ("select", "evaluate"):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    # -----------------------------
    # Selection phase
    # -----------------------------

    if phase == "select":

        run_id, response = validate_selection(body)

        if not isinstance(run_id, str):
            return invalid_input()

        fingerprint = canonical_request_hash(body)

        existing = load_run(run_id)

        if existing is not None:

            if existing["fingerprint"] != fingerprint:
                return JSONResponse(
                    status_code=409,
                    content={"error": "RUN_ID_CONFLICT"},
                )

            return JSONResponse(
                status_code=200,
                content=existing["response"],
            )

        stored = {
            "fingerprint": fingerprint,
            "response": response,
        }

        try:
            save_run(run_id, stored)
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"error": "STORAGE_UNAVAILABLE"},
            )

        return JSONResponse(
            status_code=200,
            content=response,
        )

    # -----------------------------
    # Evaluation phase
    # -----------------------------

    run_id = body.get("runId")

    stored_record = (
        load_run(run_id)
        if isinstance(run_id, str)
        else None
    )

    stored_response = (
        stored_record["response"]
        if stored_record
        else None
    )

    reasons, metric, slice_pass = evaluate(
        body,
        stored_response,
    )

    bytes_processed = body.get("bytesProcessed")

    if not safe_int(bytes_processed):
        bytes_processed = 0

    selected_id = body.get("selectedTrialId")
    supplied_digest = body.get("datasetDigest")

    admit = (
        not reasons
        and metric is not None
        and slice_pass
        and metric >= float(body["metricFloor"])
        and bytes_processed <= body["maxBytes"]
    )

    return JSONResponse(
        status_code=200,
        content={
            "runId": run_id,
            "selectedTrialId": selected_id,
            "datasetDigest": supplied_digest,
            "testMetric": metric,
            "criticalSlicePass": bool(slice_pass),
            "decision": "admit" if admit else "reject",
            "bytesProcessed": bytes_processed,
            "reasonCodes": reasons,
        },
    )