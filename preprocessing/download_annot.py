import asyncio
import base64
import contextlib
import json
import os
import tempfile
import uuid
from urllib.parse import parse_qs, unquote, urlparse

import hydra
import ray
from omegaconf import DictConfig
from rationai.empaia.workbench_api.clients.asynchronous import EmpaiaClientAiohttp
from rationai.empaia.workbench_api.token_manager.token_manager import TokenManager
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def decompress_hex(compressed: str) -> str:
    if not compressed:
        return ""
    padding_needed = len(compressed) % 4
    if padding_needed:
        compressed += "=" * (4 - padding_needed)
    try:
        return base64.urlsafe_b64decode(compressed).hex()
    except Exception:
        return ""


def format_as_uuid(hex_str: str) -> str:
    if len(hex_str) == 32:
        return str(uuid.UUID(hex_str))
    return hex_str


def parse_selection_url(ids_string: str) -> dict:
    url_cases, partial_cases = [], {}
    parts = [p for p in ids_string.split(";") if p]
    for part in parts:
        segments = part.split(":")
        case_id = format_as_uuid(decompress_hex(segments[0]) or segments[0])
        if len(segments) == 1:
            url_cases.append(case_id)
        else:
            partial_cases[case_id] = [
                format_as_uuid(decompress_hex(s) or s) for s in segments[1:]
            ]
    return {"urlCases": url_cases, "partialCases": partial_cases}


def parse_selection_urls(urls: list[str]) -> dict:
    full_cases: set[str] = set()
    partial_cases: dict[str, set[str]] = {}

    for raw_url in urls:
        if not raw_url:
            continue

        query = parse_qs(urlparse(raw_url).query)
        data_string = query.get("s", [""])[0]

        # Backward compatibility: allow passing only encoded selection payload.
        if not data_string and ";" in raw_url:
            data_string = raw_url

        if not data_string:
            continue

        parsed = parse_selection_url(unquote(data_string))

        for case_id in parsed["urlCases"]:
            full_cases.add(case_id)
            partial_cases.pop(case_id, None)

        for case_id, slide_ids in parsed["partialCases"].items():
            if case_id in full_cases:
                continue
            partial_cases.setdefault(case_id, set()).update(slide_ids)

    return {
        "urlCases": list(full_cases),
        "partialCases": {
            case_id: list(slide_ids) for case_id, slide_ids in partial_cases.items()
        },
    }


async def check_slide_annotations(
    client, case_id, slide_id, out_path, display_name=None
):
    # client needs to be fixed, returns only polygons instead of the whole json
    try:
        polygons_json = await client.get_annotation_polygons(
            case_id=case_id, slide_id=slide_id, filter_classes=None, scale_factor=1.0
        )

        items = polygons_json.get("items", [])

        if len(items) > 0:
            os.makedirs(out_path, exist_ok=True)
            file_name = f"{display_name}.json" if display_name else f"{slide_id}.json"
            save_path = os.path.join(out_path, file_name)

            with open(save_path, "w") as f:
                json.dump(polygons_json, f, indent=4)

            print(f"💾 Saved {len(items)} annotations with metadata to {file_name}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error checking {slide_id}: {e}")
        return False


async def get_case_slides_metadata(
    client: EmpaiaClientAiohttp, case_id: str
) -> dict[str, str]:
    endpoint = f"cases/{case_id}/slides"

    print("TESTING METADATA FETCH", endpoint)
    response = await client._make_request(endpoint)

    slides = (
        response["items"]
        if isinstance(response, dict) and "items" in response
        else response
    )

    return {slide["id"]: slide.get("local_id") for slide in slides}


async def run_download_task(config: DictConfig, out_path: str):
    if not ray.is_initialized():
        ray.init(namespace="empaia", ignore_reinit_error=True)

    with contextlib.suppress(Exception):
        TokenManager.kill_global_instance()

    auth_config = {
        "client_id": config.client_id,
        "client_secret": config.client_secret,
        "token_url": config.token_url,
        "authentication_url": config.authentication_url,
        "auth_scopes": config.auth_scopes,
        "workbench_base_url": config.workbench_base_url,
    }

    TokenManager.init_global_instance(**auth_config)
    client = EmpaiaClientAiohttp(config.workbench_base_url, config.app_id)

    urls = list(config.urls_to_check) if config.urls_to_check else []

    parsed = parse_selection_urls(urls)

    to_check = []

    all_case_ids = list(set(parsed["urlCases"] + list(parsed["partialCases"].keys())))

    print(f"📡 Fetching metadata for {len(all_case_ids)} cases...")
    for c_id in all_case_ids:
        try:
            mapping = await get_case_slides_metadata(client, c_id)

            if c_id in parsed["partialCases"]:
                target_slide_uuids = parsed["partialCases"][c_id]
            else:
                target_slide_uuids = list(mapping.keys())

            for s_uuid in target_slide_uuids:
                full_id = mapping.get(s_uuid, s_uuid)
                short_id = (
                    ".".join(full_id.split(".")[3:]) if "." in full_id else full_id
                )
                to_check.append((c_id, s_uuid, short_id))
        except Exception as e:
            print(f"⚠️ Could not fetch metadata for case {c_id}: {e}")
            return

    print(f"🚀 Processing {len(to_check)} slides...")
    tasks = [
        check_slide_annotations(client, c, s, out_path, name) for c, s, name in to_check
    ]
    status_list = await asyncio.gather(*tasks)

    print("\n" + "=" * 30)
    print("📊 FINAL REPORT")
    print(f"Total extracted Slides: {len(to_check)}")
    print(f"Slides with Annotations:    {sum(status_list)}")
    print("=" * 30)

    await client.close()


@with_cli_args(["+preprocessing=download_annot"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        asyncio.run(run_download_task(config, tmpdir))

        logger.experiment.log_artifacts(logger.run_id, tmpdir, "annotations")


if __name__ == "__main__":
    main()
