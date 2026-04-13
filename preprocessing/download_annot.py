import asyncio
import base64
import json
import os
import uuid
from urllib.parse import unquote

import httpx
import ray
from rationai.empaia.workbench_api.clients.asynchronous import EmpaiaClientAiohttp
from rationai.empaia.workbench_api.token_manager.token_manager import TokenManager


# Configuration
CONFIG = {
    "client_id": "af1cc4f6-b1a0-4e1f-bdec-361e226b9769",
    "client_secret": "",
    "token_url": "https://login.aai.lifescience-ri.eu/oidc/token",
    "authentication_url": "https://login.aai.lifescience-ri.eu/oidc/devicecode",
    "auth_scopes": "openid profile offline_access",
    "workbench_base_url": "https://histopat.rationai.cloud.trusted.e-infra.cz/api/wb/v3",
}
OUT_PATH = (
    "/mnt/projects/inflammatory_bowel_disease/ulcerative_colitis_dysplasia/annot_stash"
)

BASE_URL = CONFIG["workbench_base_url"]
APP_ID = "4e485b74-413e-477d-8e09-2c38ae57e582"
CASE_ID = "cc41070d-c916-4117-90f4-13c57a9910d5"
SLIDE_ID = "bc91b863-a065-419b-b90f-e709cf547565"
URL_TO_CHECK = "https://histopat.rationai.cloud.trusted.e-infra.cz/authorized/cases/selection?s=zEEHDckWQReQ9BPFepkQ1Q%3Bc5E4jjUXQXOvw78YP0ELdw%3BquXiaeh6Raei7tmRY2cv1A%3BtF0dx_exSXO8aJqleD_zQQ%3BODXhHoGPTgWFwDFSzZUPmg%3BvQCyU7_qSBet883_NdiWWA%3BnImEi3XuSRG0_xVKlyPgnw%3ByXBS-fN5RmWeTyehdTb0yQ%3BK-DFzsYJTYy8iE5MESBz5Q%3BYWt8BAq4SkKgDtOTIxlubw%3B9rF-v4FBRjqniMebSfXYFQ%3BGFVcMhmtSNm3Bf7TICthSQ%3Bps7ir74kQZWcMy0hRAYnZA%3B0-uh0sedS9eKGKCDuyHjmQ%3B-lYB4vyLTgamSBChjrF2cA%3B6byulta_TFelHclaf83Ftg%3BYtUh7QKUREShxC71zVvFJw%3Bf8GnswFaRUOs-jNi3CKPkA%3BCatNV22TQGeJJSqHPyZ8XQ%3B4c8t5oDJRbWfkIJV72Zy7A%3Bzcj9gKyrSCmUO2fVSF4u4w%3BpZCkCochQmGgD-9BgLzH-w%3BQlklXf9rRXuDuz-QYUVAkg%3BUV89qZDzQViXxcLc_1-N9Q%3Baukb-R5_R_eSzzkApWp11g%3BYVp9ruZqSFubngjCi564VA%3BL1VqwsjjQ4G7QyFTfQXDVA%3Bq-0FoDRpRFOTNYuiYIMg3g%3BI1gto9R6QV6ciASp8yDATQ%3B72IOsH7HRXK6ZO3QeOks8Q%3BSz-WQyWGRzy5Znog6I_l0g%3B9QYUcHapSQOJIq3q1l5Wfg%3B5d1YtUkbS0mNMPgTeYXQ5Q%3BfwmcI2kcTFGR3kfsm5eJbQ%3BfWIF26TVSV6MtU4yRzrNHg%3B94OIvVrWTMKXaMA3O415Vg%3B3HdKwqDAQamZvhKuSyJbjg%3BHD6bxguVSbiCOoTGCgXl7A%3BJd3zWyw3RwKLJ1IduOW7Lw%3BEr-iZTT0QzOnsdlx4mUG4w%3BvVBK5UsJTPibE4svQpixdA%3BqzyRPJL5QRWXivExHMYCVw%3Bh3HsvKCuS9ms8jGIePBecg%3Bxpn3Pw7xQIqifGd9dVHxlA%3BMfZ2bXZjSJmv1_-FR3IRrA%3Bz3kWxfUqStiqMauEUNfh4A%3BGYv5x-aJTMuvpsNHjmN6jw%3B5wK7NnxyTJGbe5u84PxQGQ%3B2792QtdFQlq3U4blDtrFvA%3BVVwxYp2SS3aCXhXA7KO5Dw%3BsPnCGymCQny3WHKElJx5SQ%3B0wHa8rIvQL6VKMqr_VDU0A%3BVS6yppDJQ2eOsLjaMcUbew%3BriWO-7XFSsmEh4HHBQhYSw%3BKi_TszKeSyeQjdMSwApJrA%3Bew4CzZJURt6ldssHulpigg%3BghdEbqCGSo2OTQXjdgeQRw%3Baqt-v1CRTCisOpXTcGC0Xg%3BwV13MuAfSpq_2dVIROftyA%3B7Zl3Ss_OQgKP_Zzz0CKqWg%3BgBago5YQQQybmWAHK8oKeQ%3Bd3XbJTH_Syqh28H0DyF2Tw%3B5HzGqNUmTMCiAt8_kYWEGA%3BjKM1Em1JQVqDOdm1Eq4tNQ%3BkKC9lEXSSCmNcFT8YYjoVA%3BaRiZu2X2SFCekm7Vyhy4-A%3By4Ox-uY5Q8yU25BWWUlBCA%3ByOZ1EmQBT7-QHPnFhzOPjA%3Bw0xTyI1-RTqnAhha7KOflw%3BSHj-1nmiQnykuqjM6rTmcw%3B"


def decompress_hex(compressed: str) -> str:
    if not compressed:
        return ""
    padding_needed = len(compressed) % 4
    if padding_needed:
        compressed += "=" * (4 - padding_needed)
    try:
        return base64.urlsafe_b64decode(compressed).hex()
    except:
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


async def check_slide_annotations(client, case_id, slide_id, display_name=None):
    """Downloads polygons and saves them using display_name (short local_id) if provided."""
    try:
        polygons = await client.get_annotation_polygons(
            case_id=case_id, slide_id=slide_id, filter_classes=None, scale_factor=1.0
        )

        if len(polygons["items"]) > 0:
            os.makedirs(OUT_PATH, exist_ok=True)
            # Use short_id if available, otherwise fallback to slide UUID
            file_name = f"{display_name}.json" if display_name else f"{slide_id}.json"
            save_path = os.path.join(OUT_PATH, file_name)

            with open(save_path, "w") as f:
                json.dump(polygons, f, indent=4)
            print(f"💾 Saved {len(polygons['items'])} polygons to {file_name}")
            return True
        else:
            return False
    except Exception as e:
        print(f"❌ Error checking {slide_id}: {e}")
        return False


async def get_case_slides_metadata(
    client: EmpaiaClientAiohttp, case_id: str
) -> dict[str, str]:
    endpoint = f"cases/{case_id}/slides"

    # This uses the root/IDP token and user-id header automatically
    response = await client._make_request(endpoint)

    # WBS v3 usually returns a list or a dict with an 'items' key
    slides = (
        response["items"]
        if isinstance(response, dict) and "items" in response
        else response
    )

    # Create a mapping of {uuid: local_id}
    # e.g. {"bc91b863...": "9965_20_HE_0"}
    return {slide["id"]: slide.get("local_id") for slide in slides}


async def run_download_task(case_id, slide_id):
    if not ray.is_initialized():
        ray.init(namespace="empaia", ignore_reinit_error=True)

    try:
        TokenManager.kill_global_instance()
    except:
        pass

    TokenManager.init_global_instance(**CONFIG)
    client = EmpaiaClientAiohttp(BASE_URL, APP_ID)

    # 1. Parse Selection URL
    data_string = unquote(URL_TO_CHECK.split("?s=")[1])
    parsed = parse_selection_url(data_string)

    # 2. Map all Case and Slide pairs
    to_check = []  # List of (case_id, slide_id, short_id)

    all_case_ids = list(set(parsed["urlCases"] + list(parsed["partialCases"].keys())))

    print(f"📡 Fetching metadata for {len(all_case_ids)} cases...")
    for c_id in all_case_ids:
        try:
            # Fetch the mapping {UUID: local_id}
            mapping = await get_case_slides_metadata(client, c_id)

            # Determine which slides in this case we care about
            if c_id in parsed["partialCases"]:
                target_slide_uuids = parsed["partialCases"][c_id]
            else:
                target_slide_uuids = list(mapping.keys())

            for s_uuid in target_slide_uuids:
                full_id = mapping.get(s_uuid, s_uuid)
                # Split logic: take everything after the 3rd dot
                short_id = (
                    ".".join(full_id.split(".")[3:]) if "." in full_id else full_id
                )
                to_check.append((c_id, s_uuid, short_id))
        except Exception as e:
            print(f"⚠️ Could not fetch metadata for case {c_id}: {e}")

    # 3. Run Downloads with proper names
    print(f"🚀 Processing {len(to_check)} slides...")
    tasks = [check_slide_annotations(client, c, s, name) for c, s, name in to_check]
    status_list = await asyncio.gather(*tasks)

    # 4. Final Report
    print("\n" + "=" * 30)
    print(f"📊 FINAL REPORT")
    print(f"Total Slides: {len(to_check)}")
    print(f"Annotated:    {sum(status_list)}")
    print("=" * 30)

    await client.close()


if __name__ == "__main__":
    asyncio.run(run_download_task(CASE_ID, SLIDE_ID))
