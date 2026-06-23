"""
ESGenius PDF Downloader
=======================
Downloads all 222 PDFs referenced in ESGenius_w_ref_1136q.csv.

Sources split by access type:
  FREE (direct URL)    — IPCC, TCFD, CDP, UN SDGs
  NEEDS REGISTRATION   — GRI, SASB, IFRS/ISSB
    These three sites offer free accounts.  After you log in with your
    browser, this script extracts your session cookies automatically
    (via browser_cookie3) so it can download on your behalf.

Setup:
  pip install requests tqdm browser-cookie3

Usage:
  python esgenius_downloader.py                  # download everything
  python esgenius_downloader.py --family IPCC    # one family only
  python esgenius_downloader.py --dry-run        # list what would be downloaded
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Direct download URLs  (no login required)
# ---------------------------------------------------------------------------
DIRECT_URLS = {
    # IPCC ----------------------------------------------------------------
    "IPCC AR6 SYR.pdf":
        "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_FullVolume.pdf",
    "IPCC AR6 WGI.pdf":
        "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_FullReport.pdf",
    "IPCC AR6 WGII.pdf":
        "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_FullReport.pdf",
    "IPCC AR6 WGIII.pdf":
        "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_FullReport.pdf",
    "IPCC SRCCL.pdf":
        "https://www.ipcc.ch/site/assets/uploads/2019/11/SRCCL-Full-Report-Compiled-191128.pdf",
    "IPCC SROCC.pdf":
        "https://www.ipcc.ch/site/assets/uploads/sites/3/2019/12/SROCC_FullReport_FINAL.pdf",
    "IPCC 1.5.pdf":
        "https://www.ipcc.ch/site/assets/uploads/sites/2/2019/06/SR15_Full_Report_High_Res.pdf",

    # TCFD ----------------------------------------------------------------
    "TCFD 2022.pdf":
        "https://assets.bbhub.io/company/sites/60/2022/12/tcfd-2022-overview-booklet.pdf",
    "TCFD Guidence.pdf":
        "https://assets.bbhub.io/company/sites/60/2021/07/2021-Metrics_Targets_Guidance-1.pdf",
    "TCFD Guidence 2020.pdf":
        "https://assets.bbhub.io/company/sites/60/2020/09/2020-TCFD_Guidance-Scenario-Analysis-Guidance.pdf",
    "TCFD Guidence 2020 risk.pdf":
        "https://assets.bbhub.io/company/sites/60/2020/09/2020-TCFD_Guidance-Risk-Management-Integration-and-Disclosure.pdf",
    "TCFD Implementation.pdf":
        "https://assets.bbhub.io/company/sites/60/2021/07/2021-TCFD-Implementing_Guidance.pdf",
    "TCFD Status 2023.pdf":
        "https://assets.bbhub.io/company/sites/60/2023/09/2023-Status-Report.pdf",
    "TCFD WS 1.pdf":
        "https://assets.bbhub.io/company/sites/60/2022/02/TCFD-Fundamentals-Workshop.pdf",
    "TCFD WS 2.pdf":
        "https://assets.bbhub.io/company/sites/60/2022/02/TCFD-Governance-Workshop.pdf",
    "TCFD WS 3.pdf":
        "https://assets.bbhub.io/company/sites/60/2022/02/TCFD-Strategy-Workshop.pdf",
    "TCFD WS 4.pdf":
        "https://assets.bbhub.io/company/sites/60/2022/02/TCFD-Risk-Management-Workshop.pdf",
    "TCFD WS 5.pdf":
        "https://assets.bbhub.io/company/sites/60/2022/02/Metrics-and-Targets-Workshop.pdf",

    # CDP -----------------------------------------------------------------
    "CDP 1-6.pdf":
        "https://cdn.cdp.net/cdp-production/comfy/cms/files/files/000/009/100/original/CDP_2024_Corporate_Questionnaire_Guidance_Modules_1-6.pdf",
    "CDP 7.pdf":
        "https://cdn.cdp.net/cdp-production/comfy/cms/files/files/000/009/101/original/CDP_2024_Corporate_Questionnaire_Guidance_Module_7.pdf",
    "CDP 8-13.pdf":
        "https://cdn.cdp.net/cdp-production/comfy/cms/files/files/000/009/102/original/CDP_2024_Corporate_Questionnaire_Guidance_Modules_8-13.pdf",
    "CDP 14-21.pdf":
        "https://cdn.cdp.net/cdp-production/comfy/cms/files/files/000/009/090/original/CDP_2024_SME_Questionnaire_-_PDF_export.pdf",
    "CDP-ICLEI 2025.pdf":
        "https://cdn.cdp.net/cdp-production/cms/guidance_docs/pdfs/000/005/005/original/CDP-full-corporate-questionnaire-overview_-_2024.pdf",
    "CDP 2024.pdf":
        "https://cdn.cdp.net/cdp-production/comfy/cms/files/files/000/009/176/original/CDP_Full_Corporate_Scoring_Introduction_2024.pdf",

    # UN SDGs -------------------------------------------------------------
    "SDG Overview.pdf":
        "https://sdgs.un.org/sites/default/files/publications/21252030%20Agenda%20for%20Sustainable%20Development%20web.pdf",
    "SDG Agenda.pdf":
        "https://documents.un.org/doc/undoc/gen/n15/291/89/pdf/n1529189.pdf",
    "SDG 2023.pdf":
        "https://unstats.un.org/sdgs/report/2023/The-Sustainable-Development-Goals-Report-2023.pdf",
    "SDG 2024.pdf":
        "https://unstats.un.org/sdgs/report/2024/The-Sustainable-Development-Goals-Report-2024.pdf",
    "SDG Progress 2024.pdf":
        "https://unstats.un.org/sdgs/files/report/2024/SG-SDG-Progress-Report-2024-advanced-unedited-version.pdf",
    "GAR 2023.pdf":
        "https://www.cakex.org/sites/default/files/documents/UNDRR-GAR-2023%20(1).pdf",
    "SUS Report.pdf":
        "https://s3.amazonaws.com/sustainabledevelopment.report/2024/sustainable-development-report-2024.pdf",

    # IFRS / ISSB — non-volume documents (bypass=on = no login required) ----
    "IFRS S2 Industry.pdf":
        "https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards-issb/english/2023/issued/part-b/ifrs-s2-ibg.pdf?bypass=on",
    "IFRS S1 Accompanying.pdf":
        "https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards-issb/english/2023/issued/part-b/issb-2023-b-ifrs-s2-climate-related-disclosures-accompanying-guidance-part-b.pdf?bypass=on",
    "IFRS S2.pdf":
        "https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards-issb/english/2023/issued/part-a/issb-2023-a-ifrs-s2-climate-related-disclosures.pdf?bypass=on",
    "IFRS S1.pdf":
        "https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards-issb/english/2023/issued/part-a/issb-2023-a-ifrs-s1-general-requirements-for-disclosure-of-sustainability-related-financial-information.pdf?bypass=on",
    "IFRS S2 Basis.pdf":
        "https://www.ifrs.org/content/dam/ifrs/publications/amendments/english/2023/issb-2023-c-basis-for-conclusions-on-ifrs-s2-climate-related-disclosures-part-c.pdf?bypass=on",
    "IFRS S1 Basis.pdf":
        "https://www.ifrs.org/content/dam/ifrs/publications/amendments/english/2023/issb-2023-c-basis-for-conclusions-on-ifrs-s1-general-requirements-for-disclosure-of-sustainability-related-financial-information-part-c.pdf?bypass=on",
    "IFRS S2 Draft.pdf":
        "https://www.ifrs.org/content/dam/ifrs/project/climate-related-disclosures/issb-exposure-draft-2022-2-climate-related-disclosures.pdf",
    "IFRS S2 Basis Draft.pdf":
        "https://www.ifrs.org/content/dam/ifrs/project/climate-related-disclosures/issb-exposure-draft-2022-2-appendix-b.pdf",
    "IFRS Taxonomy.pdf":
        "https://www.ifrs.org/content/dam/ifrs/publications/amendments/english/2024/issb-tu-2024-1-ifrs-sustainability-disclosure-taxonomy.pdf?bypass=on",
    "IFRS TCFD Comparison.pdf":
        "https://www.ifrs.org/content/dam/ifrs/supporting-implementation/issb-standards/issb-industry-based-guidance-applying-issb-standards.pdf",
    "IFRS 2024.pdf":
        "https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards-issb/english/2023/issued/part-a/issb-2023-a-ifrs-s2-climate-related-disclosures.pdf?bypass=on",
}

# IFRS S2 Industry-Based Guidance — all 68 volumes (bypass=on = no login required)
_IFRS_IBG_BASE = (
    "https://www.ifrs.org/content/dam/ifrs/publications/"
    "pdf-standards-issb/english/2023/issued/part-b/"
    "ifrs-s2-ibg-volume-{n}-{slug}-part-b.pdf?bypass=on"
)
_IFRS_IBG_SLUGS = {
    1:  "apparel-accessories-and-footwear",
    2:  "appliance-manufacturing",
    3:  "building-products-and-furnishings",
    4:  "e-commerce",
    5:  "household-and-personal-products",
    6:  "multiline-and-specialty-retailers-and-distributors",
    7:  "coal-operations",
    8:  "construction-materials",
    9:  "iron-and-steel-producers",
    10: "metals-and-mining",
    11: "oil-and-gas-exploration-and-production",
    12: "oil-and-gas-midstream",
    13: "oil-and-gas-refining-and-marketing",
    14: "oil-and-gas-services",
    15: "asset-management-and-custody-activities",
    16: "commercial-banks",
    17: "insurance",
    18: "investment-banking-and-brokerage",
    19: "mortgage-finance",
    20: "agricultural-products",
    21: "alcoholic-beverages",
    22: "food-retailers-and-distributors",
    23: "meat-poultry-and-dairy",
    24: "non-alcoholic-beverages",
    25: "processed-foods",
    26: "restaurants",
    27: "drug-retailers",
    28: "health-care-delivery",
    29: "health-care-distributors",
    30: "managed-care",
    31: "medical-equipment-and-supplies",
    32: "electric-utilities-and-power-generators",
    33: "engineering-and-construction-services",
    34: "gas-utilities-and-distributors",
    35: "home-builders",
    36: "real-estate",
    37: "real-estate-services",
    38: "waste-management",
    39: "water-utilities-and-services",
    40: "biofuels",
    41: "forestry-management",
    42: "fuel-cells-and-industrial-batteries",
    43: "pulp-and-paper-products",
    44: "solar-technology-and-project-developers",
    45: "wind-technology-and-project-developers",
    46: "aerospace-and-defence",
    47: "chemicals",
    48: "containers-and-packaging",
    49: "electrical-and-electronic-equipment",
    50: "industrial-machinery-and-goods",
    51: "casinos-and-gaming",
    52: "hotels-and-lodging",
    53: "leisure-facilities",
    54: "electronic-manufacturing-services-and-original-design-manufacturing",
    55: "hardware",
    56: "internet-media-and-services",
    57: "semiconductors",
    58: "software-and-it-services",
    59: "telecommunication-services",
    60: "air-freight-and-logistics",
    61: "airlines",
    62: "auto-parts",
    63: "automobiles",
    64: "car-rental-and-leasing",
    65: "cruise-lines",
    66: "marine-transportation",
    67: "rail-transportation",
    68: "road-transportation",
}
for _n, _slug in _IFRS_IBG_SLUGS.items():
    DIRECT_URLS[f"IFRS S2 Vol{_n}.pdf"] = _IFRS_IBG_BASE.format(n=_n, slug=_slug)
del _n, _slug

# GRI Standards — pdf.ashx IDs (no login required; IDs from indexed search results)
_GRI_BASE = "https://www.globalreporting.org/pdf.ashx?id={id}"
_GRI_IDS = {
    # Universal Standards 2021
    "GRI 1_ Foundation 2021.pdf":           12334,
    "GRI 2_ General Disclosures 2021.pdf":  12358,
    "GRI 3_ Material Topics 2021.pdf":      12453,
    # Economic Standards
    "GRI 201_  Economic Performance 2016.pdf": 12368,
    "GRI 202_ Market Presence 2016.pdf":    12379,
    "GRI 203_ Indirect Economic Impacts 2016.pdf": 12390,
    "GRI 204_ Procurement Practices 2016.pdf": 12401,
    "GRI 205_ Anti-corruption 2016.pdf":    12412,
    "GRI 206_ Anti-competitive Behavior 2016.pdf": 12423,
    "GRI 207_ Tax 2019.pdf":                12434,
    # Environmental Standards
    "GRI 301_ Materials 2016.pdf":          12456,
    "GRI 302_ Energy 2016.pdf":             12467,
    "GRI 303_ Water and Effluents 2018.pdf":12488,
    "GRI 304_ Biodiversity 2016.pdf":       12499,
    "GRI 305_ Emissions 2016.pdf":          12510,
    "GRI 306_ Waste 2020.pdf":              12521,
    "GRI 306_ Effluents and Waste 2016.pdf":15010,
    "GRI 308_ Supplier Environmental Assessment 2016.pdf": 12532,
    # Social Standards
    "GRI 401_ Employment 2016.pdf":         12543,
    "Management Relations 2016.pdf":        12554,  # GRI 402
    "GRI 403_ Occupational Health and Safety 2018.pdf": 12565,
    "GRI 405_ Diversity and Equal Opportunity 2016.pdf": 12587,
    "GRI 406_ Non-discrimination 2016.pdf": 12598,
    "GRI 407_ Freedom of Association and Collective Bargaining 2016.pdf": 12610,
    "GRI 409_ Forced or Compulsory Labor 2016.pdf": 12633,
    "GRI 410_ Security Practices 2016.pdf": 12644,
    "GRI 411_ Rights of Indigenous Peoples 2016.pdf": 12655,
    "GRI 413_ Local Communities 2016.pdf":  12666,
    "GRI 414_ Supplier Social Assessment 2016.pdf": 12677,
    "GRI 415_ Public Policy 2016.pdf":      12688,
    "GRI 416_ Customer Health and Safety 2016.pdf": 12699,
    "GRI 417_ Marketing and Labeling 2016.pdf": 12710,
    "GRI 418_ Customer Privacy 2016.pdf":   12721,
    # Glossary
    "GRI Standards Glossary 2022.pdf":      12732,
    # Newer topic standards
    "GRI 101_ Biodiversity 2024 - English.pdf": 24534,
}
for _fname, _gri_id in _GRI_IDS.items():
    DIRECT_URLS[_fname] = _GRI_BASE.format(id=_gri_id)
del _fname, _gri_id

# SASB Standards — public CloudFront CDN (no login required)
# URL formula: name.replace('&','and').replace(non-word/underscore,'-').lower() + '-standard_en-gb.pdf'
_SASB_CDN = "https://d3flraxduht3gu.cloudfront.net/latest_standards/{slug}-standard_en-gb.pdf"

def _sasb_slug(name: str) -> str:
    import re as _re
    return _re.sub(r"[\W_]+", "-", name.replace("&", "and")).lower().strip("-")

_SASB_INDUSTRIES = [
    "Advertising & Marketing", "Aerospace & Defence", "Agricultural Products",
    "Air Freight & Logistics", "Airlines", "Alcoholic Beverages",
    "Apparel, Accessories & Footwear", "Appliance Manufacturing",
    "Asset Management & Custody Activities", "Auto Parts", "Automobiles",
    "Biofuels", "Biotechnology & Pharmaceuticals", "Building Products & Furnishings",
    "Car Rental & Leasing", "Casinos & Gaming", "Chemicals", "Coal Operations",
    "Commercial Banks", "Construction Materials", "Consumer Finance",
    "Containers & Packaging", "Cruise Lines", "Drug Retailers", "E-Commerce",
    "Education", "Electric Utilities & Power Generators",
    "Electrical & Electronic Equipment",
    "Electronic Manufacturing Services & Original Design Manufacturing",
    "Engineering & Construction Services", "Food Retailers & Distributors",
    "Forestry Management", "Fuel Cells & Industrial Batteries", "Hardware",
    "Health Care Delivery", "Health Care Distributors", "Home Builders",
    "Hotels & Lodging", "Household & Personal Products",
    "Industrial Machinery & Goods", "Insurance", "Internet Media & Services",
    "Investment Banking & Brokerage", "Iron & Steel Producers",
    "Leisure Facilities", "Managed Care", "Marine Transportation",
    "Meat, Poultry & Dairy", "Media & Entertainment", "Medical Equipment & Supplies",
    "Metals & Mining", "Mortgage Finance", "Multiline and Specialty Retailers & Distributors",
    "Non-Alcoholic Beverages", "Oil & Gas – Exploration & Production",
    "Oil & Gas – Midstream", "Oil & Gas – Refining & Marketing",
    "Oil & Gas – Services",
    "Processed Foods", "Professional & Commercial Services",
    "Pulp & Paper Products", "Rail Transportation", "Real Estate",
    "Restaurants", "Road Transportation",
    "Security & Commodity Exchanges", "Semiconductors",
    "Software & IT Services", "Solar Technology & Project Developers",
    "Telecommunication Services", "Tobacco",
    "Waste Management", "Water Utilities & Services",
    "Wind Technology & Project Developers",
]
for _ind in _SASB_INDUSTRIES:
    DIRECT_URLS[f"SASB {_ind}.pdf"] = _SASB_CDN.format(slug=_sasb_slug(_ind))
del _ind

# GRI Sector Standards — direct media URLs
DIRECT_URLS.update({
    "GRI 11_ Oil and Gas Sector 2021.pdf":
        "https://www.globalreporting.org/pdf.ashx?id=12336",
    "GRI 12_ Coal Sector 2022.pdf":
        "https://www.globalreporting.org/pdf.ashx?id=13695",
    "GRI 13_ Agriculture Aquaculture and Fishing Sectors 2022.pdf":
        "https://www.globalreporting.org/pdf.ashx?id=15701",
    "GRI 14_ Mining Sector 2024 - English.pdf":
        "https://www.globalreporting.org/pdf.ashx?id=24499",
    # Joint/misc GRI docs
    "GRI SASB Joint 2021.pdf":
        "https://www.globalreporting.org/media/mlkjpn1i/gri-sasb-joint-publication-april-2021.pdf",
    "GRI Policymaker.pdf":
        "https://www.globalreporting.org/media/nmmnwfsm/gri-policymakers-guide.pdf",
    "GRI Standards 2021.pdf":
        "https://www.amauni.org/wp-content/uploads/2022/03/Set-of-GRI-Stnds-2021.pdf",
})

# ---------------------------------------------------------------------------
# Sources that need a free account (GRI / SASB / IFRS)
# After logging in with your browser, we grab the session cookie automatically.
# ---------------------------------------------------------------------------
AUTH_PORTALS = {
    "GRI":  {
        "login_url":  "https://www.globalreporting.org/",
        "cookie_domain": "globalreporting.org",
        # After login, GRI PDFs are at fixed paths; we pull them via the session.
        "note": "Register free at globalreporting.org, then re-run this script."
    },
    "SASB": {
        "login_url":  "https://sasb.ifrs.org/standards/download/",
        "cookie_domain": "sasb.ifrs.org",
        "note": "Register free at sasb.ifrs.org and log in, then re-run this script."
    },
    "IFRS": {
        "login_url":  "https://www.ifrs.org/",
        "cookie_domain": "ifrs.org",
        "note": "Register free at ifrs.org, then re-run this script."
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www.fsb-tcfd.org/",
}


def get_browser_cookies(domain: str) -> dict:
    """Try to extract cookies for *domain* from the user's default browser."""
    try:
        import browser_cookie3
    except ImportError:
        print("  [!] browser_cookie3 not installed — run: pip install browser-cookie3")
        return {}

    # Try Chrome first, then Firefox
    for loader in (browser_cookie3.chrome, browser_cookie3.firefox):
        try:
            jar = loader(domain_name=domain)
            cookies = {c.name: c.value for c in jar}
            if cookies:
                print(f"  [✓] Found {len(cookies)} cookies for {domain} from browser")
                return cookies
        except Exception:
            pass
    print(f"  [!] No browser cookies found for {domain} — make sure you are logged in.")
    return {}


def download_file(url: str, dest: Path, session: requests.Session,
                  retries: int = 3, delay: float = 2.0) -> bool:
    """Stream-download *url* to *dest*; skip if already fully downloaded."""
    existing_size = dest.stat().st_size if dest.exists() else 0

    for attempt in range(1, retries + 1):
        try:
            headers = {**HEADERS, "Range": f"bytes={existing_size}-"} if existing_size else HEADERS
            resp = session.get(url, headers=headers, stream=True, timeout=60)

            # 416 = range not satisfiable → file already complete
            if resp.status_code == 416:
                return True

            if resp.status_code not in (200, 206):
                print(f"  [!] HTTP {resp.status_code} for {url}")
                return False

            total = int(resp.headers.get("Content-Length", 0))
            is_resume = existing_size > 0 and resp.status_code == 206
            if existing_size and resp.status_code == 200:
                # Server ignored Range — restart fresh
                existing_size = 0
                dest.unlink(missing_ok=True)

            # Use a single iterator to avoid dual-consumption of the response socket
            chunks = resp.iter_content(chunk_size=65536)
            first_chunk = b""

            # On fresh downloads, peek the first bytes to reject HTML login pages
            if existing_size == 0:
                buf = b""
                for raw_chunk in chunks:
                    buf += raw_chunk
                    if len(buf) >= 512:
                        break
                first_chunk = buf
                stripped = first_chunk.lstrip(b"\xef\xbb\xbf\r\n ")
                if first_chunk and not stripped.startswith(b"%PDF"):
                    print(f"  [!] Got non-PDF response (login required?) for {url}")
                    return False

            mode = "ab" if is_resume else "wb"
            with open(dest, mode) as fh:
                with tqdm(
                    total=total or None,
                    initial=existing_size,
                    unit="B", unit_scale=True,
                    desc=dest.name[:45],
                    leave=False,
                ) as bar:
                    if first_chunk:
                        fh.write(first_chunk)
                        bar.update(len(first_chunk))
                    for chunk in chunks:
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
            return True

        except requests.RequestException as exc:
            print(f"  [!] Attempt {attempt}/{retries} failed: {exc}")
            time.sleep(delay * attempt)

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="datasets/ESGenius_w_ref_1136q.csv",
                   help="Path to the ESGenius CSV file")
    p.add_argument("--out", default="esgenius_pdfs",
                   help="Output folder (default: ./esgenius_pdfs)")
    p.add_argument("--family", choices=["IPCC", "GRI", "SASB", "IFRS", "TCFD", "CDP", "SDG"],
                   help="Download only one source family")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be downloaded without downloading")
    p.add_argument("--delay", type=float, default=1.5,
                   help="Seconds between requests (default: 1.5)")
    return p.parse_args()


def family_of(filename: str) -> str:
    if filename.startswith("IPCC"):             return "IPCC"
    if filename.startswith("GRI") or filename == "Management Relations 2016.pdf": return "GRI"
    if filename.startswith("SASB"):             return "SASB"
    if filename.startswith("IFRS"):             return "IFRS"
    if filename.startswith("TCFD"):             return "TCFD"
    if filename.startswith("CDP"):              return "CDP"
    return "SDG"


def main():
    args = parse_args()

    # Collect unique filenames from CSV
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    all_docs: set[str] = set()
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            all_docs.add(row['ref_doc'])

    # Filter by family if requested
    if args.family:
        all_docs = {d for d in all_docs if family_of(d) == args.family}

    # Separate into direct vs auth
    direct   = {d: DIRECT_URLS[d]   for d in all_docs if d in DIRECT_URLS}
    need_auth = {d for d in all_docs if d not in DIRECT_URLS}

    print(f"\n{'='*60}")
    print(f"  ESGenius PDF Downloader")
    print(f"  Total unique PDFs matched : {len(all_docs)}")
    print(f"  Direct download (no auth) : {len(direct)}")
    print(f"  Require registration      : {len(need_auth)}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("── DIRECT DOWNLOADS ──")
        for d, u in sorted(direct.items()):
            print(f"  {d}\n    → {u}")
        print("\n── NEED REGISTRATION ──")
        for d in sorted(need_auth):
            print(f"  {d}  [{family_of(d)}]")
        return

    out_dir = Path(args.out)

    # ── 1. Direct downloads ──────────────────────────────────────────────
    if direct:
        print("── Step 1: Direct downloads ──────────────────────────────")
        # IFRS.org requires login even for bypass=on URLs — grab cookies if available
        ifrs_cookies = get_browser_cookies("ifrs.org")
        gri_cookies  = get_browser_cookies("globalreporting.org")

        session = requests.Session()
        for filename, url in sorted(direct.items()):
            family = family_of(filename)
            dest_dir = out_dir / family
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / filename

            if dest.exists() and dest.stat().st_size > 50_000:
                with open(dest, "rb") as _f:
                    if _f.read(5) == b"%PDF-":
                        print(f"  [skip] {filename}")
                        continue
                dest.unlink()  # remove stale HTML login page

            # Use domain-appropriate cookies
            session.cookies.clear()
            if family == "IFRS" and ifrs_cookies:
                session.cookies.update(ifrs_cookies)
            elif family == "GRI" and gri_cookies:
                session.cookies.update(gri_cookies)

            print(f"  ↓ {filename}")
            ok = download_file(url, dest, session)
            if not ok:
                print(f"    [FAILED] — check URL manually: {url}")
            time.sleep(args.delay)

    # ── 2. Auth-required downloads ───────────────────────────────────────
    if need_auth:
        print("\n── Step 2: Registration-required downloads ───────────────")

        # Group by family
        by_family: dict[str, list[str]] = {}
        for d in need_auth:
            by_family.setdefault(family_of(d), []).append(d)

        for family, docs in sorted(by_family.items()):
            portal = AUTH_PORTALS.get(family, {})
            domain = portal.get("cookie_domain", "")
            note   = portal.get("note", "")

            print(f"\n  [{family}] — {len(docs)} files")
            if note:
                print(f"  → {note}")

            cookies = get_browser_cookies(domain) if domain else {}
            session = requests.Session()
            if cookies:
                session.cookies.update(cookies)
            else:
                print(f"  [skip] No session cookies — skipping {family}")
                print(f"         Log in at {portal.get('login_url','')} then re-run.")
                continue

            for filename in sorted(docs):
                dest_dir = out_dir / family
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / filename

                if dest.exists() and dest.stat().st_size > 50_000:
                    with open(dest, "rb") as _f:
                        if _f.read(5) == b"%PDF-":
                            print(f"  [skip] {filename}")
                            continue
                    dest.unlink()

                # Build a plausible URL for each source family
                # These are best-effort; if a URL 404s the file is skipped.
                url = build_auth_url(family, filename)
                if not url:
                    print(f"  [?] No URL known for {filename} — skipping")
                    continue

                print(f"  ↓ {filename}")
                ok = download_file(url, dest, session)
                if not ok:
                    print(f"    [FAILED] — try downloading manually from {portal.get('login_url','')}")
                time.sleep(args.delay)

    print("\n[Done]  Files saved to:", out_dir.resolve())


def build_auth_url(family: str, filename: str) -> str | None:
    """
    Construct a best-effort direct PDF URL for authenticated sources.
    These are derived from URL patterns observed on each site.
    They may change when sites update; treat as starting points.
    """
    if family == "GRI":
        # Known IDs are already in DIRECT_URLS; this fallback tries the media path.
        slug = (
            filename
            .replace(".pdf", "")
            .replace("GRI ", "")
            .replace("_ ", "-")
            .replace("_", "-")
            .replace(" ", "-")
            .lower()
        )
        return f"https://www.globalreporting.org/media/{slug}.pdf"

    if family == "SASB":
        # SASB standards are now hosted by IFRS Foundation.
        industry = (
            filename
            .replace("SASB ", "")
            .replace(".pdf", "")
            .replace(" & ", "-and-")
            .replace(" – ", "-")
            .replace(" ", "-")
            .lower()
        )
        return (
            f"https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards-sasb/"
            f"english/2023/issued/{industry}.pdf?bypass=on"
        )

    if family == "IFRS":
        # All known IFRS files are handled via DIRECT_URLS; this is a fallback.
        slug = (
            filename
            .replace(".pdf", "")
            .replace(" ", "-")
            .lower()
        )
        return f"https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards-issb/english/2023/issued/part-a/{slug}.pdf?bypass=on"

    return None


if __name__ == "__main__":
    main()