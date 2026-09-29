"""A script to construct a PEP503 compatible simple package index."""

__all__: "List[str]" = []
__author__ = "Gabriel Dorlhiac"

import os
import re
import sys
import urllib.parse
import requests
from typing import Any, Dict, List, Optional, Tuple

# --- Configuration ---
ORG: str = "slac-lcls"
LUTE_REPO: str = "lute"
OUTPUT_DIR: str = "wheels"

# --- Setup GitHub API Authentication ---
GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")
headers = {
    "Accept": "application/vnd.github.v3+json",
}
if GITHUB_TOKEN:
    headers["Authorization"] = f"token {GITHUB_TOKEN}"

session: requests.Session = requests.Session()
session.headers.update(headers)


def normalize_package_name(name: str) -> str:
    """Normalize package names according to PEP 503.

    Args:
        name (str): The retrieved package name.

    Returns:
        norm_name (str): The PEP 503 normalized name.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def fetch_org_repos(org: str) -> List[Dict[str, Any]]:
    """Retrieve all public/accessible repositories for the organization.

    Args:
        org (str): The organization name on GitHub to fetch repositories for.

    Returns:
        repos (List[Dict[str, Any]]): A list of repositories and metadata for the org.
    """
    repos: List[Dict[str, Any]] = []
    url: Optional[str] = f"https://api.github.com/orgs/{org}/repos?per_page=100"
    while url:
        response: requests.models.Response = session.get(url)
        if response.status_code != 200:
            print(
                f"Error fetching repos: {response.status_code} - {response.text}",
                file=sys.stderr,
            )
            break
        repos.extend(response.json())
        # Parse next page link header if pagination exists
        url = response.links.get("next", {}).get("url")
    return repos


def fetch_repo_releases(org: str, repo_name: str) -> List[Dict[str, Any]]:
    """Retrieve all releases and pre-releases for a given repository.

    Args:
        org (str): The organization name on GitHub.

        repo_name (str): The name of the organization's repo.

    Returns:
        releases (List[Dict[str, Any]]): A list of releases and metadata for the provided repo.
    """
    releases: List[Dict[str, Any]] = []
    url: Optional[str] = (
        f"https://api.github.com/repos/{org}/{repo_name}/releases?per_page=100"
    )
    while url:
        response: requests.models.Response = session.get(url)
        if response.status_code != 200:
            print(
                f"Error fetching releases for {repo_name}: {response.status_code}",
                file=sys.stderr,
            )
            break
        releases.extend(response.json())
        url = response.links.get("next", {}).get("url")
    return releases


def main() -> None:
    # Structure: index_data[normalized_package_name] = list of (filename, url)
    index_data: Dict[str, List[Tuple[str, str, Optional[str]]]] = {}

    print(f"Fetching repositories for organization '{ORG}'...")
    repos: List[Dict[str, str]] = fetch_org_repos(ORG)
    print(f"Found {len(repos)} repositories.")

    for repo in repos:
        repo_name: str = repo["name"]
        if repo_name != LUTE_REPO:
            continue

        print(f"Processing releases for {repo_name}...")
        releases: List[Dict[str, Any]] = fetch_repo_releases(ORG, repo_name)

        for release in releases:
            assets: List[Dict[str, Any]] = release.get("assets", [])
            for asset in assets:
                name: str = asset["name"]
                # Only index python packages (wheels, tarballs, zips)
                if name.endswith((".whl", ".tar.gz", ".zip", ".tar.bz2")):
                    # Get package name from wheel filename (e.g. ncarray-0.1.0-...)
                    # PEP 427: distribution-version-python-abi-platform.whl
                    pkg_name: str = name.split("-")[0]
                    norm_pkg_name: str = normalize_package_name(pkg_name)

                    download_url: str = asset["browser_download_url"]

                    # Add the SHA256 hash
                    digest: Optional[str] = asset.get("digest")
                    if digest and ":" in digest:
                        algo: str
                        hash_val: str
                        algo, hash_val = digest.split(":")
                        download_url = (
                            f"{download_url}#{algo.lower()}={hash_val.lower()}"
                        )

                    # Add the upload timestamp for pip/UV compat
                    # Turns out this cannot be used with PEP503. See below.
                    upload_time: Optional[str] = asset.get("created_at")

                    if norm_pkg_name not in index_data:
                        index_data[norm_pkg_name] = []

                    # Store package metadata
                    index_data[norm_pkg_name].append((name, download_url, upload_time))

    print("Generating static PEP 503 index pages...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write the package-level root index (wheels/index.html)
    root_index_path: str = os.path.join(OUTPUT_DIR, "index.html")
    with open(root_index_path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("  <title>LUTE Simple Index</title>\n")
        f.write("</head>\n<body>\n")
        f.write("  <h1>LUTE Simple Index</h1>\n")
        for pkg in sorted(index_data.keys()):
            # Sub-directory link must match the normalized name
            f.write(f'  <a href="{pkg}/">{pkg}</a><br/>\n')
        f.write("</body>\n</html>\n")

    # Write individual package pages (wheels/<pkg_name>/index.html)
    for pkg, files in index_data.items():
        pkg_dir: str = os.path.join(OUTPUT_DIR, pkg)
        os.makedirs(pkg_dir, exist_ok=True)

        pkg_index_path: str = os.path.join(pkg_dir, "index.html")
        with open(pkg_index_path, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n")
            f.write(f"  <title>{pkg} Releases</title>\n")
            f.write("</head>\n<body>\n")
            f.write(f"  <h1>{pkg} Releases</h1>\n")

            # Sort files by filename so older releases are at the top, newer at bottom
            for filename, url, upload_time in sorted(files, key=lambda x: x[0]):
                # Escaping URL characters just in case -- but have to handle
                # #sha256=... separately
                base_url: str
                fragment: Optional[str]
                if "#" in url:
                    base_url, fragment = url.split("#", 1)
                else:
                    base_url, fragment = url, None

                parsed_url: urllib.parse.ParseResult = urllib.parse.urlparse(base_url)
                safe_path: str = urllib.parse.quote(parsed_url.path, safe="/%")
                safe_url = parsed_url._replace(path=safe_path).geturl()
                if fragment:
                    safe_url = f"{safe_url}#{fragment}"

                # Can't put upload time in the URL... seem to need to provide a separate JSON endpoint
                # See PEP691 instead of PEP503
                upload_attr: str = (
                    f' data-upload-time="{upload_time}"' if upload_time else ""
                )
                f.write(f'  <a href="{safe_url}"{upload_attr}>{filename}</a><br/>\n')

            f.write("</body>\n</html>\n")

    print(f"Done! Static index files generated in '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
