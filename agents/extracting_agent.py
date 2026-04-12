import logging
import os
from typing import List, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Tag, NavigableString
import requests

from ..utils import safe_requests_get, make_text_preview

logger = logging.getLogger(__name__)


class ExtractingAgent:
    """
    Extracting Agent - fetches content from various resources.

    Responsible for accessing and extracting text content from blogs, GitHub repos, etc.
    """

    def __init__(self, api_token: str = None):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.session.proxies = {}
        if api_token:
            self.session.headers.update({
                'Authorization': f'Bearer {api_token}'
            })
        for key in ('https_proxy', 'http_proxy'):
            val = os.getenv(key)
            if val:
                self.session.proxies[key] = val

    def extract_blog_content(self, blog_url: str) -> Optional[List[str]]:
        """Extract blog content and return structured paragraphs in markdown format."""
        try:
            resp = safe_requests_get(
                blog_url,
                timeout=30,
                max_retries=5,
                sleep_time=30,
                headers=self.session.headers,
                proxies=self.session.proxies or {},
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, 'html.parser')
            for tag in soup(["script", "style"]):
                tag.decompose()
            selectors = ['article', 'main', '.post-content', '.entry-content', '.article-content']
            main_content = next(
                (soup.select_one(sel) for sel in selectors if soup.select_one(sel)),
                None
            ) or soup.find('body') or soup
            headings = list(main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
            sections = []
            intro = []

            def process_table_markdown(table_element: Tag) -> str:
                table_rows = table_element.find_all('tr')
                if not table_rows:
                    return ""

                def cell_text(cell):
                    parts = []
                    for content in cell.contents:
                        if isinstance(content, Tag) and content.name == "a" and content.has_attr("href"):
                            link_text = content.get_text(strip=True)
                            link_url = content['href']
                            parts.append(f"[{link_text}]({link_url})" if link_text and link_url else link_text)
                        elif isinstance(content, Tag):
                            parts.append(content.get_text(separator='', strip=True))
                        else:
                            parts.append(str(content).strip())
                    return "".join(parts).replace('\n', ' ')

                header_cells = table_rows[0].find_all(['th', 'td'])
                table_md = (
                    "| " + " | ".join(map(cell_text, header_cells)) + " |\n" +
                    "|" + "|".join([" --- " for _ in header_cells]) + "|\n"
                ) if header_cells else ""
                for row in table_rows[1:]:
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        table_md += "| " + " | ".join(map(cell_text, cells)) + " |\n"
                return table_md.strip()

            def element_to_markdown(element: Tag) -> str:
                def render(elem):
                    if isinstance(elem, NavigableString):
                        return str(elem)
                    name = getattr(elem, "name", None)
                    if name == "table":
                        return "\n" + process_table_markdown(elem) + "\n"
                    if name == "a" and elem.has_attr("href"):
                        link_text = elem.get_text(strip=True)
                        link_url = elem["href"]
                        return f"[{link_text}]({link_url})"
                    if name:
                        return ''.join(render(child) for child in elem.children)
                    return str(elem)
                return render(element).strip()

            for el in main_content.children:
                if isinstance(el, NavigableString):
                    continue
                if getattr(el, "name", None) in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    break
                if getattr(el, "name", None) in ['p', 'div', 'section', 'article', 'ul', 'ol', 'blockquote', 'table']:
                    value = element_to_markdown(el)
                    if value:
                        intro.append(value)
            if intro:
                sections.append('\n\n'.join(intro))
            for idx, heading in enumerate(headings):
                text = heading.get_text(strip=True)
                if not text:
                    continue
                sec = [f"# {text}"]
                next_heading = headings[idx + 1] if idx + 1 < len(headings) else None
                s = heading.next_sibling
                buf = []
                while s is not None and not (isinstance(s, Tag) and s in headings):
                    if isinstance(s, NavigableString):
                        s = s.next_sibling
                        continue
                    if next_heading and s == next_heading:
                        break
                    if getattr(s, "name", None) in ['p', 'div', 'section', 'article', 'ul', 'ol', 'blockquote', 'table']:
                        txt = element_to_markdown(s)
                        if txt:
                            buf.append(txt)
                    s = s.next_sibling
                if buf:
                    sec.append('\n\n'.join(buf))
                if len(sec) > 1:
                    sections.append('\n\n'.join(sec))
                elif len(sec) == 1:
                    sections.append(sec[0])
            if not headings:
                blocks = []
                for el in main_content.find_all(
                    ['p', 'div', 'section', 'article', 'ul', 'ol', 'blockquote', 'table'],
                    recursive=False
                ):
                    value = element_to_markdown(el)
                    if value:
                        blocks.append(value)
                if blocks:
                    sections.append('\n\n'.join(blocks))
            res, seen = [], set()
            for p in sections:
                pst = p.strip()
                if pst and pst not in seen:
                    res.append(pst)
                    seen.add(pst)
            if res:
                logger.info(
                    f"Blog content preview from {blog_url}: "
                    f"{make_text_preview(res[0], max_len=400)}"
                )
            return res
        except Exception as e:
            logger.error(f"Error fetching blog content from {blog_url}: {str(e)}")
            return None

    def extract_github_readme(self, github_url: str) -> Optional[str]:
        """Extract GitHub README content."""
        try:
            if github_url.endswith('/'):
                github_url = github_url[:-1]

            parsed = urlparse(github_url)
            if parsed.netloc not in {"github.com", "www.github.com"}:
                return None

            path_parts = [part for part in parsed.path.strip('/').split('/') if part]
            if len(path_parts) < 2:
                return None

            owner = path_parts[0]
            repo = path_parts[1].removesuffix('.git')

            branches = []
            if len(path_parts) >= 4 and path_parts[2] in {"tree", "blob", "raw"}:
                branches.append(path_parts[3])
            branches.extend(["main", "master"])

            readme_urls = []
            seen_urls = set()
            for branch in branches:
                if not branch:
                    continue
                for readme_name in ("README.md", "readme.md", "README.MD", "Readme.md"):
                    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{readme_name}"
                    if raw_url not in seen_urls:
                        readme_urls.append(raw_url)
                        seen_urls.add(raw_url)

            for url in readme_urls:
                try:
                    response = safe_requests_get(
                        url, timeout=20, max_retries=5, sleep_time=30,
                        headers=self.session.headers, proxies=self.session.proxies or {}
                    )
                    content_type = (response.headers.get('content-type') or '').lower()
                    if response.status_code == 200 and 'text/html' not in content_type:
                        logger.info(
                            f"GitHub README preview from {github_url}: "
                            f"{make_text_preview(response.text, max_len=400)}"
                        )
                        return response.text
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Error fetching GitHub README from {github_url}: {str(e)}")
        return None
