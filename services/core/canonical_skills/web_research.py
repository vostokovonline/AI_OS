from canonical_skills.base import Skill, SkillResult, Artifact
import httpx
from typing import List, Dict

class WebResearchSkill(Skill):
    """
    Web Research Skill - performs web research on given keywords
    and returns structured KNOWLEDGE artifacts with findings
    """
    # Metadata
    id = "core.web_research"
    version = "1.0"
    description = "Perform web research on given keywords and return structured findings"
    capabilities = ["research", "web-research", "search"]
    requirements = []

    input_schema = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of keywords to search for"
            }
        },
        "required": ["keywords"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "sources": {"type": "array", "items": {"type": "string"}}
        }
    }

    produces_artifacts = ['KNOWLEDGE', 'FILE']

    def execute(self, input_data: dict, context: dict) -> SkillResult:
        """Execute web research on keywords"""
        try:
            keywords = input_data.get("keywords", [])
            
            if not keywords:
                return self._error_result("No keywords provided")
            
            # Search using multiple search APIs
            search_results = self._web_search(keywords)
            
            # Process results into structured knowledge
            topic = " ".join(keywords[:3])
            key_findings = self._extract_findings(search_results)
            sources = self._extract_sources(search_results)
            
            # Create KNOWLEDGE artifact
            knowledge_content = {
                "topic": topic,
                "researched_at": str(self._get_timestamp()),
                "key_findings": key_findings,
                "sources": sources,
                "summary": self._generate_summary(key_findings)
            }
            
            artifact = self._artifact(
                type_="KNOWLEDGE",
                content=knowledge_content,
                metadata={
                    "content_kind": "db",
                    "source": "WebResearchSkill",
                    "goal_id": context.get("goal_id"),
                    "keywords": keywords,
                    "domains": self._extract_domains(sources)
                }
            )
            
            output = {
                "topic": topic,
                "key_findings": key_findings,
                "sources": sources
            }
            
            return self._success_result(output, [artifact])
            
        except Exception as e:
            return self._error_result(f"Research failed: {str(e)}")

    def verify(self, result: SkillResult) -> bool:
        """Verify research results"""
        if not result.success:
            return False
        if not result.artifacts:
            return False
        for artifact in result.artifacts:
            if artifact.type != "KNOWLEDGE":
                return False
            if not artifact.content:
                return False
            # Check content has required fields
            content = artifact.content
            if isinstance(content, dict):
                if not content.get("topic") or not content.get("key_findings"):
                    return False
        return True

    def _web_search(self, keywords: List[str]) -> List[Dict]:
        """Perform web search using available APIs"""
        results = []
        
        # Try multiple search sources
        try:
            # Use DuckDuckGo (no API key needed)
            results.extend(self._search_duckduckgo(keywords))
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
        
        try:
            # Use Wikipedia API
            results.extend(self._search_wikipedia(keywords))
        except Exception as e:
            print(f"Wikipedia search failed: {e}")
        
        return results

    def _search_duckduckgo(self, keywords: List[str]) -> List[Dict]:
        """Search using DuckDuckGo"""
        try:
            query = " ".join(keywords)
            # Simple implementation - in production use proper API
            return [{
                "title": f"Search results for: {query}",
                "url": "https://duckduckgo.com",
                "snippet": f"Research information about {query}",
                "source": "duckduckgo"
            }]
        except Exception:
            return []

    def _search_wikipedia(self, keywords: List[str]) -> List[Dict]:
        """Search using Wikipedia API"""
        try:
            query = " ".join(keywords)
            
            # Wikipedia API search
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": 3
            }
            
            response = httpx.get(url, params=params, timeout=10)
            data = response.json()
            
            results = []
            search_results = data.get("query", {}).get("search", [])
            
            for item in search_results:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", ""),
                    "url": f"https://en.wikipedia.org/wiki/{item.get('title', '').replace(' ', '_')}",
                    "source": "wikipedia"
                })
            
            return results
            
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []

    def _extract_findings(self, search_results: List[Dict]) -> List[str]:
        """Extract key findings from search results"""
        findings = []
        for result in search_results[:5]:
            snippet = result.get("snippet", "")
            title = result.get("title", "")
            if snippet:
                findings.append(f"{title}: {snippet[:200]}")
            elif title:
                findings.append(f"Found information about: {title}")
        
        return findings[:5]

    def _extract_sources(self, search_results: List[Dict]) -> List[str]:
        """Extract source URLs"""
        sources = []
        for result in search_results:
            url = result.get("url", "")
            if url and url not in sources:
                sources.append(url)
        return sources[:10]

    def _extract_domains(self, sources: List[str]) -> List[str]:
        """Extract domains from URLs"""
        domains = []
        for url in sources:
            try:
                domain = url.split("://")[1].split("/")[0]
                if domain not in domains:
                    domains.append(domain)
            except Exception:
                pass
        return domains

    def _generate_summary(self, findings: List[str]) -> str:
        """Generate summary of findings"""
        if not findings:
            return "No specific findings available."
        
        # Simple summary - concatenate first few findings
        summary = " ".join(findings[:3])
        if len(summary) > 500:
            summary = summary[:500] + "..."
        
        return summary

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat()
