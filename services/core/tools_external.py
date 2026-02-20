import os, asyncio, httpx, smtplib, subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_core.tools import tool
from github import Github, GithubException
from duckduckgo_search import DDGS
from logging_config import get_logger

logger = get_logger(__name__)

@tool
def github_action(action: str, repo_name: str, path: str = "", content: str = "", branch: str = "main"):
    """Interact with GitHub."""
    token = os.getenv("GITHUB_TOKEN")
    if not token or token == "change_me": return "Error: GITHUB_TOKEN not set"
    try:
        g = Github(token)
        repo = g.get_user().get_repo(repo_name)
        if action == "read": return repo.get_contents(path, ref=branch).decoded_content.decode("utf-8")
        elif action == "create_file":
            try:
                repo.update_file(path, "Update", content, repo.get_contents(path).sha, branch=branch)
                return "Updated"
            except GithubException as e:
                logger.debug("github_file_not_found", error=str(e))
                repo.create_file(path, "Create", content, branch=branch)
                return "Created"
            except Exception as e:
                logger.error("github_update_error", error=str(e))
                return f"GitHub Error: {str(e)}"
        elif action == "create_issue": return f"Issue: {repo.create_issue(title=path, body=content).html_url}"
        return "Unknown action"
    except Exception as e: return f"GitHub Error: {e}"

@tool
def fast_search(query: str):
    """Quick text search (DuckDuckGo)."""
    try:
        return str(DDGS().text(query, max_results=5))
    except Exception as e: return f"Search Error: {e}"

@tool
async def send_email(to_email: str, subject: str, body: str, from_name: str = "AI_OS Assistant"):
    """
    Sends an email using SMTP configuration from environment variables.

    Requires environment variables:
    - SMTP_HOST: SMTP server address (e.g., smtp.gmail.com)
    - SMTP_PORT: SMTP port (e.g., 587)
    - SMTP_USER: SMTP username
    - SMTP_PASSWORD: SMTP password or app password
    - SMTP_FROM: From email address

    Example:
        send_email("cvostokov@gmail.com", "Test Subject", "Email body content")
    """
    try:
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        smtp_from = os.getenv("SMTP_FROM", smtp_user)

        if not smtp_user or not smtp_password:
            return "Error: SMTP_USER and SMTP_PASSWORD must be set in environment variables"

        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"{from_name} <{smtp_from}>"
        msg['To'] = to_email

        # Add body
        part = MIMEText(body, 'plain', 'utf-8')
        msg.attach(part)

        # Send email
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()

        return f"✅ Email sent successfully to {to_email}"

    except Exception as e:
        return f"❌ Email send failed: {str(e)}\n\nTip: For Gmail, create an app password at https://myaccount.google.com/apppasswords"

@tool
async def deep_web_search(query: str, max_results: int = 10):
    """
    Performs deep web search with multiple sources.

    Args:
        query: Search query
        max_results: Maximum number of results (default: 10)

    Returns:
        Formatted search results from multiple sources
    """
    try:
        results = DDGS().text(query, max_results=max_results)
        formatted = "🔍 SEARCH RESULTS:\n\n"

        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result.get('title', 'N/A')}\n"
            formatted += f"   URL: {result.get('href', 'N/A')}\n"
            formatted += f"   {result.get('body', '')[:200]}...\n\n"

        return formatted

    except Exception as e:
        return f"Search error: {str(e)}"

@tool
async def browse_and_extract(url: str):
    """
    Browse a website and extract main content.

    Args:
        url: Website URL to browse

    Returns:
        Extracted content from the website
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Try to get content through websurfer
            websurfer_url = os.getenv("WEBSURFER_URL", "http://websurfer:8003")
            response = await client.post(
                f"{websurfer_url}/visit",
                json={"url": url},
                timeout=60
            )
            data = response.json()

            if data.get("status") == "success":
                return f"📄 Content from {url}:\n\nTitle: {data.get('title')}\n\n{data.get('content', '')[:5000]}"
            else:
                return f"Error: {data.get('detail')}"

    except Exception as e:
        return f"Browsing error: {str(e)}"

@tool
async def generate_website(title: str, description: str, features: list = None):
    """
    Generates a complete website with HTML, CSS, and JavaScript.

    Args:
        title: Website title
        description: Website description/purpose
        features: List of features to include (optional)

    Returns:
        Generated website code
    """
    features = features or []

    html_code = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 2.5em;
        }}
        .description {{
            font-size: 1.2em;
            color: #666;
            margin-bottom: 30px;
        }}
        .feature {{
            background: #f8f9fa;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .feature h3 {{
            color: #764ba2;
            margin-bottom: 10px;
        }}
        button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px;
            transition: transform 0.2s;
        }}
        button:hover {{
            transform: scale(1.05);
        }}
        .reflection-form {{
            margin-top: 30px;
        }}
        textarea {{
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 150px;
        }}
        .reflection-list {{
            margin-top: 30px;
        }}
        .reflection-item {{
            background: #f0f0f0;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
        }}
        .reflection-date {{
            color: #999;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="description">{description}</p>

        {"".join([f'<div class="feature"><h3>{f}</h3><p>Feature description for {f}</p></div>' for f in features])}

        <div class="reflection-form">
            <h2>📝 Daily Reflection</h2>
            <textarea id="reflectionInput" placeholder="How are you feeling today? What did you accomplish? What are you grateful for?"></textarea>
            <button onclick="saveReflection()">Save Reflection</button>
        </div>

        <div class="reflection-list" id="reflectionList">
            <h2>💭 Your Reflections</h2>
            <!-- Reflections will be loaded here -->
        </div>
    </div>

    <script>
        // Save reflection to localStorage
        function saveReflection() {{
            const input = document.getElementById('reflectionInput');
            const text = input.value.trim();

            if (!text) {{
                alert('Please write something!');
                return;
            }}

            const reflection = {{
                id: Date.now(),
                text: text,
                date: new Date().toLocaleString('ru-RU')
            }};

            // Get existing reflections
            const reflections = JSON.parse(localStorage.getItem('reflections') || '[]');
            reflections.unshift(reflection);
            localStorage.setItem('reflections', JSON.stringify(reflections));

            // Clear input and reload
            input.value = '';
            loadReflections();
        }}

        // Load reflections from localStorage
        function loadReflections() {{
            const reflections = JSON.parse(localStorage.getItem('reflections') || '[]');
            const list = document.getElementById('reflectionList');

            if (reflections.length === 0) {{
                list.innerHTML = '<h2>💭 Your Reflections</h2><p>No reflections yet. Start your journey!</p>';
                return;
            }}

            let html = '<h2>💭 Your Reflections</h2>';
            reflections.forEach(r => {{
                html += `
                    <div class="reflection-item">
                        <div class="reflection-date">${{r.date}}</div>
                        <p>${{r.text}}</p>
                    </div>
                `;
            }});

            list.innerHTML = html;
        }}

        // Load on page load
        loadReflections();
    </script>
</body>
</html>"""

    return html_code

@tool
async def deploy_website(html_code: str, repo_name: str, path: str = "index.html"):
    """
    Deploys a website to GitHub Pages.

    Args:
        html_code: HTML code to deploy
        repo_name: GitHub repository name
        path: File path in repository (default: index.html)

    Returns:
        Deployment result
    """
    try:
        token = os.getenv("GITHUB_TOKEN")
        if not token or token == "change_me":
            return "Error: GITHUB_TOKEN not set. Please configure GitHub access."

        g = Github(token)
        repo = g.get_user().get_repo(repo_name)

        try:
            # Try to update existing file
            contents = repo.get_contents(path)
            repo.update_file(path, "Update website", html_code, contents.sha)
            action = "updated"
        except GithubException as e:
            logger.debug("github_file_not_found_for_deploy", error=str(e))
            # Create new file
            repo.create_file(path, "Create website", html_code)
            action = "created"
        except Exception as e:
            logger.error("github_deploy_error", error=str(e))
            return f"❌ Deployment failed: {str(e)}"

        # Get GitHub Pages URL
        pages_url = f"https://{repo.owner.login}.github.io/{repo_name}/"

        return f"✅ Website {action} successfully!\n\n🌐 Access it at: {pages_url}\n\nNote: GitHub Pages may take a few minutes to deploy."

    except Exception as e:
        return f"❌ Deployment failed: {str(e)}\n\nMake sure you have:\n1. Created a GitHub repository\n2. Enabled GitHub Pages in repository settings\n3. Set GITHUB_TOKEN environment variable"
