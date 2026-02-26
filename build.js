const fs = require('fs');
const path = require('path');
const { marked } = require('marked');

// Read README
const readme = fs.readFileSync(path.join(__dirname, 'README.md'), 'utf-8');

// Convert to HTML
const content = marked.parse(readme);

const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ASR-Based Data Acquisition for Low-Resource Fongbe and Hausa</title>
  <style>
    :root {
      --bg: #ffffff;
      --text: #1a1a2e;
      --text-secondary: #4a4a6a;
      --accent: #2563eb;
      --accent-light: #dbeafe;
      --border: #e2e8f0;
      --code-bg: #f1f5f9;
      --table-stripe: #f8fafc;
      --success: #059669;
      --warning: #d97706;
    }

    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0f172a;
        --text: #e2e8f0;
        --text-secondary: #94a3b8;
        --accent: #60a5fa;
        --accent-light: #1e3a5f;
        --border: #334155;
        --code-bg: #1e293b;
        --table-stripe: #1e293b;
        --success: #34d399;
        --warning: #fbbf24;
      }
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
      line-height: 1.7;
      font-size: 16px;
    }

    .container {
      max-width: 860px;
      margin: 0 auto;
      padding: 2rem 1.5rem 4rem;
    }

    h1 {
      font-size: 1.9rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
      line-height: 1.3;
      letter-spacing: -0.02em;
    }

    h2 {
      font-size: 1.45rem;
      font-weight: 600;
      margin-top: 2.5rem;
      margin-bottom: 0.75rem;
      padding-bottom: 0.4rem;
      border-bottom: 2px solid var(--accent);
    }

    h3 {
      font-size: 1.15rem;
      font-weight: 600;
      margin-top: 1.8rem;
      margin-bottom: 0.5rem;
    }

    h4 {
      font-size: 1.05rem;
      font-weight: 600;
      margin-top: 1.4rem;
      margin-bottom: 0.4rem;
    }

    p { margin-bottom: 1rem; }

    a {
      color: var(--accent);
      text-decoration: none;
    }
    a:hover { text-decoration: underline; }

    blockquote {
      border-left: 4px solid var(--accent);
      background: var(--accent-light);
      padding: 0.8rem 1.2rem;
      margin: 1rem 0;
      border-radius: 0 6px 6px 0;
    }
    blockquote p { margin-bottom: 0; }

    code {
      font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
      font-size: 0.88em;
      background: var(--code-bg);
      padding: 0.15em 0.4em;
      border-radius: 4px;
    }

    pre {
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.2rem;
      overflow-x: auto;
      margin: 1rem 0;
    }
    pre code {
      background: none;
      padding: 0;
      font-size: 0.85em;
      line-height: 1.6;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0;
      font-size: 0.92em;
    }

    th, td {
      padding: 0.6rem 0.8rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }

    th {
      font-weight: 600;
      background: var(--code-bg);
      white-space: nowrap;
    }

    tr:nth-child(even) { background: var(--table-stripe); }
    tr:last-child td { border-bottom: 2px solid var(--border); }

    ul, ol {
      margin: 0.5rem 0 1rem 1.5rem;
    }
    li { margin-bottom: 0.3rem; }

    hr {
      border: none;
      border-top: 1px solid var(--border);
      margin: 2rem 0;
    }

    strong { font-weight: 600; }

    /* Table of Contents styling */
    h2#table-of-contents + ul {
      list-style: none;
      margin-left: 0;
      padding: 1rem;
      background: var(--code-bg);
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    h2#table-of-contents + ul li {
      padding: 0.2rem 0;
    }
    h2#table-of-contents + ul ul {
      list-style: none;
      margin-left: 1.2rem;
      margin-bottom: 0;
    }

    @media (max-width: 640px) {
      .container { padding: 1rem; }
      h1 { font-size: 1.5rem; }
      h2 { font-size: 1.25rem; }
      table { font-size: 0.82em; }
      th, td { padding: 0.4rem 0.5rem; }
    }
  </style>
</head>
<body>
  <div class="container">
    ${content}
  </div>
</body>
</html>`;

// Write output
const outDir = path.join(__dirname, 'public');
if (!fs.existsSync(outDir)) fs.mkdirSync(outDir);
fs.writeFileSync(path.join(outDir, 'index.html'), html);

console.log('Built index.html successfully');
