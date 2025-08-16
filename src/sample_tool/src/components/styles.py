"""
CSS styles for the PPG analysis tool.
"""

APP_INDEX_STRING = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>PPG Filter Lab — Window Mode (Wide)</title>
    {%favicon%}
    {%css%}
    <style>
        body { 
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
            background:#0f172a; 
            color:#e2e8f0; 
            margin:0; 
        }
        .wrap { 
            max-width: 1760px; 
            margin: 24px auto; 
            padding: 0 16px; 
        }
        .header { 
            display:flex; 
            justify-content:space-between; 
            align-items:center; 
            margin-bottom:16px; 
        }
        .title { 
            font-size: 24px; 
            font-weight: 700; 
        }
        .subtle { 
            color:#94a3b8; 
        }
        /* Make the plots column wider */
        .grid { 
            display:grid; 
            grid-template-columns: 340px 2.1fr 0.9fr; 
            gap: 18px; 
        }
        .card { 
            background:#111827; 
            border:1px solid #1f2937; 
            border-radius:14px; 
            padding:14px;
            box-shadow: 0 6px 18px rgba(0,0,0,.35); 
        }
        .section-title { 
            font-size:13px; 
            text-transform:uppercase; 
            color:#93c5fd; 
            letter-spacing:.12em; 
            margin: 6px 0 10px; 
        }
        .row { 
            display:flex; 
            gap:10px; 
            align-items:center; 
            flex-wrap: wrap; 
        }
        label { 
            font-size: 12px; 
            color:#a5b4fc; 
        }
        .hint { 
            font-size: 12px; 
            color:#94a3b8; 
        }
        input, select { 
            background:#0b1220; 
            color:#e2e8f0; 
            border:1px solid #23324a; 
            border-radius:8px; 
            padding:8px 10px; 
        }
        .btn { 
            background:#2563eb; 
            border:none; 
            color:#fff; 
            padding:8px 12px; 
            border-radius:10px; 
            cursor:pointer; 
        }
        .btn.secondary { 
            background:#334155; 
        }
        .pill { 
            background:#0b1220; 
            border:1px solid #23324a; 
            border-radius:999px; 
            padding:4px 8px; 
            font-size:12px; 
        }
        .upload { 
            border: 1px dashed #334155; 
            padding: 12px; 
            border-radius: 12px; 
            text-align:center; 
            color:#94a3b8; 
        }
        .upload:hover { 
            background:#0b1220; 
        }
    </style>
</head>
<body>
    <div class="wrap">
        {%app_entry%}
    </div>
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""
