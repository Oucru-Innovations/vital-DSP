from dash import html
from webapp.layout.header import header
from webapp.layout.sidebar import sidebar
from webapp.layout.main_content import main_content
from webapp.layout.footer import footer

layout = html.Div([header, sidebar, main_content, footer])
